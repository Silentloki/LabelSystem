"""Read-only helpers for importing a YOLO dataset into LabelSystem.

The GUI owns all mutations (copying images, writing ``jsons/*.json``, backing up
an existing project, and refreshing widgets).  This module deliberately does
none of those things.  It only inspects a dataset and returns the exact data
needed for a safe import.

Supported YOLO layout::

    dataset/
      data.yaml
      images/<split-or-subfolder>/image.png
      labels/<split-or-subfolder>/image.txt

Image/label matching is performed with the *relative path below* ``images``
and ``labels``.  For example, ``images/train/a/one.png`` matches
``labels/train/a/one.txt``.  This avoids confusing ``train/foo.png`` with
``val/foo.png`` while inspecting the source dataset.  LabelSystem currently
stores JSON files by stem, so records with colliding target stems are returned
but marked ``can_import=False`` instead of being silently overwritten.
"""

from __future__ import annotations

import ast
import json
import math
import re
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

try:  # Pillow is already used by the application, but keep this helper optional.
    from PIL import Image
except ImportError:  # pragma: no cover - depends on the packaged runtime
    Image = None  # type: ignore[assignment]


IMAGE_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"})

# These values match Utils.test.TestWindow.get_status_counts().
STATUS_UNLABELED = 0
STATUS_BAD = 1
STATUS_COMPLETE_GOOD = 2

_EPSILON = 1e-8
_CLASS_ID_RE = re.compile(r"^\d+$")

Issue = Dict[str, Any]
ProgressCallback = Callable[..., None]


def _issue(
    code: str,
    message: str,
    *,
    path: Optional[Union[str, Path]] = None,
    line: Optional[int] = None,
) -> Issue:
    """Create a serializable report item used by both the GUI and tests."""
    item: Issue = {"code": code, "message": message}
    if path is not None:
        item["path"] = str(path)
    if line is not None:
        item["line"] = int(line)
    return item


def _normalise_key(path: Path) -> str:
    """A Windows-safe relative key for matching paths/stems."""
    return path.as_posix().casefold()


def _empty_annotation_json(image_size: Optional[Tuple[int, int]] = None) -> Dict[str, Any]:
    width, height = image_size or (0, 0)
    return {
        "image_width": int(width or 0),
        "image_height": int(height or 0),
        "annotations": [],
    }


def _annotation(
    annotation_type: str,
    label: str,
    points: Sequence[Tuple[float, float]],
) -> Dict[str, Any]:
    """Return LabelSystem's current annotation shape (including ``lable`` typo)."""
    return {
        "type": annotation_type,
        "lable": label,
        "points": [{"x": float(x), "y": float(y)} for x, y in points],
    }


def _strip_yaml_comment(value: str) -> str:
    """Strip an unquoted ``#`` comment without needing a YAML dependency."""
    quote: Optional[str] = None
    escaped = False
    for index, char in enumerate(value):
        if quote:
            if quote == '"' and char == "\\" and not escaped:
                escaped = True
                continue
            if char == quote and not escaped:
                quote = None
            escaped = False
            continue
        if char in {"'", '"'}:
            quote = char
        elif char == "#":
            return value[:index].rstrip()
    return value.rstrip()


def _split_top_level(value: str, separator: str = ",") -> List[str]:
    """Split compact YAML lists/maps while preserving quoted commas."""
    pieces: List[str] = []
    start = 0
    quote: Optional[str] = None
    escaped = False
    square_depth = 0
    curly_depth = 0
    for index, char in enumerate(value):
        if quote:
            if quote == '"' and char == "\\" and not escaped:
                escaped = True
                continue
            if char == quote and not escaped:
                quote = None
            escaped = False
            continue
        if char in {"'", '"'}:
            quote = char
        elif char == "[":
            square_depth += 1
        elif char == "]":
            square_depth = max(0, square_depth - 1)
        elif char == "{":
            curly_depth += 1
        elif char == "}":
            curly_depth = max(0, curly_depth - 1)
        elif char == separator and square_depth == 0 and curly_depth == 0:
            pieces.append(value[start:index].strip())
            start = index + 1
    pieces.append(value[start:].strip())
    return pieces


def _split_unquoted_once(value: str, separator: str = ":") -> Optional[Tuple[str, str]]:
    quote: Optional[str] = None
    escaped = False
    for index, char in enumerate(value):
        if quote:
            if quote == '"' and char == "\\" and not escaped:
                escaped = True
                continue
            if char == quote and not escaped:
                quote = None
            escaped = False
            continue
        if char in {"'", '"'}:
            quote = char
        elif char == separator:
            return value[:index].strip(), value[index + 1 :].strip()
    return None


def _yaml_scalar(value: str) -> str:
    """Return a human-readable YAML scalar for a class name.

    Class names emitted by this project are simple strings.  The small parser
    also accepts quoted Chinese names and common compact YOLO ``names`` forms
    without bringing PyYAML into the packaged application.
    """
    value = _strip_yaml_comment(value).strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        try:
            parsed = ast.literal_eval(value)
        except (SyntaxError, ValueError):
            parsed = value[1:-1]
        return str(parsed).strip()
    return value.strip()


def _parse_nc(lines: Iterable[str]) -> Optional[int]:
    for raw_line in lines:
        stripped = _strip_yaml_comment(raw_line).strip()
        match = re.match(r"^nc\s*:\s*(.+)$", stripped)
        if not match:
            continue
        try:
            return int(_yaml_scalar(match.group(1)))
        except (TypeError, ValueError):
            return None
    return None


def _parse_inline_names(value: str, report: Dict[str, Any], path: Path) -> Optional[Dict[int, str]]:
    value = _strip_yaml_comment(value).strip()
    if value.startswith("[") and value.endswith("]"):
        inner = value[1:-1].strip()
        if not inner:
            return {}
        return {index: _yaml_scalar(item) for index, item in enumerate(_split_top_level(inner))}

    if value.startswith("{") and value.endswith("}"):
        inner = value[1:-1].strip()
        if not inner:
            return {}
        result: Dict[int, str] = {}
        for entry in _split_top_level(inner):
            pair = _split_unquoted_once(entry)
            if pair is None:
                report["errors"].append(
                    _issue("invalid_names_map", "data.yaml 的 names 映射格式无效。", path=path)
                )
                return None
            raw_index, raw_name = pair
            try:
                index = int(_yaml_scalar(raw_index))
            except ValueError:
                report["errors"].append(
                    _issue("invalid_class_index", f"类别索引无效：{raw_index!r}。", path=path)
                )
                return None
            result[index] = _yaml_scalar(raw_name)
        return result

    report["errors"].append(
        _issue("unsupported_names_syntax", "只支持 names 的列表、映射或缩进块写法。", path=path)
    )
    return None


def parse_yolo_data_yaml(data_yaml_path: Union[str, Path]) -> Dict[str, Any]:
    """Parse class names from a YOLO ``data.yaml`` without PyYAML.

    Returns ``{"categories": [...], "warnings": [...], "errors": [...]}``.
    Supported forms are the forms YOLO users encounter most often::

        names: [划伤, 脏污]
        names: {0: 划伤, 1: 脏污}
        names:
          0: 划伤
          1: 脏污
        names:
          - 划伤
          - 脏污
    """
    path = Path(data_yaml_path).expanduser()
    report: Dict[str, Any] = {"path": str(path), "categories": [], "warnings": [], "errors": []}
    if not path.is_file():
        report["errors"].append(_issue("missing_data_yaml", "未找到 data.yaml。", path=path))
        return report

    try:
        lines = path.read_text(encoding="utf-8-sig").splitlines()
    except (OSError, UnicodeError) as exc:
        report["errors"].append(
            _issue("unreadable_data_yaml", f"无法读取 data.yaml：{exc}", path=path)
        )
        return report

    names_map: Optional[Dict[int, str]] = None
    for line_index, raw_line in enumerate(lines):
        stripped_without_comment = _strip_yaml_comment(raw_line)
        match = re.match(r"^(\s*)names\s*:\s*(.*)$", stripped_without_comment)
        if not match:
            continue

        base_indent = len(match.group(1).expandtabs(4))
        inline_value = match.group(2).strip()
        if inline_value:
            names_map = _parse_inline_names(inline_value, report, path)
            break

        names_map = {}
        list_index = 0
        for child_line in lines[line_index + 1 :]:
            if not child_line.strip() or child_line.lstrip().startswith("#"):
                continue
            child_indent = len(child_line) - len(child_line.lstrip(" \t"))
            if child_indent <= base_indent:
                break
            child = _strip_yaml_comment(child_line).strip()
            if not child:
                continue
            if child.startswith("-"):
                raw_name = child[1:].strip()
                if not raw_name:
                    report["errors"].append(
                        _issue("empty_class_name", "names 列表中存在空类别名。", path=path)
                    )
                    continue
                names_map[list_index] = _yaml_scalar(raw_name)
                list_index += 1
                continue
            pair = _split_unquoted_once(child)
            if pair is None:
                report["errors"].append(
                    _issue("invalid_names_entry", f"无法解析 names 条目：{child!r}。", path=path)
                )
                continue
            raw_index, raw_name = pair
            try:
                class_index = int(_yaml_scalar(raw_index))
            except ValueError:
                report["errors"].append(
                    _issue("invalid_class_index", f"类别索引无效：{raw_index!r}。", path=path)
                )
                continue
            names_map[class_index] = _yaml_scalar(raw_name)
        break

    if names_map is None:
        report["errors"].append(
            _issue("missing_names", "data.yaml 中未找到 names 类别定义。", path=path)
        )
        return report

    if not names_map:
        report["errors"].append(_issue("empty_names", "data.yaml 的 names 不能为空。", path=path))
        return report

    invalid_indexes = [index for index in names_map if index < 0]
    if invalid_indexes:
        report["errors"].append(
            _issue("negative_class_index", "类别索引不能为负数。", path=path)
        )
    expected_indexes = list(range(max(names_map) + 1)) if names_map else []
    missing_indexes = [index for index in expected_indexes if index not in names_map]
    if missing_indexes:
        report["errors"].append(
            _issue(
                "non_contiguous_class_indexes",
                f"names 的类别索引必须从 0 连续编号，缺少：{missing_indexes}。",
                path=path,
            )
        )

    categories = [_yaml_scalar(names_map[index]) for index in expected_indexes if index in names_map]
    empty_names = [index for index, name in enumerate(categories) if not name]
    if empty_names:
        report["errors"].append(
            _issue("empty_class_name", f"names 中存在空类别名，索引：{empty_names}。", path=path)
        )

    nc = _parse_nc(lines)
    if nc is not None and nc != len(categories):
        report["warnings"].append(
            _issue(
                "nc_names_mismatch",
                f"data.yaml 的 nc={nc}，但 names 中有 {len(categories)} 个类别；将以 names 为准。",
                path=path,
            )
        )
    report["categories"] = categories
    return report


def load_yolo_categories(data_yaml_path: Union[str, Path]) -> List[str]:
    """Return only the ordered class names from a YOLO ``data.yaml``.

    ``parse_yolo_data_yaml`` is the diagnostic form used by the inspector.
    This lightweight helper deliberately returns a list because callers that
    only need the class lookup should not have to unpack a report object.
    Invalid/missing YAML therefore yields ``[]``; use the parser above when an
    error message is required.
    """
    return list(parse_yolo_data_yaml(data_yaml_path)["categories"])


def _parse_class_id(raw: str) -> Optional[int]:
    if not _CLASS_ID_RE.fullmatch(raw):
        return None
    try:
        return int(raw)
    except ValueError:  # Defensive only; regex already protects int().
        return None


def _parse_normalised_float(raw: str) -> Optional[float]:
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(value):
        return None
    return value


def _in_unit_interval(value: float) -> bool:
    return -_EPSILON <= value <= 1.0 + _EPSILON


def _clamp_unit(value: float) -> float:
    if -_EPSILON <= value < 0:
        return 0.0
    if 1.0 < value <= 1.0 + _EPSILON:
        return 1.0
    return float(value)


def _clean_polygon(points: Sequence[Tuple[float, float]]) -> List[Tuple[float, float]]:
    """Remove repeated adjacent points and the optional closing point."""
    cleaned: List[Tuple[float, float]] = []
    for point in points:
        if not cleaned or abs(point[0] - cleaned[-1][0]) > _EPSILON or abs(point[1] - cleaned[-1][1]) > _EPSILON:
            cleaned.append(point)
    if len(cleaned) > 1 and abs(cleaned[0][0] - cleaned[-1][0]) <= _EPSILON and abs(cleaned[0][1] - cleaned[-1][1]) <= _EPSILON:
        cleaned.pop()
    return cleaned


def _polygon_area_twice(points: Sequence[Tuple[float, float]]) -> float:
    return sum(
        point[0] * points[(index + 1) % len(points)][1]
        - points[(index + 1) % len(points)][0] * point[1]
        for index, point in enumerate(points)
    )


def _validate_polygon(points: Sequence[Tuple[float, float]]) -> Tuple[Optional[List[Tuple[float, float]]], Optional[str]]:
    cleaned = _clean_polygon(points)
    if len(cleaned) < 3:
        return None, "多边形至少需要 3 个不同顶点。"
    if abs(_polygon_area_twice(cleaned)) <= _EPSILON:
        return None, "多边形面积为 0。"
    return cleaned, None


def _read_image_size(image_path: Path) -> Tuple[Tuple[int, int], Optional[Issue]]:
    """Read dimensions when possible; dimensions are helpful but not required to import normalized YOLO labels."""
    if Image is None:
        return (0, 0), _issue(
            "image_size_unavailable",
            "当前运行环境没有 Pillow，已使用 0×0 图像尺寸；标注坐标仍保持归一化。",
            path=image_path,
        )
    try:
        with Image.open(image_path) as image:
            width, height = image.size
        return (int(width), int(height)), None
    except Exception as exc:  # Pillow has multiple format-specific exception classes.
        return (0, 0), _issue(
            "image_size_unavailable",
            f"无法读取图像尺寸：{exc}；标注坐标仍保持归一化。",
            path=image_path,
        )


def parse_yolo_label_file(
    label_path: Optional[Union[str, Path]],
    categories: Sequence[str],
    *,
    image_size: Optional[Tuple[int, int]] = None,
) -> Dict[str, Any]:
    """Convert one YOLO TXT file into LabelSystem's JSON annotation payload.

    The function is intentionally all-or-nothing for a non-empty TXT: if any
    line is invalid, ``can_import`` is false and the returned JSON has no
    annotations.  That prevents a GUI caller from accidentally importing only
    part of a bad label file.
    """
    annotation_json = _empty_annotation_json(image_size)
    result: Dict[str, Any] = {
        "source_label_path": str(label_path) if label_path is not None else None,
        "status": STATUS_UNLABELED,
        "label_state": "missing",
        "annotation_json": annotation_json,
        "annotations_count": 0,
        "line_count": 0,
        "warnings": [],
        "errors": [],
        "can_import": True,
    }
    if label_path is None:
        return result

    path = Path(label_path)
    if not path.is_file():
        return result

    try:
        raw_lines = path.read_text(encoding="utf-8-sig").splitlines()
    except (OSError, UnicodeError) as exc:
        result["status"] = STATUS_BAD
        result["label_state"] = "invalid"
        result["errors"].append(
            _issue("unreadable_label", f"无法读取标签文件：{exc}", path=path)
        )
        result["can_import"] = False
        return result

    nonempty_lines = [(line_number, line.strip()) for line_number, line in enumerate(raw_lines, start=1) if line.strip()]
    result["line_count"] = len(nonempty_lines)
    if not nonempty_lines:
        result["status"] = STATUS_COMPLETE_GOOD
        result["label_state"] = "empty"
        return result

    result["status"] = STATUS_BAD
    result["label_state"] = "annotated"
    annotations: List[Dict[str, Any]] = []

    for line_number, line in nonempty_lines:
        fields = line.split()
        if len(fields) < 5:
            result["errors"].append(
                _issue(
                    "invalid_yolo_line",
                    "YOLO 标签行至少需要类别和 4 个坐标值。",
                    path=path,
                    line=line_number,
                )
            )
            continue

        class_id = _parse_class_id(fields[0])
        if class_id is None:
            result["errors"].append(
                _issue(
                    "invalid_class_id",
                    f"类别编号必须是非负整数：{fields[0]!r}。",
                    path=path,
                    line=line_number,
                )
            )
            continue
        if class_id >= len(categories):
            result["errors"].append(
                _issue(
                    "class_id_out_of_range",
                    f"类别编号 {class_id} 超出 data.yaml names 范围（0 到 {max(len(categories) - 1, 0)}）。",
                    path=path,
                    line=line_number,
                )
            )
            continue

        numbers = [_parse_normalised_float(value) for value in fields[1:]]
        if any(value is None for value in numbers):
            result["errors"].append(
                _issue(
                    "invalid_coordinate",
                    "标签行包含非数字或非有限坐标。",
                    path=path,
                    line=line_number,
                )
            )
            continue
        coordinates = [float(value) for value in numbers if value is not None]

        if len(fields) == 5:
            center_x, center_y, width, height = coordinates
            if not all(_in_unit_interval(value) for value in coordinates):
                result["errors"].append(
                    _issue(
                        "coordinate_out_of_range",
                        "矩形 YOLO 坐标必须在 0 到 1 之间。",
                        path=path,
                        line=line_number,
                    )
                )
                continue
            if width <= _EPSILON or height <= _EPSILON:
                result["errors"].append(
                    _issue(
                        "invalid_rectangle_size",
                        "矩形的宽和高必须大于 0。",
                        path=path,
                        line=line_number,
                    )
                )
                continue
            x1 = center_x - width / 2.0
            y1 = center_y - height / 2.0
            x2 = center_x + width / 2.0
            y2 = center_y + height / 2.0
            if not all(_in_unit_interval(value) for value in (x1, y1, x2, y2)):
                result["errors"].append(
                    _issue(
                        "rectangle_outside_image",
                        "矩形边界超出图像的归一化范围。",
                        path=path,
                        line=line_number,
                    )
                )
                continue
            annotations.append(
                _annotation(
                    "rect",
                    str(categories[class_id]),
                    [(_clamp_unit(x1), _clamp_unit(y1)), (_clamp_unit(x2), _clamp_unit(y2))],
                )
            )
            continue

        if len(coordinates) % 2 != 0:
            result["errors"].append(
                _issue(
                    "invalid_polygon_coordinate_count",
                    "多边形标签必须由成对的 x y 坐标组成。",
                    path=path,
                    line=line_number,
                )
            )
            continue
        if len(coordinates) < 6:
            result["errors"].append(
                _issue(
                    "invalid_polygon_vertex_count",
                    "多边形至少需要 3 个顶点。",
                    path=path,
                    line=line_number,
                )
            )
            continue
        if not all(_in_unit_interval(value) for value in coordinates):
            result["errors"].append(
                _issue(
                    "coordinate_out_of_range",
                    "多边形 YOLO 坐标必须在 0 到 1 之间。",
                    path=path,
                    line=line_number,
                )
            )
            continue
        polygon_points = [
            (_clamp_unit(coordinates[index]), _clamp_unit(coordinates[index + 1]))
            for index in range(0, len(coordinates), 2)
        ]
        polygon_points, polygon_error = _validate_polygon(polygon_points)
        if polygon_error:
            result["errors"].append(
                _issue("invalid_polygon", polygon_error, path=path, line=line_number)
            )
            continue
        annotations.append(_annotation("polygon", str(categories[class_id]), polygon_points or []))

    if result["errors"]:
        result["label_state"] = "invalid"
        result["can_import"] = False
        # Do not expose a tempting partial result to a caller that writes JSON.
        result["annotation_json"] = _empty_annotation_json(image_size)
        return result

    result["annotation_json"]["annotations"] = annotations
    result["annotations_count"] = len(annotations)
    return result


def _normalise_json_point(
    point: Any,
    width: int,
    height: int,
) -> Optional[Tuple[float, float]]:
    if not isinstance(point, Mapping) or "x" not in point or "y" not in point:
        return None
    x = _parse_normalised_float(str(point["x"]))
    y = _parse_normalised_float(str(point["y"]))
    if x is None or y is None:
        return None
    if not _in_unit_interval(x) or not _in_unit_interval(y):
        if width > 0 and height > 0:
            x /= width
            y /= height
        else:
            return None
    if not _in_unit_interval(x) or not _in_unit_interval(y):
        return None
    return _clamp_unit(x), _clamp_unit(y)


def _normalise_json_dimension(value: Any) -> int:
    try:
        dimension = int(value)
    except (TypeError, ValueError):
        return 0
    return dimension if dimension > 0 else 0


def parse_json_annotation_file(annotation_path: Union[str, Path]) -> Dict[str, Any]:
    """Validate/normalise a LabelSystem JSON annotation file for ``导入标注``.

    It accepts legacy category keys (``label``/``category``) but always writes
    the compatible ``lable`` key in ``annotation_json``.  Pixel coordinates are
    normalised when valid image dimensions are present; otherwise annotations
    must already use normalized 0--1 coordinates.
    """
    path = Path(annotation_path).expanduser()
    result: Dict[str, Any] = {
        "source_annotation_path": str(path),
        "annotation_json": None,
        "categories": [],
        "warnings": [],
        "errors": [],
        "can_import": False,
    }
    if not path.is_file():
        result["errors"].append(_issue("missing_json_annotation", "未找到 JSON 标注文件。", path=path))
        return result
    try:
        raw_data = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        result["errors"].append(_issue("invalid_json_annotation", f"无法读取 JSON 标注：{exc}", path=path))
        return result
    if not isinstance(raw_data, Mapping):
        result["errors"].append(_issue("invalid_json_root", "JSON 标注根节点必须是对象。", path=path))
        return result

    width = _normalise_json_dimension(raw_data.get("image_width", raw_data.get("width", 0)))
    height = _normalise_json_dimension(raw_data.get("image_height", raw_data.get("height", 0)))
    raw_annotations = raw_data.get("annotations", [])
    if not isinstance(raw_annotations, list):
        result["errors"].append(_issue("invalid_json_annotations", "annotations 必须是数组。", path=path))
        return result

    annotations: List[Dict[str, Any]] = []
    categories: List[str] = []
    known_categories = set()
    for index, raw_annotation in enumerate(raw_annotations, start=1):
        if not isinstance(raw_annotation, Mapping):
            result["errors"].append(
                _issue("invalid_json_annotation", "标注项必须是对象。", path=path, line=index)
            )
            continue
        annotation_type = str(raw_annotation.get("type", "")).strip().lower()
        if annotation_type not in {"rect", "polygon"}:
            result["errors"].append(
                _issue("unsupported_annotation_type", f"不支持的标注类型：{annotation_type!r}。", path=path, line=index)
            )
            continue
        label = str(
            raw_annotation.get("lable")
            or raw_annotation.get("label")
            or raw_annotation.get("category")
            or ""
        ).strip()
        if not label:
            result["errors"].append(
                _issue("missing_annotation_label", "标注缺少类别名称。", path=path, line=index)
            )
            continue
        raw_points = raw_annotation.get("points", [])
        if not isinstance(raw_points, list):
            result["errors"].append(
                _issue("invalid_json_points", "points 必须是数组。", path=path, line=index)
            )
            continue
        points = [_normalise_json_point(point, width, height) for point in raw_points]
        if any(point is None for point in points):
            result["errors"].append(
                _issue("invalid_json_point", "标注坐标无效或超出范围。", path=path, line=index)
            )
            continue
        normalised_points = [point for point in points if point is not None]
        if annotation_type == "rect":
            if len(normalised_points) < 2:
                result["errors"].append(
                    _issue("invalid_json_rectangle", "矩形至少需要两个点。", path=path, line=index)
                )
                continue
            x_values = [normalised_points[0][0], normalised_points[1][0]]
            y_values = [normalised_points[0][1], normalised_points[1][1]]
            if max(x_values) - min(x_values) <= _EPSILON or max(y_values) - min(y_values) <= _EPSILON:
                result["errors"].append(
                    _issue("invalid_json_rectangle", "矩形宽和高必须大于 0。", path=path, line=index)
                )
                continue
            normalised_points = [(min(x_values), min(y_values)), (max(x_values), max(y_values))]
        else:
            normalised_points, polygon_error = _validate_polygon(normalised_points)
            if polygon_error:
                result["errors"].append(
                    _issue("invalid_json_polygon", polygon_error, path=path, line=index)
                )
                continue
        annotations.append(_annotation(annotation_type, label, normalised_points or []))
        if label not in known_categories:
            known_categories.add(label)
            categories.append(label)

    if result["errors"]:
        return result
    payload = _empty_annotation_json((width, height))
    payload["annotations"] = annotations
    result["annotation_json"] = payload
    result["categories"] = categories
    result["can_import"] = True
    return result


def _emit_progress(
    callback: Optional[ProgressCallback],
    phase: str,
    current: int,
    total: int,
    path: Optional[Path] = None,
) -> None:
    """Emit a small serializable event, accepting common callback signatures.

    Preferred signature is ``callback(event_dict)``.  A four-argument callback
    ``callback(phase, current, total, path_string)`` is also accepted so a Qt
    caller can update a progress bar without wrapping the event.
    """
    if callback is None:
        return
    event = {
        "phase": phase,
        "current": int(current),
        "total": int(total),
        "path": str(path) if path is not None else None,
    }
    try:
        callback(event)
    except TypeError:
        callback(phase, current, total, str(path) if path is not None else None)


def _new_dataset_result(root: Path) -> Dict[str, Any]:
    return {
        "dataset_root": str(root),
        "data_yaml_path": str(root / "data.yaml"),
        "images_root": str(root / "images"),
        "labels_root": str(root / "labels"),
        "categories": [],
        "records": [],
        "fatal_errors": [],
        "warnings": [],
        "errors": [],
        "summary": {
            "images_found": 0,
            "records": 0,
            "ready_records": 0,
            "unlabeled": 0,
            "bad": 0,
            "complete_good": 0,
            "invalid_records": 0,
            "invalid_label_lines": 0,
            "stem_collisions": 0,
            "orphan_label_files": 0,
            "ignored_image_files": 0,
        },
        "can_import": False,
    }


def _add_fatal(result: Dict[str, Any], issue: Issue) -> None:
    result["fatal_errors"].append(issue)
    result["errors"].append(issue)


def _relative_label_key(labels_root: Path, label_path: Path) -> str:
    return _normalise_key(label_path.relative_to(labels_root).with_suffix(""))


def _relative_image_key(images_root: Path, image_path: Path) -> str:
    return _normalise_key(image_path.relative_to(images_root).with_suffix(""))


def inspect_yolo_dataset(
    dataset_root: Union[str, Path],
    progress_callback: Optional[ProgressCallback] = None,
) -> Dict[str, Any]:
    """Inspect a YOLO dataset and return a no-write import plan for the GUI.

    ``records`` is the integration point for the GUI.  For every supported
    source image it contains:

    * ``source_image_path`` / ``source_image_relpath`` for copying;
    * ``source_label_path`` (or ``None``) and its source state;
    * ``target_filename`` / ``target_stem`` for conflict checks;
    * LabelSystem's ``annotation_json`` payload;
    * status ``0`` (missing TXT), ``1`` (non-empty TXT), or ``2`` (empty TXT);
    * per-record ``warnings``, ``errors``, and ``can_import``.

    The function does not copy, create, delete, or overwrite any file.  A bad
    record does not stop valid sibling records from being returned; only
    structural errors are stored in ``fatal_errors``.
    """
    root = Path(dataset_root).expanduser()
    try:
        root = root.resolve()
    except OSError:
        root = root.absolute()
    result = _new_dataset_result(root)
    if not root.is_dir():
        _add_fatal(result, _issue("missing_dataset_root", "选择的数据集目录不存在。", path=root))
        return result

    data_yaml_path = root / "data.yaml"
    yaml_report = parse_yolo_data_yaml(data_yaml_path)
    result["categories"] = list(yaml_report["categories"])
    result["warnings"].extend(yaml_report["warnings"])
    if yaml_report["errors"]:
        for issue in yaml_report["errors"]:
            _add_fatal(result, issue)

    images_root = root / "images"
    labels_root = root / "labels"
    if not images_root.is_dir():
        _add_fatal(result, _issue("missing_images_directory", "数据集缺少 images 目录。", path=images_root))
    if not labels_root.is_dir():
        _add_fatal(result, _issue("missing_labels_directory", "数据集缺少 labels 目录。", path=labels_root))
    if not images_root.is_dir() or not labels_root.is_dir():
        return result

    all_image_files = sorted((path for path in images_root.rglob("*") if path.is_file()), key=lambda item: item.as_posix().casefold())
    image_files = [path for path in all_image_files if path.suffix.casefold() in IMAGE_EXTENSIONS]
    ignored_images = [path for path in all_image_files if path.suffix.casefold() not in IMAGE_EXTENSIONS]
    result["summary"]["images_found"] = len(image_files)
    result["summary"]["ignored_image_files"] = len(ignored_images)
    for path in ignored_images:
        result["warnings"].append(
            _issue(
                "unsupported_image_extension",
                f"已跳过不支持的图像文件：{path.name}。",
                path=path,
            )
        )
    if not image_files:
        _add_fatal(result, _issue("no_supported_images", "images 目录中没有可导入的图像。", path=images_root))
        return result

    labels_by_key: Dict[str, Path] = {}
    label_keys_with_collisions = set()
    if labels_root.is_dir():
        label_files = sorted((path for path in labels_root.rglob("*.txt") if path.is_file()), key=lambda item: item.as_posix().casefold())
        for label_path in label_files:
            key = _relative_label_key(labels_root, label_path)
            existing = labels_by_key.get(key)
            if existing is not None:
                label_keys_with_collisions.add(key)
                result["warnings"].append(
                    _issue(
                        "duplicate_label_path",
                        "多个 TXT 标签映射到同一相对路径，相关标签将不会自动导入。",
                        path=label_path,
                    )
                )
            else:
                labels_by_key[key] = label_path

    image_keys = {_relative_image_key(images_root, image_path) for image_path in image_files}
    if labels_root.is_dir():
        for key, label_path in labels_by_key.items():
            if key not in image_keys:
                result["summary"]["orphan_label_files"] += 1
                result["warnings"].append(
                    _issue(
                        "orphan_label_file",
                        "找到了没有对应图像的 TXT 标签，已跳过。",
                        path=label_path,
                    )
                )

    _emit_progress(progress_callback, "scan", 0, len(image_files), None)
    by_target_stem: Dict[str, List[Dict[str, Any]]] = {}
    for current, image_path in enumerate(image_files, start=1):
        relative_path = image_path.relative_to(images_root)
        relative_key = _relative_image_key(images_root, image_path)
        label_path = labels_by_key.get(relative_key)
        image_size, image_size_warning = _read_image_size(image_path)
        label_result = parse_yolo_label_file(label_path, result["categories"], image_size=image_size)
        record: Dict[str, Any] = {
            "source_image_path": str(image_path),
            "source_image_relpath": relative_path.as_posix(),
            "source_label_path": str(label_path) if label_path is not None else None,
            "source_label_relpath": (
                label_path.relative_to(labels_root).as_posix() if label_path is not None and labels_root.is_dir() else None
            ),
            "target_filename": image_path.name,
            "target_stem": image_path.stem,
            "status": label_result["status"],
            "label_state": label_result["label_state"],
            "annotation_json": label_result["annotation_json"],
            "annotations_count": label_result["annotations_count"],
            "warnings": list(label_result["warnings"]),
            "errors": list(label_result["errors"]),
            "can_import": bool(label_result["can_import"]),
        }
        if image_size_warning is not None:
            record["warnings"].append(image_size_warning)
        if relative_key in label_keys_with_collisions:
            record["errors"].append(
                _issue(
                    "duplicate_label_path",
                    "多个 TXT 标签映射到这张图像，无法安全确定要导入的标签。",
                    path=image_path,
                )
            )
            record["can_import"] = False
            record["label_state"] = "invalid"
            record["annotation_json"] = _empty_annotation_json(image_size)

        result["records"].append(record)
        by_target_stem.setdefault(image_path.stem.casefold(), []).append(record)
        result["warnings"].extend(record["warnings"])
        result["errors"].extend(record["errors"])
        _emit_progress(progress_callback, "parse", current, len(image_files), image_path)

    for colliding_records in by_target_stem.values():
        if len(colliding_records) < 2:
            continue
        result["summary"]["stem_collisions"] += 1
        source_paths = [record["source_image_relpath"] for record in colliding_records]
        for record in colliding_records:
            record["errors"].append(
                _issue(
                    "target_stem_collision",
                    "多个源图像会写入相同的 jsons/<stem>.json：" + ", ".join(source_paths),
                    path=record["source_image_path"],
                )
            )
            record["can_import"] = False
            result["errors"].append(record["errors"][-1])

    for record in result["records"]:
        if record["status"] == STATUS_UNLABELED:
            result["summary"]["unlabeled"] += 1
        elif record["status"] == STATUS_BAD:
            result["summary"]["bad"] += 1
        elif record["status"] == STATUS_COMPLETE_GOOD:
            result["summary"]["complete_good"] += 1
        if record["can_import"]:
            result["summary"]["ready_records"] += 1
        else:
            result["summary"]["invalid_records"] += 1
        result["summary"]["invalid_label_lines"] += sum(
            1 for issue in record["errors"] if issue.get("line") is not None
        )
    result["summary"]["records"] = len(result["records"])
    result["can_import"] = not result["fatal_errors"] and result["summary"]["ready_records"] > 0
    _emit_progress(progress_callback, "complete", len(image_files), len(image_files), None)
    return result


class AnnotationImporter:
    """GUI-facing compatibility facade around the read-only module functions.

    The original window integration uses a small class with static methods.
    Keeping that shape here means the GUI can remain intentionally ignorant of
    report internals, while module-level functions remain convenient for tests
    and future non-GUI callers.
    """

    @staticmethod
    def inspect_yolo_dataset(
        dataset_root: Union[str, Path],
        progress_callback: Optional[ProgressCallback] = None,
    ) -> Dict[str, Any]:
        inspection = inspect_yolo_dataset(dataset_root, progress_callback=progress_callback)
        entries: List[Dict[str, Any]] = []
        for record in inspection["records"]:
            # Keep the explicit source fields as well as these concise aliases.
            # The aliases are what Utils.test's import UI consumes.
            record["image_path"] = record["source_image_path"]
            record["stem"] = record["target_stem"]
            record["json_data"] = record["annotation_json"]
            entries.append(record)
        inspection["entries"] = entries
        inspection["class_names"] = list(inspection["categories"])
        inspection["names"] = list(inspection["categories"])
        return inspection

    @staticmethod
    def parse_json_annotation_file(annotation_path: Union[str, Path]) -> Dict[str, Any]:
        """Return the validated JSON payload expected by the annotation UI.

        The underlying module function returns a rich diagnostics report.  The
        GUI needs just a JSON payload and already catches exceptions, so expose
        invalid source files as a helpful ``ValueError`` here.
        """
        parsed = parse_json_annotation_file(annotation_path)
        if not parsed["can_import"] or not isinstance(parsed["annotation_json"], dict):
            details = "; ".join(str(issue.get("message", issue)) for issue in parsed["errors"])
            raise ValueError(details or "JSON 标注格式无效。")
        return parsed["annotation_json"]


__all__ = [
    "IMAGE_EXTENSIONS",
    "STATUS_UNLABELED",
    "STATUS_BAD",
    "STATUS_COMPLETE_GOOD",
    "parse_yolo_data_yaml",
    "load_yolo_categories",
    "parse_yolo_label_file",
    "parse_json_annotation_file",
    "inspect_yolo_dataset",
    "AnnotationImporter",
]
