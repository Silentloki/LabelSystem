"""Contract tests for the pure YOLO-dataset inspection service.

The UI turns the records returned by ``inspect_yolo_dataset`` into a LabelSystem
project.  Keeping this test independent of PyQt makes the conversion rules easy
to validate without launching the application.
"""

from __future__ import annotations

import struct
import shutil
import unittest
import uuid
import zlib
import json
from pathlib import Path

from Utils.AnnotationImporter import (
    inspect_yolo_dataset,
    load_yolo_categories,
    parse_json_annotation_file,
)
from Utils.test import TestWindow


def _png_chunk(kind: bytes, payload: bytes) -> bytes:
    return (
        struct.pack(">I", len(payload))
        + kind
        + payload
        + struct.pack(">I", zlib.crc32(kind + payload) & 0xFFFFFFFF)
    )


def _write_png(path: Path, width: int = 32, height: int = 16) -> None:
    """Write a small valid RGB PNG using only the standard library."""
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = b"".join(b"\x00" + (b"\x7f\x7f\x7f" * width) for _ in range(height))
    content = (
        b"\x89PNG\r\n\x1a\n"
        + _png_chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + _png_chunk(b"IDAT", zlib.compress(raw))
        + _png_chunk(b"IEND", b"")
    )
    path.write_bytes(content)


def _record_for_filename(result: dict, filename: str) -> dict:
    matches = [
        record
        for record in result["records"]
        if Path(record["source_image_path"]).name == filename
    ]
    if len(matches) != 1:
        raise AssertionError(f"Expected one record for {filename!r}, got {len(matches)}")
    return matches[0]


class AnnotationImporterTest(unittest.TestCase):
    def setUp(self) -> None:
        # The sandbox may forbid the system Temp directory, so keep all test
        # fixtures under the workspace and remove each generated directory.
        self.temp_base = Path(__file__).resolve().parent / ".tmp_test_workspace"
        self.temp_base.mkdir(exist_ok=True)
        # Do not use tempfile.TemporaryDirectory here: on Windows it creates a
        # directory with a restrictive ACL that the sandboxed Python process
        # cannot use for nested fixture files.
        self.temp_root = self.temp_base / f"annotation-importer-{uuid.uuid4().hex}"
        self.temp_root.mkdir()
        self.dataset_root = self.temp_root / "yolo_dataset"
        self.dataset_root.mkdir()
        (self.dataset_root / "data.yaml").write_text(
            "train: images/train\n"
            "val: images/val\n\n"
            "nc: 2\n"
            "names:\n"
            "  0: 划伤\n"
            "  1: 脏污\n",
            encoding="utf-8",
        )

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_root, ignore_errors=False)
        try:
            self.temp_base.rmdir()
        except OSError:
            # Do not remove a shared directory if another test is still using it.
            pass

    def _image(self, split: str, filename: str) -> Path:
        image_path = self.dataset_root / "images" / split / filename
        _write_png(image_path)
        return image_path

    def _label(self, split: str, filename: str, content: str) -> Path:
        label_path = self.dataset_root / "labels" / split / filename
        label_path.parent.mkdir(parents=True, exist_ok=True)
        label_path.write_text(content, encoding="utf-8")
        return label_path

    def test_inspects_chinese_categories_rectangles_polygons_and_sample_statuses(self) -> None:
        """The basic YOLO layout maps losslessly to LabelSystem's JSON schema."""
        self._image("train", "rect.png")
        self._label("train", "rect.txt", "0 0.5 0.25 0.4 0.2\n")

        self._image("train", "polygon.png")
        self._label("train", "polygon.txt", "1 0.1 0.2 0.9 0.2 0.6 0.8\n")

        self._image("val", "empty.png")
        self._label("val", "empty.txt", "")

        self._image("val", "missing.png")

        self.assertEqual(load_yolo_categories(self.dataset_root / "data.yaml"), ["划伤", "脏污"])
        result = inspect_yolo_dataset(self.dataset_root)
        self.assertEqual(result["categories"], ["划伤", "脏污"])
        self.assertEqual(len(result["records"]), 4)
        self.assertFalse(result["errors"])

        rect = _record_for_filename(result, "rect.png")
        self.assertTrue(rect["can_import"])
        self.assertEqual(rect["status"], 1)
        self.assertEqual(rect["annotation_json"]["image_width"], 32)
        self.assertEqual(rect["annotation_json"]["image_height"], 16)
        self.assertEqual(
            rect["annotation_json"]["annotations"],
            [
                {
                    "type": "rect",
                    "lable": "划伤",
                    "points": [
                        {"x": 0.3, "y": 0.15},
                        {"x": 0.7, "y": 0.35},
                    ],
                }
            ],
        )

        polygon = _record_for_filename(result, "polygon.png")
        self.assertTrue(polygon["can_import"])
        self.assertEqual(polygon["status"], 1)
        self.assertEqual(
            polygon["annotation_json"]["annotations"],
            [
                {
                    "type": "polygon",
                    "lable": "脏污",
                    "points": [
                        {"x": 0.1, "y": 0.2},
                        {"x": 0.9, "y": 0.2},
                        {"x": 0.6, "y": 0.8},
                    ],
                }
            ],
        )

        empty = _record_for_filename(result, "empty.png")
        self.assertTrue(empty["can_import"])
        self.assertEqual(empty["status"], 2)
        self.assertEqual(empty["annotation_json"]["annotations"], [])

        missing = _record_for_filename(result, "missing.png")
        self.assertTrue(missing["can_import"])
        self.assertEqual(missing["status"], 0)
        self.assertIsNone(missing["source_label_path"])

    def test_rejects_duplicate_stems_across_splits(self) -> None:
        """JSON files are currently named by stem, so train/val stems must be unique."""
        self._image("train", "duplicate.png")
        self._label("train", "duplicate.txt", "0 0.5 0.5 0.2 0.2\n")
        self._image("val", "duplicate.png")
        self._label("val", "duplicate.txt", "1 0.5 0.5 0.2 0.2\n")

        result = inspect_yolo_dataset(self.dataset_root)
        duplicate_records = [
            record for record in result["records"] if record["target_stem"] == "duplicate"
        ]
        self.assertEqual(len(duplicate_records), 2)
        self.assertTrue(all(not record["can_import"] for record in duplicate_records))
        self.assertTrue(all(record["errors"] for record in duplicate_records))
        self.assertTrue(result["errors"])

    def test_rejects_invalid_class_and_malformed_label_rows(self) -> None:
        """A bad TXT must make only its matching image non-importable and report why."""
        self._image("train", "bad.txt-name.png")
        self._label(
            "train",
            "bad.txt-name.txt",
            "9 0.5 0.5 0.2 0.2\n"  # class outside data.yaml
            "0 0.5 invalid 0.2 0.2\n"  # non-numeric coordinate
            "0 1.2 0.5 0.2 0.2\n"  # normalized coordinate outside [0, 1]
            "0 0.1 0.2 0.3\n",  # neither bbox nor polygon
        )
        self._image("train", "valid.png")
        self._label("train", "valid.txt", "1 0.4 0.4 0.2 0.2\n")

        result = inspect_yolo_dataset(self.dataset_root)
        invalid = _record_for_filename(result, "bad.txt-name.png")
        valid = _record_for_filename(result, "valid.png")

        self.assertFalse(invalid["can_import"])
        self.assertTrue(invalid["errors"])
        self.assertTrue(result["errors"])
        self.assertTrue(valid["can_import"])
        self.assertEqual(valid["status"], 1)

    def test_normalises_json_annotation_aliases_and_pixel_points(self) -> None:
        """JSON import preserves labels while converting pixel points to 0--1."""
        source = self.temp_root / "external.json"
        source.write_text(
            '{\n'
            '  "image_width": 1000,\n'
            '  "image_height": 500,\n'
            '  "annotations": [\n'
            '    {"type": "rect", "label": "划伤", "points": '
            '[{"x": 100, "y": 50}, {"x": 400, "y": 250}]},\n'
            '    {"type": "polygon", "category": "脏污", "points": '
            '[{"x": 0.1, "y": 0.2}, {"x": 0.8, "y": 0.2}, {"x": 0.5, "y": 0.7}]}\n'
            '  ]\n'
            '}\n',
            encoding="utf-8",
        )

        result = parse_json_annotation_file(source)

        self.assertTrue(result["can_import"])
        self.assertEqual(result["categories"], ["划伤", "脏污"])
        self.assertEqual(
            result["annotation_json"]["annotations"],
            [
                {
                    "type": "rect",
                    "lable": "划伤",
                    "points": [{"x": 0.1, "y": 0.1}, {"x": 0.4, "y": 0.5}],
                },
                {
                    "type": "polygon",
                    "lable": "脏污",
                    "points": [
                        {"x": 0.1, "y": 0.2},
                        {"x": 0.8, "y": 0.2},
                        {"x": 0.5, "y": 0.7},
                    ],
                },
            ],
        )


class LegacyProjectStateTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_base = Path(__file__).resolve().parent / ".tmp_test_workspace"
        self.temp_base.mkdir(exist_ok=True)
        self.temp_root = self.temp_base / f"legacy-project-{uuid.uuid4().hex}"
        self.temp_root.mkdir()

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_root, ignore_errors=False)
        try:
            self.temp_base.rmdir()
        except OSError:
            pass

    def test_migrates_legacy_empty_dict_state_to_lists(self) -> None:
        """Projects created by the old dialog should never expose a dict to append()."""
        paths, paths_repaired = TestWindow._normalise_loaded_relative_paths({})
        flags, flags_repaired = TestWindow._normalise_loaded_flags({}, paths)

        self.assertEqual(paths, [])
        self.assertEqual(flags, [])
        self.assertTrue(paths_repaired)
        self.assertTrue(flags_repaired)

    def test_recognises_only_exact_orphaned_dataset_files_for_recovery(self) -> None:
        """A failed old import can be retried without adopting arbitrary user files."""
        source = self.temp_root / "source.png"
        target_image = self.temp_root / "images" / "source.png"
        target_json = self.temp_root / "jsons" / "source.json"
        _write_png(source)
        target_image.parent.mkdir()
        target_json.parent.mkdir()
        shutil.copy2(source, target_image)
        expected_json = {
            "image_width": 32,
            "image_height": 16,
            "annotations": [],
        }
        target_json.write_text(json.dumps(expected_json, ensure_ascii=False), encoding="utf-8")

        class RecoveryProbe:
            _dataset_json_matches = staticmethod(TestWindow._dataset_json_matches)

        probe = RecoveryProbe()
        self.assertTrue(
            TestWindow._is_recoverable_dataset_entry(
                probe, str(source), str(target_image), str(target_json), expected_json
            )
        )
        self.assertFalse(
            TestWindow._is_recoverable_dataset_entry(
                probe,
                str(source),
                str(target_image),
                str(target_json),
                {**expected_json, "annotations": [{"type": "rect"}]},
            )
        )


class AutoTraverseToggleTest(unittest.TestCase):
    class _Timer:
        def __init__(self):
            self.active = False
            self.start_calls = 0

        def isActive(self):
            return self.active

        def start(self):
            self.active = True
            self.start_calls += 1

        def stop(self):
            self.active = False

    class _Action:
        def __init__(self):
            self.text = "自动遍历"

        def setText(self, text):
            self.text = text

    class _ListWidget:
        def __init__(self, count):
            self._count = count
            self.rows = []
            self.current_row = -1

        def count(self):
            return self._count

        def setCurrentRow(self, row):
            self.current_row = row
            self.rows.append(row)

        def currentRow(self):
            return self.current_row

    class _Window:
        stop_batch_run = TestWindow.stop_batch_run

        def __init__(self, count):
            self.timer1 = AutoTraverseToggleTest._Timer()
            self.run_action = AutoTraverseToggleTest._Action()
            self.listWidget = AutoTraverseToggleTest._ListWidget(count)
            self.current_index = 0
            self.total_items = 0

        def sync_categories_from_data(self):
            return False

    def test_second_click_stops_and_end_restores_button_text(self) -> None:
        window = self._Window(3)

        TestWindow.batch_run(window)
        self.assertTrue(window.timer1.isActive())
        self.assertEqual(window.run_action.text, "停止遍历")
        self.assertEqual(window.timer1.start_calls, 1)

        TestWindow.batch_run(window)
        self.assertFalse(window.timer1.isActive())
        self.assertEqual(window.run_action.text, "自动遍历")

        TestWindow.batch_run(window)
        for _ in range(4):
            TestWindow.process_next_item(window)
        self.assertEqual(window.listWidget.rows[-3:], [0, 1, 2])
        self.assertFalse(window.timer1.isActive())
        self.assertEqual(window.run_action.text, "自动遍历")


if __name__ == "__main__":
    unittest.main(verbosity=2)
