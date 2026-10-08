"""Generate a small synthetic project without copying business data."""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

from PIL import Image, ImageDraw


def create_demo(destination: Path) -> Path:
    destination = destination.resolve()
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"Destination is not empty; choose another --output: {destination}")
    (destination / "images").mkdir(parents=True, exist_ok=True)
    (destination / "jsons").mkdir(parents=True, exist_ok=True)
    names = ["01_scratch.png", "02_dent.png", "03_good.png", "04_unlabelled.png"]
    for index, name in enumerate(names):
        image = Image.new("RGB", (960, 640), "#eef2f7")
        draw = ImageDraw.Draw(image)
        for x in range(0, 960, 40):
            draw.line((x, 0, x, 640), fill="#e0e7ef")
        for y in range(0, 640, 40):
            draw.line((0, y, 960, y), fill="#e0e7ef")
        draw.rounded_rectangle((100, 100, 860, 530), radius=36, fill="#b9c4d2", outline="#77869b", width=5)
        draw.rounded_rectangle((130, 130, 830, 500), radius=26, fill="#d6dee8", outline="#a0aebe", width=2)
        draw.text((130, 150), "LabelSystem / SYNTHETIC DEMO", fill="#334155", font_size=25)
        draw.text((130, 192), "Generated artwork - no production inspection data", fill="#52637a", font_size=17)
        annotations = []
        if index == 0:
            draw.line((300, 340, 655, 385), fill="#42566d", width=7)
            draw.line((310, 350, 650, 394), fill="#f7f9fc", width=3)
            annotations = [{"type": "rect", "lable": "scratch", "points": [
                {"x": 280 / 960, "y": 315 / 640}, {"x": 680 / 960, "y": 420 / 640}]}]
        elif index == 1:
            draw.polygon([(520, 325), (640, 305), (705, 370), (650, 430), (535, 410)], fill="#899bb0")
            draw.line([(520, 325), (640, 305), (705, 370)], fill="#f7f9fc", width=5)
            annotations = [{"type": "polygon", "lable": "dent", "points": [
                {"x": x / 960, "y": y / 640} for x, y in [(510, 315), (645, 295), (715, 370), (655, 440), (525, 420)]]}]
        elif index == 3:
            draw.line((420, 360, 630, 330), fill="#66778e", width=5)
        draw.text((130, 460), name.removesuffix(".png"), fill="#334155", font_size=21)
        image.save(destination / "images" / name)
        if index != 3:
            (destination / "jsons" / (Path(name).stem + ".json")).write_text(json.dumps({
                "image_width": 960, "image_height": 640, "annotations": annotations}, indent=2), encoding="utf-8")
    (destination / "label.txt").write_text("scratch\ndent\n", encoding="utf-8")
    (destination / "datafile.dat").write_bytes(pickle.dumps(["images/" + name for name in names]))
    (destination / "flagfile.dat").write_bytes(pickle.dumps([1, 1, 2, 0]))
    (destination / "sample_tree.json").write_text(json.dumps({
        "groups": {"scratch": ["long"], "dent": []},
        "assignments": {"images/01_scratch.png": {"scratch": "long"}}}, indent=2), encoding="utf-8")
    (destination / "config.json").write_text(json.dumps({
        "name": "demo_project", "description": "Synthetic demonstration only",
        "datafile": "datafile.dat", "flagfile": "flagfile.dat"}, indent=2), encoding="utf-8")
    return destination


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "demo_project")
    args = parser.parse_args()
    try:
        print(f"Created synthetic project: {create_demo(args.output)}")
    except FileExistsError as exc:
        parser.exit(1, f"{exc}\n")
