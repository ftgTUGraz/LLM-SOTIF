import os
import sys
import argparse
from pathlib import Path
import json
import time
from typing import Dict, List

try:
    from inference_sdk import InferenceHTTPClient
except Exception as e:
    InferenceHTTPClient = None  # Will be checked at runtime
import requests


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_IMAGES_ROOT = REPO_ROOT / "PeSOTIF" / "images"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "PeSOTIF" / "image_output_YOLOV11"
CLASSES_FILE = REPO_ROOT / "PeSOTIF" / "classes.txt"


def load_class_mappings(classes_file: Path) -> Dict[str, int]:
    """
    Load class names from classes.txt and build two-directional mappings.
    Normalizes by replacing spaces/underscores and lowercasing for robust matching.
    """
    if not classes_file.exists():
        raise FileNotFoundError(f"classes.txt not found at {classes_file}")
    with classes_file.open("r", encoding="utf-8") as f:
        classes = [line.strip() for line in f if line.strip()]
    name_to_id: Dict[str, int] = {}
    for idx, name in enumerate(classes):
        key1 = name.lower().replace(" ", "_")
        key2 = name.lower().replace("_", " ")
        name_to_id[key1] = idx
        name_to_id[key2] = idx
    return name_to_id


def ensure_package():
    # inference-sdk is optional now; will fall back to HTTP if missing
    return


def iter_image_files(images_root: Path) -> List[Path]:
    exts = {".jpg", ".jpeg", ".png", ".JPG", ".PNG", ".JPEG"}
    files: List[Path] = []
    for root, _, filenames in os.walk(images_root):
        for fn in filenames:
            if Path(fn).suffix in exts:
                files.append(Path(root) / fn)
    files.sort()
    return files


def normalize_bbox(px_x: float, px_y: float, px_w: float, px_h: float, img_w: float, img_h: float):
    # Roboflow returns center-based xywh in pixels
    x_c = px_x / max(1.0, img_w)
    y_c = px_y / max(1.0, img_h)
    w_n = px_w / max(1.0, img_w)
    h_n = px_h / max(1.0, img_h)
    # clamp to [0,1]
    x_c = max(0.0, min(1.0, x_c))
    y_c = max(0.0, min(1.0, y_c))
    w_n = max(0.0, min(1.0, w_n))
    h_n = max(0.0, min(1.0, h_n))
    return x_c, y_c, w_n, h_n


def write_predictions_txt(out_file: Path, lines: List[str]):
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with out_file.open("w", encoding="utf-8") as f:
        if lines:
            f.write("\n".join(lines))


def _infer_via_sdk(client: "InferenceHTTPClient", image_path: Path, model_id: str) -> dict:
    try:
        return client.infer(str(image_path), model_id=model_id)
    except Exception:
        return {}


def _workflow_via_sdk(client: "InferenceHTTPClient", image_path: Path, workspace_name: str, workflow_id: str, use_cache: bool) -> dict:
    try:
        return client.run_workflow(
            workspace_name=workspace_name,
            workflow_id=workflow_id,
            images={"image": str(image_path)},
            use_cache=use_cache,
        )
    except Exception:
        return {}


def _infer_via_http(image_path: Path, model_id: str, api_url: str, api_key: str) -> dict:
    # Try detect.roboflow.com first, then serverless.roboflow.com
    endpoints = [
        f"https://detect.roboflow.com/{model_id}",
        f"{api_url.rstrip('/')}/{model_id}",
    ]
    files = {"file": open(image_path, "rb")}
    last_err = None
    for url in endpoints:
        try:
            resp = requests.post(url, params={"api_key": api_key}, files=files, timeout=60)
            if resp.ok:
                return resp.json()
            last_err = f"HTTP {resp.status_code}: {resp.text[:200]}"
        except Exception as e:
            last_err = str(e)
    # fallback empty
    return {}


def _collect_detections(obj) -> List[dict]:
    # Recursively collect detections with keys like class/x/y/width/height
    detections = []
    if isinstance(obj, dict):
        # exact predictions list
        if "predictions" in obj and isinstance(obj["predictions"], list):
            for it in obj["predictions"]:
                if isinstance(it, dict) and all(k in it for k in ("class", "x", "y", "width", "height")):
                    detections.append(it)
        # walk values
        for v in obj.values():
            detections.extend(_collect_detections(v))
    elif isinstance(obj, list):
        for v in obj:
            detections.extend(_collect_detections(v))
    return detections


def infer_image(
    client: "InferenceHTTPClient",
    image_path: Path,
    model_id: str,
    name_to_id: Dict[str, int],
    api_url: str,
    api_key: str,
    workflow_id: str = "",
    workspace_name: str = "",
    use_cache: bool = True,
) -> List[str]:
    """
    Runs inference for a single image and returns lines in the 6-column normalized format:
    class_id x_center y_center width height safety_critical
    safety_critical is defaulted to 0 (non-critical) because Roboflow API does not provide it.
    """
    # Prefer workflow via SDK if specified; else standard infer
    if workflow_id and client is not None:
        result = _workflow_via_sdk(client, image_path, workspace_name, workflow_id, use_cache)
    else:
        if client is not None:
            result = _infer_via_sdk(client, image_path, model_id)
        else:
            result = _infer_via_http(image_path, model_id, api_url, api_key)

    # Expected schema: result["predictions"] list, and possibly result["image"] with width/height
    preds = []
    if isinstance(result, dict):
        if "predictions" in result:
            preds = result.get("predictions", [])
        else:
            # workflow results: try to collect from nested structure
            preds = _collect_detections(result)
    img_w = None
    img_h = None
    if isinstance(result, dict):
        imeta = result.get("image") or {}
        img_w = imeta.get("width")
        img_h = imeta.get("height")

    lines: List[str] = []
    for det in preds:
        if not isinstance(det, dict):
            continue
        cls_name = str(det.get("class", "")).strip().lower()
        cls_key = cls_name.replace(" ", "_")
        if cls_key not in name_to_id:
            # try inverse normalization
            cls_key2 = cls_name.replace("_", " ")
            if cls_key2 in name_to_id:
                cls_key = cls_key2
            else:
                # unrecognized class name -> skip
                continue
        cls_id = name_to_id[cls_key]

        x = float(det.get("x", 0.0))
        y = float(det.get("y", 0.0))
        w = float(det.get("width", 0.0))
        h = float(det.get("height", 0.0))

        # if API did not include image size, try to read via PIL
        if not img_w or not img_h:
            try:
                from PIL import Image  # lazy import
                with Image.open(image_path) as im:
                    img_w, img_h = im.size
            except Exception:
                img_w, img_h = 1.0, 1.0

        x_c, y_c, w_n, h_n = normalize_bbox(x, y, w, h, float(img_w), float(img_h))
        safety_critical = 0
        lines.append(f"{cls_id} {x_c} {y_c} {w_n} {h_n} {safety_critical}")

    return lines


def main():
    parser = argparse.ArgumentParser(description="Run Roboflow inference over PeSOTIF images and write YOLO-format outputs.")
    parser.add_argument("--images_root", type=str, default=str(DEFAULT_IMAGES_ROOT))
    parser.add_argument("--output_root", type=str, default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--api_url", type=str, default="https://serverless.roboflow.com")
    parser.add_argument("--api_key", type=str, default=os.getenv("ROBOFLOW_API_KEY", "gSwuuB9RHApu7QaxjEmE"))
    parser.add_argument("--model_id", type=str, default="pesotif/4")
    parser.add_argument("--limit", type=int, default=-1, help="Limit number of images for a quick run; -1 for all")
    parser.add_argument("--run_eval", type=str, default="true", choices=["true", "false"], help="Run evaluation after inference")
    parser.add_argument("--workflow_id", type=str, default="", help="Roboflow workflow id; if provided and SDK available, run workflow")
    parser.add_argument("--workspace_name", type=str, default="", help="Roboflow workspace name for workflow")
    parser.add_argument("--use_cache", type=str, default="true", choices=["true", "false"], help="Cache workflow definition")

    args = parser.parse_args()

    ensure_package()

    images_root = Path(args.images_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    name_to_id = load_class_mappings(CLASSES_FILE)

    client = None
    if InferenceHTTPClient is not None:
        try:
            client = InferenceHTTPClient(api_url=args.api_url, api_key=args.api_key)
        except Exception:
            client = None

    image_files = iter_image_files(images_root)
    if args.limit and args.limit > 0:
        image_files = image_files[: args.limit]

    total = len(image_files)
    print(f"Discovered {total} images under {images_root}.")
    processed = 0
    t0 = time.time()

    use_cache = True if args.use_cache.lower() == "true" else False

    for img_path in image_files:
        rel = img_path.relative_to(images_root)
        out_dir = output_root / rel.parent
        out_file = out_dir / (img_path.stem + ".txt")

        lines = infer_image(
            client,
            img_path,
            args.model_id,
            name_to_id,
            args.api_url,
            args.api_key,
            workflow_id=args.workflow_id,
            workspace_name=args.workspace_name,
            use_cache=use_cache,
        )
        write_predictions_txt(out_file, lines)

        processed += 1
        if processed % 50 == 0 or processed == total:
            dt = time.time() - t0
            print(f"Processed {processed}/{total} images in {dt:.1f}s")

    print(f"Inference complete. Outputs written to: {output_root}")

    if args.run_eval.lower() == "true":
        # Call the repository evaluation script
        eval_script = REPO_ROOT / "result_evaluation.py"
        if not eval_script.exists():
            print("Evaluation script not found; skipping evaluation.")
            return
        cmd = [
            sys.executable,
            str(eval_script),
            "--test_label", str(output_root),
            "--dataset_label", str(REPO_ROOT / "PeSOTIF" / "labels"),
            "--mmap_cal", "true",
        ]
        print("Running evaluation...")
        import subprocess
        try:
            subprocess.run(cmd, check=False)
        except Exception as e:
            print(f"Evaluation failed: {e}")


if __name__ == "__main__":
    main()


