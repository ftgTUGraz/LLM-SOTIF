#!/usr/bin/env python3
"""
Filter RT-DETRv4 inference labels (COCO 80-class) to PeSOTIF 11-class format.

RT-DETRv4 outputs YOLO-format labels with COCO class indices (0-79).
PeSOTIF uses 11 classes: car, bus, truck, train, bike, motor, person, rider,
                          traffic_sign, traffic_light, traffic_cone

This script:
  1. Reads each label file (1.txt to 1126.txt) from the input folder.
  2. Keeps only detections whose COCO class maps to a PeSOTIF class.
  3. Remaps the class index to the PeSOTIF scheme.
  4. Writes the filtered labels to an output folder.

Usage:
    python filter_coco_to_pesotif.py <input_folder> [output_folder]

If output_folder is not specified, it defaults to <input_folder>_pesotif.
"""

import argparse
import os
import sys

# ─── COCO 80-class index → PeSOTIF class index mapping ───
# Only the COCO classes that have a PeSOTIF equivalent are listed.
# COCO classes without a PeSOTIF match are simply dropped.
#
# PeSOTIF classes (0-indexed):
#   0: car          ← COCO  2 (car)
#   1: bus          ← COCO  5 (bus)
#   2: truck        ← COCO  6 (truck)
#   3: train        ← COCO  7 (train)
#   4: bike         ← COCO  1 (bicycle)
#   5: motor        ← COCO  3 (motorcycle)
#   6: person       ← COCO  0 (person)
#   7: rider        ← (no COCO equivalent)
#   8: traffic_sign ← (no COCO equivalent)
#   9: traffic_light← COCO  9 (traffic light)
#  10: traffic_cone ← (no COCO equivalent)
#
# Note: "rider", "traffic_sign", and "traffic_cone" have no direct COCO
# counterpart, so no COCO detections will map to classes 7, 8, or 10.

COCO_TO_PESOTIF = {
    0: 6,   # person       → PeSOTIF 6
    1: 4,   # bicycle      → PeSOTIF 4 (bike)
    2: 0,   # car          → PeSOTIF 0
    3: 5,   # motorcycle   → PeSOTIF 5 (motor)
    5: 1,   # bus          → PeSOTIF 1
    6: 2,   # truck        → PeSOTIF 2
    7: 3,   # train        → PeSOTIF 3
    9: 9,   # traffic light→ PeSOTIF 9
}

PESOTIF_CLASSES = [
    "car", "bus", "truck", "train", "bike",
    "motor", "person", "rider", "traffic_sign",
    "traffic_light", "traffic_cone",
]


def filter_label_file(input_path, output_path):
    """Filter a single YOLO-format label file.

    Returns:
        tuple: (total_lines, kept_lines, dropped_lines)
    """
    kept_lines = []
    total = 0
    dropped = 0

    with open(input_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            total += 1
            parts = line.split()
            coco_cls = int(parts[0])

            if coco_cls in COCO_TO_PESOTIF:
                pesotif_cls = COCO_TO_PESOTIF[coco_cls]
                # Replace class index, keep bbox (and optional confidence) intact
                parts[0] = str(pesotif_cls)
                kept_lines.append(" ".join(parts))
            else:
                dropped += 1

    with open(output_path, "w") as f:
        if kept_lines:
            f.write("\n".join(kept_lines) + "\n")
        # If no detections remain, write an empty file (valid YOLO convention)

    return total, len(kept_lines), dropped


def main():
    parser = argparse.ArgumentParser(
        description="Filter RT-DETRv4 COCO labels to PeSOTIF 11-class format."
    )
    parser.add_argument(
        "input_folder",
        help="Folder containing RT-DETRv4 inference labels (1.txt – 1126.txt).",
    )
    parser.add_argument(
        "output_folder",
        nargs="?",
        default=None,
        help="Output folder for filtered labels. Default: <input_folder>_pesotif",
    )
    args = parser.parse_args()

    input_folder = args.input_folder
    output_folder = args.output_folder or (input_folder.rstrip("/\\") + "_pesotif")

    if not os.path.isdir(input_folder):
        print(f"Error: Input folder '{input_folder}' does not exist.", file=sys.stderr)
        sys.exit(1)

    os.makedirs(output_folder, exist_ok=True)

    # Collect label files: expect 1.txt through 1126.txt
    label_files = []
    for fname in os.listdir(input_folder):
        if fname.endswith(".txt"):
            # Accept any .txt file (flexible in case naming varies)
            label_files.append(fname)

    label_files.sort(key=lambda x: int(x.replace(".txt", "")) if x.replace(".txt", "").isdigit() else 0)

    if not label_files:
        print(f"Error: No .txt files found in '{input_folder}'.", file=sys.stderr)
        sys.exit(1)

    total_files = 0
    total_detections = 0
    total_kept = 0
    total_dropped = 0
    empty_after_filter = 0

    for fname in label_files:
        input_path = os.path.join(input_folder, fname)
        output_path = os.path.join(output_folder, fname)

        t, k, d = filter_label_file(input_path, output_path)
        total_files += 1
        total_detections += t
        total_kept += k
        total_dropped += d
        if k == 0 and t > 0:
            empty_after_filter += 1

    # ─── Summary ───
    print(f"{'=' * 55}")
    print(f"  RT-DETRv4 COCO → PeSOTIF Label Filter - Summary")
    print(f"{'=' * 55}")
    print(f"  Input folder :  {input_folder}")
    print(f"  Output folder:  {output_folder}")
    print(f"  Files processed:  {total_files}")
    print(f"{'─' * 55}")
    print(f"  Total detections:     {total_detections:>7}")
    print(f"  Kept (mapped):        {total_kept:>7}  ({100*total_kept/max(total_detections,1):.1f}%)")
    print(f"  Dropped (unmapped):   {total_dropped:>7}  ({100*total_dropped/max(total_detections,1):.1f}%)")
    print(f"  Files empty after:    {empty_after_filter:>7}")
    print(f"{'─' * 55}")
    print(f"  COCO → PeSOTIF mapping used:")
    for coco_id, pesotif_id in sorted(COCO_TO_PESOTIF.items()):
        coco_names = [
            "person", "bicycle", "car", "motorcycle", "airplane",
            "bus", "truck", "train", "boat", "traffic light",
        ]
        coco_name = coco_names[coco_id] if coco_id < len(coco_names) else f"class_{coco_id}"
        print(f"    COCO {coco_id:>2} ({coco_name:<14}) → PeSOTIF {pesotif_id:>2} ({PESOTIF_CLASSES[pesotif_id]})")
    print(f"{'=' * 55}")
    print(f"\n  Note: 'rider', 'traffic_sign', 'traffic_cone' have no")
    print(f"  COCO equivalent — no detections map to these classes.\n")


if __name__ == "__main__":
    main()
