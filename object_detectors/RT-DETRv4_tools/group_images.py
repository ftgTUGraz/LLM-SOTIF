#!/usr/bin/env python3
"""
Group images from the rawimages dataset into four subsets:
  1. Natural    – all images under any "Natural" folder
  2. Handcraft  – all images under any "Handcraft" folder
  3. Environment – all images under the "Environment" subtree
  4. Object      – all images under the "Object" subtree

Usage:
    python group_images.py <path_to_rawimages> [output_dir]

Output structure:
    <output_dir>/
        Natural/        1.jpg, 2.jpg, ...
        Handcraft/      1.jpg, 2.jpg, ...
        Environment/    1.jpg, 2.jpg, ...
        Object/         1.jpg, 2.jpg, ...
"""

import argparse
import shutil
from pathlib import Path


def collect_images(root: Path) -> dict[str, list[Path]]:
    """Walk the tree and bucket every .jpg into the four groups."""
    groups: dict[str, list[Path]] = {
        "Natural": [],
        "Handcraft": [],
        "Environment": [],
        "Object": [],
    }

    for img in sorted(root.rglob("*")):
        if not img.is_file() or img.suffix.lower() not in (".jpg", ".jpeg"):
            continue

        parts = img.relative_to(root).parts  # e.g. ('Environment', 'Rain', 'Natural', 'x.jpg')

        # Environment / Object membership (top-level category)
        if "Environment" in parts:
            groups["Environment"].append(img)
        elif "Object" in parts:
            groups["Object"].append(img)

        # Natural / Handcraft membership (leaf folder name)
        if "Natural" in parts:
            groups["Natural"].append(img)
        elif "Handcraft" in parts:
            groups["Handcraft"].append(img)

    return groups


def export(groups: dict[str, list[Path]], output_dir: Path) -> None:
    """Copy images into numbered output folders."""
    for name, images in groups.items():
        dest = output_dir / name
        dest.mkdir(parents=True, exist_ok=True)

        for idx, src in enumerate(images, start=1):
            shutil.copy2(src, dest / f"{idx}.jpg")

        print(f"  {name:.<20s} {len(images)} images")


def main() -> None:
    parser = argparse.ArgumentParser(description="Group rawimages into four subsets.")
    parser.add_argument("rawimages", type=Path, help="Path to the rawimages root folder")
    parser.add_argument("output", type=Path, nargs="?", default=None,
                        help="Output directory (default: <rawimages>/../grouped_images)")
    args = parser.parse_args()

    root = args.rawimages.resolve()
    if not root.is_dir():
        raise SystemExit(f"Error: {root} is not a directory")

    output_dir = (args.output or root.parent / "grouped_images").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Scanning:  {root}")
    print(f"Output:    {output_dir}\n")

    groups = collect_images(root)
    export(groups, output_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
