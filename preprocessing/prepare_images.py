#!/usr/bin/env python3
"""
------------------------------------------------------------------------------
Description:
    This script resizes images to a standard resolution (maintaining aspect ratio)
    and draws a normalized ruler/grid overlay (green border and ticks) on the
    top and left edges. It preserves the input directory structure.

Usage:
    python prepare_images.py --input /path/to/raw_images --output /path/to/save

Optional Arguments:
    --width <int>  : Target maximum width (default: 800)
    --height <int> : Target maximum height (default: 600)
------------------------------------------------------------------------------
"""

import os
import argparse
import cv2
from PIL import Image


def resize_image_aspect_ratio(input_path, output_path, max_width, max_height):
    with Image.open(input_path) as img:
        original_width, original_height = img.size

        width_ratio = max_width / original_width
        height_ratio = max_height / original_height
        scale_factor = min(width_ratio, height_ratio)

        new_width = int(original_width * scale_factor)
        new_height = int(original_height * scale_factor)

        # resize using high-quality resampling
        resized_img = img.resize((new_width, new_height), Image.LANCZOS).convert("RGB")

        resized_img.save(output_path)


def prepare_image_normalized(img_pth, output_path, target_w, target_h):
    resize_image_aspect_ratio(img_pth, output_path, target_w, target_h)

    # load the image
    img = cv2.imread(output_path)
    height, width, _ = img.shape

    # draw rectangle border
    cv2.rectangle(img, (0, 0), (width - 1, height - 1), color=(0, 255, 0), thickness=2)

    # add rulers on the top (width-based, every 1/10th)
    step_x = width / 10
    for i in range(9):
        x = int(round((i + 1) * step_x))
        cv2.line(img, (x, 0), (x, 20), (0, 255, 0), 1)
        label = f"0.{(i + 1)}"
        cv2.putText(img, label, (x - 8, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

    # add rulers on the left (height-based, every 1/10th)
    step_y = height / 10
    for i in range(9):
        y = int(round((i + 1) * step_y))
        cv2.line(img, (0, y), (20, y), (0, 255, 0), 1)
        label = f"0.{(i + 1)}"
        cv2.putText(img, label, (25, y + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

    # save the final image
    cv2.imwrite(output_path, img)
    print(f"Final result image saved to: {output_path}")


if __name__ == "__main__":
    # setup command line arguments
    parser = argparse.ArgumentParser(description="Resize images and add normalized rulers.")
    parser.add_argument("--input", required=True, help="root folder containing input images")
    parser.add_argument("--output", required=True, help="root folder to save processed images")
    parser.add_argument("--width", type=int, default=800, help="target width for resizing")
    parser.add_argument("--height", type=int, default=600, help="target height for resizing")

    args = parser.parse_args()

    count = 0

    # walk through all folders and files
    for root, _, files in os.walk(args.input):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                input_file_path = os.path.join(root, file)

                relative_path = os.path.relpath(root, args.input)
                output_folder = os.path.join(args.output, relative_path)
                os.makedirs(output_folder, exist_ok=True)
                output_file_path = os.path.join(output_folder, file)

                try:
                    prepare_image_normalized(input_file_path, output_file_path, args.width, args.height)
                    count += 1
                except Exception as e:
                    print(f"error processing {file}: {e}")

    print(f"Total pictures processed: {count}")



