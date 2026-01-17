#!/usr/bin/env python3
"""
------------------------------------------------------------------------------
Description:
    This script reads label files (YOLO format + safety flag), matches them
    with corresponding images, draws bounding boxes, and saves the results.
    It preserves the subdirectory structure of the input folders.

Input Format (.txt):
    <class_id> <x_center> <y_center> <width> <height> <safety_critical>
    Values should be normalized (0.0 to 1.0).
    safety_critical: 0 (green box), 1 (red box).

Usage:
    python visualize_boxes.py --labels /path/to/txts --images /path/to/imgs --output /path/to/save

Optional Arguments:
    --thickness <int>   : Line thickness (default: 1)
    --ext <string>      : Image extension to look for (default: .jpg)
------------------------------------------------------------------------------
"""

import os
import sys
import argparse
import cv2


def parse_arguments():
    # setup command line arguments
    parser = argparse.ArgumentParser(description="Visualize bounding boxes on images.")
    parser.add_argument("--labels", required=True, help="path to folder containing label text files")
    parser.add_argument("--images", required=True, help="path to folder containing original images")
    parser.add_argument("--output", required=True, help="path to save output images")
    parser.add_argument("--thickness", type=int, default=1, help="bounding box thickness")
    parser.add_argument("--ext", default=".jpg", help="image file extension (e.g., .jpg, .png)")
    return parser.parse_args()


def draw_boxes_normalized(image_path, txt_path, output_path, thickness):
    # check if image exists
    if not os.path.exists(image_path):
        print(f"warning: image not found at {image_path}, skipping.")
        return False

    image = cv2.imread(image_path)
    if image is None:
        print(f"error: failed to load image {image_path}")
        return False

    height, width, _ = image.shape

    with open(txt_path, 'r') as f:
        lines = f.readlines()

    for line in lines:
        try:
            # parse the 6 columns based on user format
            parts = list(map(float, line.split()))
            class_id = int(parts[0])
            x_center, y_center, box_width, box_height = parts[1:5]
            safety_critical = int(parts[5])

            # calculate pixel coordinates
            x_min = int((x_center - box_width / 2) * width)
            y_min = int((y_center - box_height / 2) * height)
            x_max = int((x_center + box_width / 2) * width)
            y_max = int((y_center + box_height / 2) * height)

            # determine color based on safety flag
            # 0 = green, 1 = red
            color = (0, 255, 0) if safety_critical == 0 else (0, 0, 255)

            # draw rectangle
            cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, thickness)

            label_text = f"Class {class_id}"
            cv2.putText(image, label_text, (x_min, y_max + 20), cv2.FONT_HERSHEY_SIMPLEX, 1, color, thickness)

        except ValueError:
            print(f"skipping malformed line in {txt_path}")
            continue

    # save final image
    cv2.imwrite(output_path, image)
    return True


def main():
    args = parse_arguments()
    count = 0

    print("Starting processing...")

    # walk through all folders in labels directory
    for root, _, files in os.walk(args.labels):
        for file in files:
            if file.lower().endswith('.txt'):
                label_path = os.path.join(root, file)

                relative_path = os.path.relpath(root, args.labels)
                output_folder = os.path.join(args.output, relative_path)
                pic_folder = os.path.join(args.images, relative_path)

                os.makedirs(output_folder, exist_ok=True)

                image_name = os.path.splitext(file)[0] + args.ext
                pic_path = os.path.join(pic_folder, image_name)
                output_path = os.path.join(output_folder, image_name)

                # process the file
                success = draw_boxes_normalized(pic_path, label_path, output_path, args.thickness)
                if success:
                    print(f"saved: {output_path}")
                    count += 1

    print(f"Total processed pictures: {count}")


if __name__ == "__main__":
    main()





