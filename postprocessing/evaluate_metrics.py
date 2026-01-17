"""
Description:
  This script evaluates the performance of a model by comparing
  prediction text files against ground truth text files. It calculates Intersection
  over Union (IoU), Precision, Recall, and supports calculating mmAP (mean Average
  Precision across IoU thresholds .50 to .95).

  Input File Format (Space-separated .txt files):
  <class_id> <x_center> <y_center> <width> <height> <safety_critical_flag>
  - Coordinates should be normalized (0.0 to 1.0).

Usage:
  1. Basic Evaluation (Single IoU threshold):
     python evaluate_metrics.py --pred <path_to_preds> --gt <path_to_gt> --output <path_to_save>

  2. Calculate mmAP (0.50:0.95):
     python evaluate_metrics.py --pred <path_to_preds> --gt <path_to_gt> --mmap

Arguments:
  --pred        Path to the folder containing prediction labels.
  --gt          Path to the folder containing ground truth labels.
  --output      (Optional) Path to save detailed result text files.
  --iou         (Optional) IoU threshold for matching (default: 0.5).
  --mmap        (Flag) If set, calculates mean AP across IoU 0.5 to 0.95.
  --recall_lim  (Optional) Threshold to count files with low recall (default: 0.1).
"""

import os
import sys
import argparse
import numpy as np


def calculate_iou(box1, box2):
    """
    Calculates the Intersection over Union (IoU) of two bounding boxes.
    Format: (x_center, y_center, width, height) - normalized.
    """
    box1_x_min = box1[0] - box1[2] / 2
    box1_y_min = box1[1] - box1[3] / 2
    box1_x_max = box1[0] + box1[2] / 2
    box1_y_max = box1[1] + box1[3] / 2

    box2_x_min = box2[0] - box2[2] / 2
    box2_y_min = box2[1] - box2[3] / 2
    box2_x_max = box2[0] + box2[2] / 2
    box2_y_max = box2[1] + box2[3] / 2

    x_intersection_min = max(box1_x_min, box2_x_min)
    y_intersection_min = max(box1_y_min, box2_y_min)
    x_intersection_max = min(box1_x_max, box2_x_max)
    y_intersection_max = min(box1_y_max, box2_y_max)

    intersection_width = max(0, x_intersection_max - x_intersection_min)
    intersection_height = max(0, y_intersection_max - y_intersection_min)
    intersection_area = intersection_width * intersection_height

    box1_area = box1[2] * box1[3]
    box2_area = box2[2] * box2[3]

    union_area = box1_area + box2_area - intersection_area

    return intersection_area / (union_area + 1e-6)


def calculate_metrics(test_labels_path, ground_truth_path, output_path, iou_threshold=0.5, write=False):
    """
    Compares a single prediction file against a ground truth file.
    Returns Recall and Precision for that file.
    """
    if not os.path.exists(ground_truth_path):
        # if ground truth doesn't exist we assume 0 objects
        # if predictions exist they are all False Positives
        return 0.0, 0.0

    with open(test_labels_path, 'r') as f:
        t_lines = f.readlines()

    with open(ground_truth_path, 'r') as f:
        gt_lines = f.readlines()

    tp = 0
    fp = 0
    gt_matched = set()
    iou_scores = []
    matched_points = []

    for test in t_lines:
        try:
            parts = list(map(float, test.split()))
            test_class, xc, yc, w, l = parts[0], parts[1], parts[2], parts[3], parts[4]
        except ValueError:
            continue

        best_iou = 0
        best_gt_idx = -1
        matched_gt_center = (0, 0)

        # compare with all ground truths
        for idx, gt in enumerate(gt_lines):
            try:
                g_parts = list(map(float, gt.split()))
                gt_class, gxc, gyc, gw, gl = g_parts[0], g_parts[1], g_parts[2], g_parts[3], g_parts[4]
            except ValueError:
                continue

            if gt_class == test_class:
                iou_score = calculate_iou((xc, yc, w, l), (gxc, gyc, gw, gl))
                if iou_score > best_iou:
                    best_iou = iou_score
                    best_gt_idx = idx
                    matched_gt_center = (gxc, gyc)

        if best_iou >= iou_threshold and best_gt_idx not in gt_matched:
            tp += 1
            gt_matched.add(best_gt_idx)
            iou_scores.append(best_iou)
            matched_points.append(matched_gt_center)
        else:
            fp += 1

    fn = len(gt_lines) - len(gt_matched)  # false negatives

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    if write and output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as f:
            f.write(f"Recall: {recall:.3f}\n")
            f.write(f"Precision: {precision:.3f}\n")
            f.write(f"True positives: {tp}\n")
            f.write(f"False positives: {fp}\n")
            f.write(f"False negatives: {fn}\n")
            for i, score in enumerate(iou_scores):
                f.write(f"IOU Score: {score:.3f}, GT center: {matched_points[i]}\n")

    return recall, precision


def process_dataset(pred_dir, gt_dir, out_dir, iou_thresh, recall_limit, verbose=True):
    """
    Walks through the dataset and aggregates metrics.
    """
    recalls = []
    precisions = []

    processed_count = 0
    low_recall_count = 0

    for root, _, files in os.walk(pred_dir):
        for file in files:
            if not file.lower().endswith('.txt'):
                continue

            pred_file_path = os.path.join(root, file)

            relative_path = os.path.relpath(root, pred_dir)
            gt_folder = os.path.join(gt_dir, relative_path)
            gt_file_path = os.path.join(gt_folder, file)

            out_file_path = None
            should_write = False
            if out_dir:
                out_folder = os.path.join(out_dir, relative_path)
                out_file_path = os.path.join(out_folder, file)
                should_write = True

            rec, pre = calculate_metrics(
                pred_file_path,
                gt_file_path,
                out_file_path,
                iou_threshold=iou_thresh,
                write=should_write
            )

            recalls.append(rec)
            precisions.append(pre)
            processed_count += 1

            if rec < recall_limit:
                low_recall_count += 1

    if processed_count == 0:
        if verbose:
            print("No .txt files found to process.")
        return 0.0, 0.0

    avg_recall = np.mean(recalls)
    avg_precision = np.mean(precisions)

    if verbose:
        print(f"Processed images: {processed_count}")
        print(f"IoU Threshold: {iou_thresh:.2f}")
        print(f"Average Recall: {avg_recall:.4f}")
        print(f"Average Precision: {avg_precision:.4f}")
        print(f"Files with Recall < {recall_limit}: {low_recall_count}")

    return avg_recall, avg_precision


def main():
    parser = argparse.ArgumentParser(description="Evaluate Object Detection Predictions against Ground Truth.")
    parser.add_argument("--pred", required=True, help="Path to the directory containing Prediction labels (.txt)")
    parser.add_argument("--gt", required=True, help="Path to the directory containing Ground Truth labels (.txt)")
    parser.add_argument("--output", default=None,
                        help="Path to save detailed evaluation results per file. (Default: None)")
    parser.add_argument("--iou", type=float, default=0.5, help="IoU threshold for a positive match. (Default: 0.5)")
    parser.add_argument("--recall_lim", type=float, default=0.1,
                        help="Threshold to flag low recall files. (Default: 0.1)")
    parser.add_argument("--mmap", action="store_true", help="Calculate mmAP@50:95 (ignores --iou argument if set).")

    args = parser.parse_args()

    if not os.path.exists(args.pred):
        print(f"Error: Prediction path not found: {args.pred}")
        sys.exit(1)
    if not os.path.exists(args.gt):
        print(f"Error: Ground Truth path not found: {args.gt}")
        sys.exit(1)

    if args.mmap:
        print("--- Calculating mmAP@50:95 ---")
        precision_at_thresholds = []

        # loop from 0.5 to 0.95 with step 0.05
        for idx in range(10):
            current_iou = 0.5 + (0.05 * idx)
            print(f"\nProcessing step {idx + 1}/10 (IoU: {current_iou:.2f})...")

            # don't write individual file outputs during mmAP calculation
            _, avg_pre = process_dataset(
                args.pred, args.gt, None, current_iou, args.recall_lim, verbose=True
            )
            precision_at_thresholds.append(avg_pre)

        mmap = np.mean(precision_at_thresholds)
        print("=" * 40)
        print(f"Final mmAP@50:95: {mmap:.4f}")
        print("=" * 40)

    else:
        print("--- Single IoU Evaluation ---")
        process_dataset(
            args.pred, args.gt, args.output, args.iou, args.recall_lim, verbose=True
        )


if __name__ == "__main__":
    main()
