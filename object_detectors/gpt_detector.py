#!/usr/bin/env python3
"""
Description:
    This script processes a directory of images, sends them to the OpenAI
    GPT API (e.g., GPT-4o) to detect objects (cars, pedestrians, signs, etc.),
    and assesses if they are safety-critical. The output is saved as a structured
    text file compatible with label formats (Class X Y W H Safety).

    It recursively scans the input directory and mirrors the folder structure
    in the output directory.

Usage:
    python gpt_detector.py --input <path_to_images> --output <path_to_results> [options]

Arguments:
    -i, --input       Path to the root folder containing input images (.jpg).
    -o, --output      Path to the root folder where .txt files will be saved.
    -m, --model       OpenAI model version (default: gpt-4o).
    -k, --key         OpenAI API Key (optional if OPENAI_API_KEY env var is set)
    -l, --limit       Maximum number of images to process (default: 1000)
"""

import os
import re
import json
import time
import random
import base64
import argparse
import numpy as np
from pathlib import Path
from colorama import Fore, Style, init

from openai import OpenAI, APIError

# colorama for colored terminal output
init(autoreset=True)

# -----------------------------------------------------------------------------
# PROMPT CONFIGURATION
# -----------------------------------------------------------------------------

SYSTEM_PROMPT = """As an image recognition expert, your task is to analyse images from dashcam footage and provide output in JSON format with the following keys only: class, x_center, y_center, width, length and safety critical.

- class represents the class of an object which divided like this: 0 car, 1 bus, 2 truck, 3 train, 4 bike, 5 motor, 6 person, 7 rider, 8 traffic sign, 9 traffic light, and 10 traffic cone
- the coordinate of object inside of given picture should be normalized to 0-1, with the reference point [0,0] at the top left corner and the reference point [1,1] at the bottom right corner of the picture
- x center and y center should represent the coordinates of the center of the detected object within the image
- width and length represent a bounding box that frames exactly the outline of one object
- safety critical represent whether the object is safety critical for the driver proceed
- to give you reference of coordinates, rulers with a marker every 1/10th of width and height are drawn on top and left of image and the whole picture is being framed

Each {} represents an object in the picture. Please adhere strictly to this output structure:
[
{
  "class": int value,
  "x_center": float value to two decimal places,
  "y_center": float value to two decimal places,
  "width": float value to two decimal places,
  "length": float value to two decimal places,
  "safety_critical": int value
}
]
Note: Do not include any additional data or keys outside of what has been specified.
"""

USER_INSTRUCTION = "Please analyse this dash-cam picture carefully and give all objects you can recognise."


# -----------------------------------------------------------------------------
# CORE FUNCTIONS
# -----------------------------------------------------------------------------

def encode_image(image_path):
    """
    Encodes an image file to a base64 string required by OpenAI Vision API.
    """
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def call_gpt(client, image_path, system_prompt, user_text, model, max_retries=3):
    """
    Calls the OpenAI API with exponential backoff for retries.
    """
    retries = 0
    base_delay = 1  # in seconds

    while retries < max_retries:
        try:
            base64_image = encode_image(image_path)

            response = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": system_prompt
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": user_text
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                },
                            },
                        ],
                    }
                ],
                max_tokens=2000
            )

            return response.choices[0].message.content

        except APIError as e:
            retries += 1
            if retries >= max_retries:
                print(
                    Fore.RED + Style.BRIGHT + f"[X] GPT API call failed after {max_retries} retries. Final error: {e}")
                return None

            delay = base_delay * (2 ** retries) + random.uniform(0, 1)
            print(
                Fore.YELLOW + Style.BRIGHT + f"[!] APIError: {e}. Retrying in {delay:.2f} seconds... (Attempt {retries}/{max_retries})")
            time.sleep(delay)

        except Exception as e:
            print(Fore.RED + Style.BRIGHT + f"[X] An unexpected error occurred: {e}")
            return None

    return None


def parse_and_save_response(raw_data, output_path):
    """
    Parses the raw JSON string from GPT, validates the schema,
    and writes the formatted .txt file.
    """
    try:
        data = json.loads(raw_data)
        content = data.get("content", "")
        # regex to find the JSON array in case the model adds markdown text around it
        match = re.search(r"\[.*\]", content, re.DOTALL)
        if not match:
            print(Fore.RED + Style.BRIGHT + "[X] Error:" + Style.RESET_ALL + " No JSON array found in the raw data.")
            return False

        json_array = match.group(0)
        cleared_data = json.loads(json_array)

        formatted_lines = []

        try:
            for item in cleared_data:
                class_id = item["class"]
                x = item["x_center"]
                y = item["y_center"]
                w = item["width"]
                l = item["length"]
                sc_raw = item["safety_critical"]
                sc = int(sc_raw) if isinstance(sc_raw, (int, float)) else (1 if sc_raw else 0)

                formatted_lines.append(f"{class_id} {x} {y} {w} {l} {sc}")

        except KeyError as e:
            print(Fore.RED + Style.BRIGHT + "[X]" + Style.RESET_ALL + f" Caught a KeyError (Missing Data): {e}")
            return False

        if formatted_lines:
            result_content = "\n".join(formatted_lines)

            with open(output_path, "w", encoding='utf-8') as o_file:
                o_file.write(result_content)

            return True
        else:
            print(Fore.RED + Style.BRIGHT + "[X]" + Style.RESET_ALL + " No valid data extracted to write to file.")
            return False

    except json.JSONDecodeError:
        print(Fore.RED + Style.BRIGHT + "[X] Error: " + Style.RESET_ALL + " Failed to decode JSON.")
        return False
    except Exception as e:
        print(Fore.RED + Style.BRIGHT + "[X]" + Style.RESET_ALL + f" Unexpected parsing error: {e}")
        return False


# -----------------------------------------------------------------------------
# MAIN LOOP
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="GPT Dashcam Analyzer")
    parser.add_argument("--input", "-i", required=True, help="Root input directory containing images")
    parser.add_argument("--output", "-o", required=True, help="Root output directory for text files")
    parser.add_argument("--model", "-m", default="gpt-4o", help="OpenAI Model ID")
    parser.add_argument("--limit", "-l", type=int, default=1000, help="Max number of images to process")
    parser.add_argument("--key", "-k", help="OpenAI API Key (Optional if env var is set)")

    args = parser.parse_args()

    # setup API key
    api_key = args.key or os.environ.get("OPENAI_API_KEY") or os.environ.get("GPT_API_KEY")
    if not api_key:
        print(Fore.RED + "Error: API Key not found. Please set OPENAI_API_KEY environment variable or use --key flag.")
        return

    client = OpenAI(api_key=api_key)

    input_root = Path(args.input)
    output_root = Path(args.output)

    if not input_root.exists():
        print(Fore.RED + f"Error: Input directory {input_root} does not exist.")
        return

    processed_count = 0
    process_times = np.array([])

    print(Fore.CYAN + Style.BRIGHT + f"Starting Processing...")
    print(f"Model: {args.model}")
    print(f"Reading from: {input_root}")
    print(f"Saving to:    {output_root}")
    print("-" * 60)

    for root, _, files in os.walk(input_root):
        if processed_count >= args.limit:
            print(Fore.RED + Style.BRIGHT + f"Limit of {args.limit} images reached!")
            break

        image_files = [f for f in files if f.lower().endswith(('.jpg', '.png', '.jpeg'))]

        for file in image_files:
            if processed_count >= args.limit:
                break

            input_file_path = Path(root) / file
            relative_path = input_file_path.relative_to(input_root)

            output_folder = output_root / relative_path.parent
            output_folder.mkdir(parents=True, exist_ok=True)
            output_file_path = output_folder / (input_file_path.stem + ".txt")

            # check if output already exists
            if output_file_path.exists():
                print(Fore.GREEN + "Skipped: " + Style.RESET_ALL + f"{file} already exists")
                continue

            print("-" * 60)
            print(f"Processing: {relative_path}")

            call_time_start = time.time()
            raw_data = call_gpt(
                client,
                input_file_path,
                SYSTEM_PROMPT,
                USER_INSTRUCTION,
                args.model
            )

            if raw_data is None:
                print(Fore.RED + Style.BRIGHT + "[X]" + Style.RESET_ALL + " API response was None. Skipping.")
                continue

            success = parse_and_save_response(raw_data, output_file_path)

            call_time_end = time.time()
            duration = call_time_end - call_time_start
            process_times = np.append(process_times, duration)

            if success:
                processed_count += 1
                print(Fore.GREEN + Style.BRIGHT + "[OK]" + Style.RESET_ALL +
                      f" Finished {processed_count} | Time: {duration:.2f}s | Saved to: {output_file_path.name}")
            else:
                print(Fore.RED + f"Failed to save results for {file}")

    # final stats
    print("=" * 60)
    if len(process_times) > 0:
        time_mean = np.mean(process_times)
        print(Fore.CYAN + Style.BRIGHT + f"Job Complete.")
        print(f"Total processed: {processed_count}")
        print(f"Mean process time: {time_mean:.2f} seconds")
    else:
        print("No images were processed.")


if __name__ == "__main__":
    main()
