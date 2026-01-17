import os
import sys
import json
import base64
from openai import OpenAI
import cv2
import numpy as np

# System prompt from the project
SYSTEM_PROMPT = """As an image recognition expert for pictures, your task is to analyse images and provide output.

- class represents the class of an object which divided like this: 0 car, 1 bus, 2 truck, 3 train, 4 bike, 5 motor, 6 person, 7 rider, 8 traffic sign, 9 traffic light, and 10 traffic cone
- safety critical represent whether the object is safety critical for the driver, 1 for unsafe and 0 for safe
- rulers are drawn on top and left of image, to give you reference of coordinates and the whole picture is being framed

Each line represents an object in the picture. Please adhere strictly to this output structure:
[
{
  "class": value,
  "safety critical": value
}
]

Note: Do not include any additional data or keys outside of what has been specified.
"""

USER_PROMPT = "You are an very experienced driver, please analyse this dash-cam picture carefully and give all objects you can recognise."

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def main():
    if len(sys.argv) < 2:
        print("Usage: python quick_start.py <path_to_image> [api_key]")
        print("If api_key is not provided, it looks for OPENAI_API_KEY environment variable.")
        return

    image_path = sys.argv[1]
    
    if len(sys.argv) > 2:
        api_key = sys.argv[2]
    else:
        api_key = os.getenv("OPENAI_API_KEY")
        
    if not api_key:
        print("Error: No API key provided.")
        return

    if not os.path.exists(image_path):
        print(f"Error: Image file '{image_path}' not found.")
        return

    print(f"Processing image: {image_path}...")

    client = OpenAI(api_key=api_key)

    base64_image = encode_image(image_path)

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": USER_PROMPT},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=300
        )
        
        content = response.choices[0].message.content
        print("\n--- Result ---")
        print(content)
        print("--------------\n")
        
    except Exception as e:
        print(f"Error calling OpenAI API: {e}")

if __name__ == "__main__":
    main()

