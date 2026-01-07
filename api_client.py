#!/usr/bin/env python3
"""
Simple client to call RT-1 API with batch prediction.
"""

import requests
import json
import base64
import argparse

def encode_image_to_base64(image_path):
    """Encode image to base64 string."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def predict_bbox(api_url, image_path, text_instruction="Skewer the food item", action=None, food_type=None):
    """Send batch prediction request to the API."""
    image_base64 = encode_image_to_base64(image_path)
    
    data = {
        'image': image_base64,
        'text': text_instruction,
        'batch_size': 16,
        'enable_dropout': True
    }
    
    if action:
        data['action'] = action
    if food_type:
        data['food_type'] = food_type
    
    response = requests.post(f"{api_url}/predict_batch", json=data)
    
    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(f"API call failed: {response.status_code} - {response.text}")

def main():
    parser = argparse.ArgumentParser(description='Call RT-1 API')
    parser.add_argument('--api_url', default='https://b66c5e1f834a.ngrok-free.app', help='API server URL')
    parser.add_argument('--image', required=True, help='Path to input image')
    parser.add_argument('--text', default='Skewer the food item', help='Text instruction')
    parser.add_argument('--action', help='Action type (skewering, scooping, twirling)')
    parser.add_argument('--food_type', help='Food type (chicken, banana, etc.)')
    
    args = parser.parse_args()
    
    try:
        result = predict_bbox(args.api_url, args.image, args.text, args.action, args.food_type)
        print(json.dumps(result, indent=2))
    except Exception as e:
        print(f"Error: {e}")

if __name__ == '__main__':
    main()
