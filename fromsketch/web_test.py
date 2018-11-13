# encoding: utf-8
"""
@author: xyliao
@contact: xyliao1993@qq.com
"""
import base64
import requests
import argparse
import json
import os

# Initialize the PyTorch REST API endpoint URL.
PyTorch_REST_API_URL = 'http://127.0.0.1:5000/fromsketch'


def predict_result(image_path):
    # Initialize image path
    image = open(image_path, 'rb').read()
    payload = {'image': image}
    r=requests.post(PyTorch_REST_API_URL,files=payload)
    with open(os.path.join(os.path.dirname(__file__),"output/images/test.jpeg"),'wb') as str2image:
        str2image.write(r.content)
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Classification demo')
    parser.add_argument('--file', type=str, help='test image file')
    args = parser.parse_args()
    predict_result(args.file)
