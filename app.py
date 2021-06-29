from genericpath import exists
from PIL import Image
from flask import Flask, request, Response, jsonify, send_from_directory, abort, render_template
from io import BytesIO
import aiohttp
import asyncio
import sys
import os
import imageio
import argparse
import requests
import cv2
import numpy as np
from pathlib import Path
from werkzeug.utils import secure_filename
from modules import get_prediction
import hashlib
from flask_ngrok import run_with_ngrok
from flask_cors import CORS
from flask import render_template, redirect, url_for, flash, request, Response
import os
from werkzeug.utils import secure_filename
import requests
import tldextract
import pytube


parser = argparse.ArgumentParser('YOLOv5 Online Food Recognition')
parser.add_argument('--ngrok', action='store_true',
                    default=False, help="Run on local or ngrok")
parser.add_argument('--host',  type=str,
                    default='192.168.100.4:4000', help="Local IP")
parser.add_argument('--debug', action='store_true',
                    default=False, help="Run app in debug mode")


app = Flask(__name__, template_folder='templates', static_folder='static')
cors = CORS(app, resources={r"/api/*": {"origins": "*"}})

app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 1


UPLOAD_FOLDER = './static/assets/uploads'
VIDEO_FOLDER = './static/assets/videos'
DETECTION_FOLDER = './static/assets/detections'

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['DETECTION_FOLDER'] = DETECTION_FOLDER
app.config['VIDEO_FOLDER'] = VIDEO_FOLDER

IMAGE_ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
VIDEO_ALLOWED_EXTENSIONS = {'mp4'}


def allowed_file_image(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in IMAGE_ALLOWED_EXTENSIONS


def allowed_file_video(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in VIDEO_ALLOWED_EXTENSIONS


def make_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def download_yt(url):
    youtube = pytube.YouTube(url)
    video = youtube.streams.first()
    path = video.download(app.config['VIDEO_FOLDER'])

    return path


def download(url):
    ext = tldextract.extract(url)
    if ext.domain == 'youtube':
        try:
            make_dir(app.config['VIDEO_FOLDER'])
        except:
            pass
        print('Youtube')
        path = download_yt(url)
    else:
        make_dir(app.config['UPLOAD_FOLDER'])
        filename = url.split('/')[-1]
        path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_2)',
                   'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                   'Accept-Charset': 'ISO-8859-1,utf-8;q=0.7,*;q=0.3',
                   'Accept-Encoding': 'none',
                   'Accept-Language': 'en-US,en;q=0.8',
                   'Connection': 'keep-alive'}
        r = requests.get(url, stream=True, headers=headers)
        with open(path, "wb") as file:
            file.write(r.content)

    return path


def save_upload(file):
    filename = secure_filename(file.filename)
    if allowed_file_image(filename):
        make_dir(app.config['UPLOAD_FOLDER'])
        path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    elif allowed_file_video(filename):
        try:
            make_dir(app.config['VIDEO_FOLDER'])
        except:
            pass
        path = os.path.join(app.config['VIDEO_FOLDER'], filename)
    file.save(path)

    return path


path = Path(__file__).parent


@app.route('/')
def homepage():
    return render_template("index.html")


@app.route('/analyze', methods=['POST'])
def analyze():
    f = request.files['file']
    iou = request.form.get('threshold-range')
    confidence = request.form.get('confidence-range')
    model_types = request.form.get('model-types')
    tta = request.form.get('tta')
    ensemble = request.form.get('ensemble')

    print('iou: ', iou)
    print('confidence: ', confidence)
    print('model_types: ', model_types)
    print('tta: ', tta)
    print('ensemble: ', ensemble)

    ori_file_name = secure_filename(f.filename)
    _, ext = os.path.splitext(ori_file_name)

    # Get cache name by hashing image
    data = f.read()
    filename = hashlib.md5(data).hexdigest() + f'{ext}'

    # save file to /static/uploads
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    np_img = np.fromstring(data, np.uint8)
    img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
    cv2.imwrite(filepath, img)

    # predict image
    output_path = os.path.join(app.config['DETECTION_FOLDER'], filename)
    filename2, result_dict = get_prediction(
        filepath,
        output_path,
        model_name="yolov5m",
        ensemble=False,
        min_conf=0.25,
        min_iou=0.65)

    return render_template("detect.html", fname=filename, fname2=filename, result_dict=result_dict)


@app.after_request
def add_header(response):
    if 'Cache-Control' not in response.headers:
        response.headers['Cache-Control'] = 'public, no-store, no-cache, must-revalidate, post-check=0, pre-check=0, max-age=0'
        response.headers['Pragma'] = 'no-cache'
        response.headers['Expires'] = '-1'
    return response


if __name__ == '__main__':
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    if not os.path.exists(DETECTION_FOLDER):
        os.makedirs(DETECTION_FOLDER, exist_ok=True)

    args = parser.parse_args()

    if args.ngrok:
        run_with_ngrok(app)
        app.run()
    else:
        hostname = str.split(args.host, ':')
        if len(hostname) == 1:
            port = 4000
        else:
            port = hostname[1]
        host = hostname[0]
        app.run(host=host, port=port, debug=args.debug)
