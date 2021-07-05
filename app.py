import aiohttp
import asyncio
import sys
import os
import argparse
import requests
import cv2
import numpy as np
import tldextract
import pytube
import hashlib
import tldextract
import pytube

from genericpath import exists
from PIL import Image
from flask import Flask, request, render_template, redirect, url_for
from io import BytesIO
from pathlib import Path
from werkzeug.utils import secure_filename
from modules import get_prediction, get_video_prediction
from flask_ngrok import run_with_ngrok
from flask_cors import CORS
from werkzeug.utils import secure_filename


parser = argparse.ArgumentParser('YOLOv5 Online Food Recognition')
parser.add_argument('--ngrok', action='store_true',
                    default=False, help="Run on local or ngrok")
parser.add_argument('--host',  type=str,
                    default='192.168.100.4:4000', help="Local IP")
parser.add_argument('--debug', action='store_true',
                    default=False, help="Run app in debug mode")


app = Flask(__name__, template_folder='templates', static_folder='static')
CORS(app, resources={r"/api/*": {"origins": "*"}})

app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 1


UPLOAD_FOLDER = './static/assets/uploads'
CSV_FOLDER = './static/csv'
VIDEO_FOLDER = './static/assets/videos'
DETECTION_FOLDER = './static/assets/detections'

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['CSV_FOLDER'] = CSV_FOLDER
app.config['DETECTION_FOLDER'] = DETECTION_FOLDER
app.config['VIDEO_FOLDER'] = VIDEO_FOLDER

IMAGE_ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
VIDEO_ALLOWED_EXTENSIONS = {'mp4', 'avi', '3gp'}


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

def hash_video(video_path):
    _, ext = os.path.splitext(video_path)
    stream = cv2.VideoCapture(video_path)
    success, ori_frame = stream.read()
    stream.release()
    stream=None
    image_bytes = cv2.imencode('.jpg', ori_frame)[1].tobytes()
    filename = hashlib.md5(image_bytes).hexdigest() + f'{ext}'
    return filename

def download(url):
    ext = tldextract.extract(url)
    if ext.domain == 'youtube':
        try:
            make_dir(app.config['VIDEO_FOLDER'])
        except:
            pass
        print('Youtube')
        ori_path = download_yt(url)
        filename = hash_video(ori_path)

        path = os.path.join(app.config['VIDEO_FOLDER'], filename)
        try:
            os.rename(ori_path, path)
        except:
            pass
    else:
        make_dir(app.config['UPLOAD_FOLDER'])
        headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_2)',
                   'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                   'Accept-Charset': 'ISO-8859-1,utf-8;q=0.7,*;q=0.3',
                   'Accept-Encoding': 'none',
                   'Accept-Language': 'en-US,en;q=0.8',
                   'Connection': 'keep-alive'}
        r = requests.get(url, stream=True, headers=headers)

        # Get cache name by hashing image
        data = r.content
        ori_filename = url.split('/')[-1]
        _, ext = os.path.splitext(ori_filename)
        filename = hashlib.md5(data).hexdigest() + f'{ext}'

        path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

        with open(path, "wb") as file:
            file.write(r.content)

    return filename, path


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


def file_type(path):
    filename = path.split('/')[-1]
    if allowed_file_image(filename):
        filetype = 'image'
    elif allowed_file_video(filename):
        filetype = 'video'
    else:
        filetype = 'invalid'
    return filetype


path = Path(__file__).parent


@app.route('/')
def homepage():
    return render_template("index.html")


@app.route('/about')
def about_page():
    return render_template("about.html")


@app.route('/url')
def detect_by_url_page():
    return render_template("url.html")


@app.route('/analyze', methods=['POST'])
def analyze():
    if request.method == 'POST':
        result_dict=None
        filename = None
        file_type=None
        if 'url-button' in request.form:
            url = request.form['url_link']
            filename, filepath = download(url)
            print('Filename', filename)
            print('Upload filepath', filepath)
            filetype = file_type(filepath)

        if 'upload-button' in request.form:
            f = request.files['file']
            ori_file_name = secure_filename(f.filename)
            _, ext = os.path.splitext(ori_file_name)
            filetype = file_type(ori_file_name)
            
            if filetype == 'image':
                # Get cache name by hashing image
                data = f.read()
                filename = hashlib.md5(data).hexdigest() + f'{ext}'

                # save file to /static/uploads
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                np_img = np.fromstring(data, np.uint8)
                img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
                cv2.imwrite(filepath, img)
            elif filetype == 'video':
                temp_filepath = os.path.join(app.config['UPLOAD_FOLDER'], ori_file_name)
                f.save(temp_filepath)
                filename = hash_video(temp_filepath)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                os.rename(temp_filepath, filepath)

        iou = request.form.get('threshold-range')
        confidence = request.form.get('confidence-range')
        model_types = request.form.get('model-types')
        enhanced = request.form.get('enhanced')
        ensemble = request.form.get('ensemble')
        ensemble = True if ensemble == 'on' else False
        enhanced = True if enhanced == 'on' else False
        model_types = str.lower(model_types)
        min_conf = float(confidence)/100
        min_iou = float(iou)/100

        if filetype == 'image':
            out_name = "Image Result"
            output_path = os.path.join(
                app.config['DETECTION_FOLDER'], filename)

            filename2, result_dict = get_prediction(
                filepath,
                output_path,
                model_name=model_types,
                ensemble=ensemble,
                min_conf=min_conf,
                min_iou=min_iou,
                enhance_labels=enhanced)

        elif filetype == 'video':
            out_name = "Video Result"
            output_path = os.path.join(
                app.config['DETECTION_FOLDER'], filename)
            get_video_prediction(
                filepath,
                output_path,
                model_name=model_types,
                min_conf=min_conf,
                min_iou=min_iou,
                enhance_labels=enhanced)
        else:
            error_msg = "Invalid input url!!!"
            return render_template('detect_url.html', error_msg=error_msg)
        
        if 'url-button' in request.form:
            return render_template('detect_url.html', out_name=out_name, fname=filename, filetype=filetype)

        return render_template("detect.html", filetype=filetype, fname=filename, fname2=filename, result_dict=result_dict)

    return redirect("/")


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
    if not os.path.exists(VIDEO_FOLDER):
        os.makedirs(VIDEO_FOLDER, exist_ok=True)
    if not os.path.exists(CSV_FOLDER):
        os.makedirs(CSV_FOLDER, exist_ok=True)

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
        app.run(host=host, port=port, debug=args.debug, use_reloader=False)
