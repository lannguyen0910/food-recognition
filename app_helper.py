from app import app
from flask import render_template, redirect, url_for, flash, request, Response
import os
from werkzeug.utils import secure_filename
import requests
import tldextract
import pytube

UPLOAD_FOLDER = './assets/uploads'
VIDEO_FOLDER = './assets/videos'
DETECTION_FOLDER = './assets/detections'

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
