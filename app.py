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
from pathlib import Path
from werkzeug.utils import secure_filename
from modules import get_prediction
from flask_ngrok import run_with_ngrok

parser = argparse.ArgumentParser('YOLOv5 Online Food Recognition')
parser.add_argument('--type', type=str, default='local', help="Run on local or ngrok")
parser.add_argument('--host',  type=str, default='192.168.100.4', help="Local IP")



app = Flask(__name__, template_folder='templates', static_folder='assets')
UPLOAD_FOLDER = './assets/uploads'
DETECTION_FOLDER = './assets/detections'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['DETECTION_FOLDER'] = DETECTION_FOLDER

path = Path(__file__).parent


@app.route('/')
def homepage():
    # html_file = path / 'template' / 'index.html'
    # print(html_file)
    return render_template("index.html")


@app.route('/analyze', methods=['POST'])
def analyze():
    f = request.files['file']
    print('F: ', f)
    # create a secure filename
    filename = secure_filename(f.filename)
    print('Filename: ', filename)
    # save file to /static/uploads
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    output_path=os.path.join(app.config['DETECTION_FOLDER'], filename)
    print('File path: ', filepath)
    f.save(filepath)

    filename2, result_dict = get_prediction(filepath, output_path, model_name="yolov5m")

    # img_data = await request.form()
    # print('Img data: ', img_data)
    # img_bytes = await (img_data['file'].read())
    # filename = img_data['file'].filename
    # img = Image.open(BytesIO(img_bytes))
    # prediction = learn.predict(img)[0]
    # return JSONResponse({'result': str(prediction)})
    # html_file = path / 'template' / 'index.html'
    return render_template("detect.html", fname=filename, fname2=filename2, result_dict=result_dict)


if __name__ == '__main__':
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    if not os.path.exists(DETECTION_FOLDER):
        os.makedirs(DETECTION_FOLDER, exist_ok=True)

    args = parser.parse_args()

    if args.type == 'ngrok':
        run_with_ngrok(app)
    else:
        app.run(host=args.host, port=4000, debug=True)
