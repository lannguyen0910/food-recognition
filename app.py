from PIL import Image
from flask import Flask, request, Response, jsonify, send_from_directory, abort, render_template
from io import BytesIO
import aiohttp
import asyncio
import sys
import os
import imageio
from pathlib import Path
from werkzeug.utils import secure_filename


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
    print('File path: ', filepath)
    f.save(filepath)
    # img_data = await request.form()
    # print('Img data: ', img_data)
    # img_bytes = await (img_data['file'].read())
    # filename = img_data['file'].filename
    # img = Image.open(BytesIO(img_bytes))
    # prediction = learn.predict(img)[0]
    # return JSONResponse({'result': str(prediction)})
    # html_file = path / 'template' / 'index.html'
    return render_template("detect.html", fname=filename)


if __name__ == '__main__':
    app.run(port=4000, debug=True)
