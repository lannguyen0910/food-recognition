import os
import threading
import argparse
import getpass

from flask_cors import CORS
from flask import Flask
from pyngrok import ngrok, conf

from backend.routes import set_routes
from backend.constants import UPLOAD_FOLDER, CSV_FOLDER, DETECTION_FOLDER, SEGMENTATION_FOLDER, METADATA_FOLDER

parser = argparse.ArgumentParser('Online Food Recognition')
parser.add_argument('--ngrok', action='store_true',
                    default=False, help="Run on local or ngrok")
parser.add_argument('--host', type=str, default='localhost', help="Local IP")
parser.add_argument('--port', type=int, default=5000, help="Local port")
parser.add_argument('--debug', action='store_true',
                    default=False, help="Run app in debug mode")

args = parser.parse_args()

if __name__ == '__main__':
    app = Flask(__name__, template_folder='templates', static_folder='static')

    CORS(app, resources={r"/api/*": {"origins": "*"}})
    app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
    app.config['CSV_FOLDER'] = CSV_FOLDER
    app.config['DETECTION_FOLDER'] = DETECTION_FOLDER
    app.config['SEGMENTATION_FOLDER'] = SEGMENTATION_FOLDER

    if args.ngrok:
        print("Enter your authtoken, which can be copied from https://dashboard.ngrok.com/get-started/your-authtoken")
        conf.get_default().auth_token = getpass.getpass()
        public_url = ngrok.connect(args.port).public_url
        print(
            f" * ngrok tunnel \"{public_url}\" -> \"http://127.0.0.1:{args.port}/\"")
        app.config['BASE_URL'] = public_url
    else:
        app.config['BASE_URL'] = f"http://{args.host}:{args.port}"

    set_routes(app)

    for folder in [UPLOAD_FOLDER, DETECTION_FOLDER, SEGMENTATION_FOLDER, CSV_FOLDER, METADATA_FOLDER]:
        os.makedirs(folder, exist_ok=True)

    threading.Thread(target=app.run, kwargs={
        "host": args.host, "port": args.port, "debug": args.debug, "use_reloader": False}).start()
