import setuptools

setuptools.setup(
    name="theseus",
    version='0.0.1',
    packages=setuptools.find_packages(),
    python_requires='>=3.7',
    install_requires=[
        "numpy",
        "torch",        
        "gunicorn==20.1.0"
        "albumentations==0.5.2"
        "pyyaml>=5.1"
        "webcolors"
        "tensorboard"
        "tqdm"
        "ensemble-boxes"
        "timm"
        "omegaconf"
        "pycocotools"
        "gdown==4.3.0"
        "flask-cors"
        "flask_ngrok"
        "cryptography"
        "tabulate"
        "segmentation-models-pytorch"
        "opencv-python-headless==4.2.0.32 "
    ],
)