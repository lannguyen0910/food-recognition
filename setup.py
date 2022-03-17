# Adapted from: https://github.com/serengil/deepface/blob/master/setup.py
import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="food-recognition",
    version='0.0.1',
    author="Hoang-Lan Nguyen",
    author_email="18120051@student.hcmus.edu.vn",
    description="A Baseline Food Recognition using Theseus - A general template for most Pytorch projects",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/lannguyen0910/food-recognition",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent"
    ],
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
