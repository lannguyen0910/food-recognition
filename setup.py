# Adapted from: https://github.com/serengil/deepface/blob/master/setup.py
import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", 'r') as f:
    reqs = f.read().splitlines()

setuptools.setup(
    name="food-recognition",
    version='0.0.1',
    author="Hoang-Lan Nguyen",
    author_email="nhlan091000@gmail.com",
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
    install_requires=reqs,
)
