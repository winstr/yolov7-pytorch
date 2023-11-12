from setuptools import setup, find_packages


with open("requirements.txt") as f:
    required = f.read().splitlines()

setup(
    name="yolov7",
    version="0.1.0",
    packages=find_packages(where="yolov7"),
    install_requires=required
)
