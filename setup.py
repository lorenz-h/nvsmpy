from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup_kwargs = {
    "name": "nvsmpy",
    "version": "0.2.2",
    "author": "lorenz-h",
    "author_email": "lorenz.hetzel@yahoo.de",
    "description": "Find unoccupied GPUs on multi-user systems",
    "long_description": long_description,
    "long_description_content_type": "text/markdown",
    "url": "https://github.com/lorenz-h/nvsmpy",
    "packages": find_packages(),
    "classifiers": [
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
    ],
    "install_requires": [
        "psutil",
        "pynvml"
    ],
    "python_requires": '>:3.5',
}

setup(**setup_kwargs)
