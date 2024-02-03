# Phase1-DRL

Using the files from Kirkados' JSR2020_D4PG as a base for DRL-based guidance of spacecraft proximity operations trained via D4PG.

Set-up coding environment as described below. This has been tested using Python 3.6.8 on Windows10 and Python 3.6.13 in a Conda environment on Ubuntu 20.04.6 LTS.

    pip install tensorflow-gpu==1.12.0
    pip install tensorflow==1.12.0
    pip install protobuf==3.19.6
    pip install psutil
    pip install pyvirtualdisplay
    pip install scipy==1.5.2
    pip install matplotlib
    pip uninstall numpy
    pip install numpy==1.16.4
    pip install ffmpeg-python
    pip install shapely==1.8.5.post1

In Windows: extract, add bin folder to PATH: https://github.com/GyanD/codexffmpeg/releases/tag/2022-03-28-git-5ee198f9aa

To train a model, modify settings in environment_envs123456.py and settings.py. Then run 

Windows:
    
    py main.py

Ubuntu:

    python main.py

