# Liveness Detection Algorithm

## Demo
![Liveness Detection Demo](https://github.com/revaldianggara/Liveness-Detection-Algorithm/blob/main/demo2.gif)

## Workflows
![Workflows](https://github.com/revaldianggara/Liveness-Detection-Algorithm/blob/main/workflows.png)

## Overview
This project implements a Liveness Detection algorithm using Python to prevent spoofing during a face recognition process. The goal is to determine whether the face presented is real or a static photo/video.

## Challenge Description
The challenge requires the development of a liveness detection system that classifies the input as either "Real" or "Fake". The solution should include:

- **Passive Detection**: Detect if the user is holding a static photo or a phone displaying an image/video.
- **Active Detection**: Optionally include an active function where a random question is posed on the screen, and inference is made based on the user's response.

## Requirements
- Python 3.11

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/revaldianggara/Liveness-Detection-Algorithm.git
   cd Liveness-Detection-Algorithm

2. pip install -r requirements.txt

3. python get_data_predictor.py 
this script for download dlib sahep predicor dataset nuaaa

4. running
for evaluate metrics accuracy for liveness detection algorithm

## Run Code
```bash
python main.py --active_mode False
```
Press 'a' to enable Active Mode.
Press 'p' to enable Passive Mode.
Press 'q'to quit the program.

## About Dataset
Face Anti-Spoofing Dataset

X.Tan, Y.Li, J.Liu and L.Jiang.
Face Liveness Detection from A Single Image with Sparse Low Rank Bilinear Discriminative Model,
In: Proceedings of 11th European Conference on Computer Vision (ECCV'10), Crete, Greece. September 2010

![Workflows](https://github.com/revaldianggara/Liveness-Detection-Algorithm/blob/main/dataset.png)

source: 
   1. https://www.kaggle.com/datasets/aleksandrpikul222/nuaaaa
   2. https://www.researchgate.net/publication/262405100_Face_Liveness_Detection_from_a_Single_Image_with_Sparse_Low_Rank_Bilinear_Discriminative_Model
