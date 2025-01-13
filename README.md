# Liveness Detection Algorithm

## Demo
![Liveness Detection Demo](https://github.com/revaldianggara/Liveness-Detection-Algorithm/blob/main/demo.gif)

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

3. download dlib predicor: https://www.kaggle.com/datasets/sergiovirahonda/shape-predictor-68-face-landmarksdat

## Run Code
```bash
python main.py --active_mode False
```
Press 'a' to enable Active Mode.
Press 'p' to enable Passive Mode.
Press 'q'to quit the program.