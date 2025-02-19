# Liveness Detection Algorithm

## Overview
This project implements a Liveness Detection algorithm using Python to prevent spoofing during a face recognition process. The goal is to determine whether the face presented is real or a static photo/video.

## Challenge Description
The challenge requires the development of a liveness detection system that classifies the input as either "Real" or "Fake". The solution should include:

- **Passive Detection**: Detect if the user is holding a static photo or a phone displaying an image/video.
- **Active Detection**: Optionally include an active function where a random question is posed on the screen, and inference is made based on the user's response.

## Demo
![Liveness Detection Demo](https://github.com/revaldianggara/Liveness-Detection-Algorithm/blob/main/demo2.gif)

## Workflows
![Workflows](https://github.com/revaldianggara/Liveness-Detection-Algorithm/blob/main/workflows.png)

## Requirements
- Python 3.11

## Installation
1. Clone the repository:
```bash
git clone https://github.com/revaldianggara/Liveness-Detection-Algorithm.git
cd Liveness-Detection-Algorithm
```

2.  install required package
```bash
pip install -r requirements.txt
```

3. Run Data Pipeline: Download the required Dlib predictor and the raw NUAA Face Anti-Spoofing Dataset by running:
```bash
python get_data_predictor.py 
```

4. Evaluate the Algorithm: Run the evaluation script to test the algorithm's performance on the dataset:
```bash
python evaluate.py 
```

5. Real-Time Camera Testing:
```bash
python main.py 
```

## Usage
   - Press 'a' to enable Active Mode.

   - Press 'p' to enable Passive Mode.

   - Press 'q'to quit the program.

## Evaluation Result
   ### Dataset Path
   Dataset Path: data/test

   ### Confusion Matrix
   ![Confusion Matrix](https://github.com/revaldianggara/Liveness-Detection-Algorithm/blob/main/confusion_matrix.png)


   ### Detailed Classification Report
   |              | precision | recall | f1-score | support |
   |--------------|-----------|--------|----------|----------|
   | Fake         | 1.00      | 0.87   | 0.93     | 45      |
   | Real         | 0.88      | 1.00   | 0.94     | 45      |
   | accuracy     |           |        | 0.93     | 90      |
   | macro avg    | 0.94      | 0.93   | 0.93     | 90      |
   | weighted avg | 0.94      | 0.93   | 0.93     | 90      |

   ### Summary Matrix
   Total Images Processed: 90
   - Accuracy: 0.9333
   - Precision: 0.8824 
   - Recall: 1.0000
   - F1 Score: 0.9375

## About Dataset
Face Anti-Spoofing Dataset

X.Tan, Y.Li, J.Liu and L.Jiang.
Face Liveness Detection from A Single Image with Sparse Low Rank Bilinear Discriminative Model,
In: Proceedings of 11th European Conference on Computer Vision (ECCV'10), Crete, Greece. September 2010

![dataset-overview](https://github.com/revaldianggara/Liveness-Detection-Algorithm/blob/main/dataset.png)

source dataset and predictor: 
   1. https://www.researchgate.net/publication/262405100_Face_Liveness_Detection_from_a_Single_Image_with_Sparse_Low_Rank_Bilinear_Discriminative_Model
   2. https://www.kaggle.com/datasets/aleksandrpikul222/nuaaaa
   3. https://www.kaggle.com/datasets/sergiovirahonda/shape-predictor-68-face-landmarksdat
