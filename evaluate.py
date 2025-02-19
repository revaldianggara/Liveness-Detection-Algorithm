import os
import cv2
import numpy as np
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from main import LivenessDetection


class DatasetEvaluator:
    def __init__(self, liveness_detector, dataset_path, confidence_threshold=0.6):
        """
        Initialize the evaluator.

        Args:
            liveness_detector: Instance of your LivenessDetection class.
            dataset_path: Path to the test dataset folder with 'Real' and 'Fake' subfolders.
            confidence_threshold: Threshold for classifying images as 'Real'.
        """
        self.liveness_detector = liveness_detector
        self.dataset_path = dataset_path
        self.confidence_threshold = confidence_threshold
        self.ground_truth = []
        self.predictions = []
        self.file_paths = []  # Store file paths for failed predictions analysis
        self.confidences = []  # Store confidence scores

    def evaluate(self):
        """
        Evaluate the liveness detection algorithm on the dataset.
        """
        print("Starting evaluation...")

        for label in ["Real", "Fake"]:
            folder_path = os.path.join(self.dataset_path, label)
            if not os.path.exists(folder_path):
                print(f"Warning: Folder not found - {folder_path}")
                continue

            print(f"\nProcessing {label} images...")
            for filename in os.listdir(folder_path):
                file_path = os.path.join(folder_path, filename)
                print(f"Processing: {filename}")

                image = cv2.imread(file_path)
                if image is None:
                    print(f"Could not read image {file_path}")
                    continue

                try:
                    # Process image through the liveness detection pipeline
                    results = self.liveness_detector.analyze_frame(image)

                    if len(results) > 0:
                        _, prediction, confidence = results[0]
                        
                        # Apply confidence threshold
                        if prediction == "Real" and confidence < self.confidence_threshold:
                            prediction = "Fake"  # Reclassify as 'Fake' if confidence is low
                        
                        self.predictions.append(prediction)
                        self.confidences.append(confidence)
                    else:
                        print(f"No face detected in {filename}, marking as Fake")
                        self.predictions.append("Fake")
                        self.confidences.append(0.0)

                    # Store the ground truth and file path
                    self.ground_truth.append(label)
                    self.file_paths.append(file_path)

                except Exception as e:
                    print(f"Error processing {filename}: {str(e)}")
                    continue

        # Calculate and display metrics
        self._calculate_metrics()

    def plot_confusion_matrix(self):
        """
        Plot and save confusion matrix as a heatmap.
        """
        cm = confusion_matrix(self.ground_truth, self.predictions)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Fake', 'Real'],
                    yticklabels=['Fake', 'Real'])
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')

        # Save the plot
        plt.savefig('confusion_matrix.png')
        plt.close()

    def analyze_failures(self):
        """
        Analyze and log failed predictions.
        """
        failed_cases = []
        for gt, pred, confidence, file_path in zip(self.ground_truth, self.predictions, self.confidences, self.file_paths):
            if gt != pred:
                failed_cases.append({
                    'file': os.path.basename(file_path),
                    'true_label': gt,
                    'predicted': pred,
                    'confidence': confidence
                })

                # Save misclassified images for further analysis
                self._save_failed_image(file_path, pred, gt)

        return failed_cases

    def _save_failed_image(self, file_path, predicted_label, true_label):
        """
        Save misclassified images to a separate folder for debugging.
        """
        failed_dir = "failed_cases"
        os.makedirs(failed_dir, exist_ok=True)

        # Create subfolder for each true label
        subfolder = os.path.join(failed_dir, f"{true_label}_as_{predicted_label}")
        os.makedirs(subfolder, exist_ok=True)

        # Save the image
        filename = os.path.basename(file_path)
        save_path = os.path.join(subfolder, filename)
        image = cv2.imread(file_path)
        cv2.imwrite(save_path, image)

    def _calculate_metrics(self):
        """
        Calculate and print evaluation metrics.
        """
        # Create timestamp for the log file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_filename = f"evaluation_metrics_{timestamp}.txt"

        # Calculate metrics
        accuracy = accuracy_score(self.ground_truth, self.predictions)
        precision = precision_score(self.ground_truth, self.predictions, pos_label='Real')
        recall = recall_score(self.ground_truth, self.predictions, pos_label='Real')
        f1 = f1_score(self.ground_truth, self.predictions, pos_label='Real')

        # Generate classification report
        class_report = classification_report(self.ground_truth, self.predictions,
                                             target_names=["Fake", "Real"])

        # Analyze failed cases
        failed_cases = self.analyze_failures()

        # Create detailed log content
        log_content = [
            "=== Liveness Detection Evaluation Results ===",
            f"\nTimestamp: {timestamp}",
            f"\nDataset Path: {self.dataset_path}",
            f"\nTotal Images Processed: {len(self.ground_truth)}",
            f"\nMetrics:",
            f"Accuracy: {accuracy:.4f}",
            f"Precision: {precision:.4f}",
            f"Recall: {recall:.4f}",
            f"F1 Score: {f1:.4f}",
            f"\nDetailed Classification Report:",
            class_report,
            f"\nFailed Cases Analysis:",
            f"Total Failed Predictions: {len(failed_cases)}",
            "\nDetailed Failed Cases:"
        ]

        # Add failed cases details
        for case in failed_cases:
            log_content.append(
                f"\nFile: {case['file']}"
                f"\nTrue Label: {case['true_label']}"
                f"\nPredicted: {case['predicted']}"
                f"\nConfidence: {case['confidence']:.2f}"
            )

        # Write to log file
        with open(log_filename, "w") as log_file:
            log_file.write("\n".join(log_content))

        # Print summary to console
        print("\n=== Evaluation Results ===")
        print(f"Total Images Processed: {len(self.ground_truth)}")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"\nDetailed results saved to: {log_filename}")

        # Generate confusion matrix plot
        self.plot_confusion_matrix()


if __name__ == "__main__":
    # Path to your test dataset
    dataset_path = "data/test"

    try:
        # Initialize liveness detection
        print("Initializing Liveness Detection system...")
        liveness_detector = LivenessDetection()

        # Run the evaluation
        print("Starting evaluation process...")
        evaluator = DatasetEvaluator(liveness_detector, dataset_path, confidence_threshold=0.7)
        evaluator.evaluate()

    except Exception as e:
        print(f"Error during evaluation: {str(e)}")
