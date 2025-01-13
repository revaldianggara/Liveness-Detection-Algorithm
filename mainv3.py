import cv2
import dlib
import numpy as np
import time
from collections import deque
import random


class LivenessDetection:
    def __init__(self):
        """
        Initialize the liveness detection system.
        """
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
        self.active_mode = False  # Passive Mode by default
        self.scores_history = deque(maxlen=15)  # For averaging scores
        self.current_prompt = None  # The current action prompt

    def passive_detection(self, frame, face):
        """
        Perform passive texture analysis (e.g., Laplacian variance).

        Args:
            frame (numpy.ndarray): The input frame.
            face (dlib.rectangle): Detected face bounding box.

        Returns:
            float: Normalized texture confidence score.
        """
        face_roi = frame[face.top():face.bottom(), face.left():face.right()]
        gray_face = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        laplacian = cv2.Laplacian(gray_face, cv2.CV_64F).var()
        return min(max(laplacian / 100.0, 0.0), 1.0)

    def detect_reflection(self, frame, face):
        """
        Detect reflections in the face region.

        Args:
            frame (numpy.ndarray): The input frame.
            face (dlib.rectangle): Detected face bounding box.

        Returns:
            float: Normalized reflection score (lower is better).
        """
        face_roi = frame[face.top():face.bottom(), face.left():face.right()]
        hsv_face = cv2.cvtColor(face_roi, cv2.COLOR_BGR2HSV)
        brightness = cv2.mean(hsv_face[:, :, 2])[0]
        return min(brightness / 255.0, 1.0)

    def blink_detection(self, landmarks):
        """
        Detect blinks using Eye Aspect Ratio (EAR).

        Args:
            landmarks (dlib.full_object_detection): Facial landmarks.

        Returns:
            float: 1.0 if blink is detected, else 0.0.
        """
        left_eye = [landmarks.part(i) for i in range(36, 42)]
        right_eye = [landmarks.part(i) for i in range(42, 48)]

        left_ear = self.calculate_ear(left_eye)
        right_ear = self.calculate_ear(right_eye)

        return 1.0 if (left_ear < 0.25 or right_ear < 0.25) else 0.0

    def calculate_ear(self, eye):
        """
        Calculate Eye Aspect Ratio (EAR).

        Args:
            eye (list): List of eye landmark points.

        Returns:
            float: EAR value.
        """
        A = np.linalg.norm(np.array([eye[1].x, eye[1].y]) - np.array([eye[5].x, eye[5].y]))
        B = np.linalg.norm(np.array([eye[2].x, eye[2].y]) - np.array([eye[4].x, eye[4].y]))
        C = np.linalg.norm(np.array([eye[0].x, eye[0].y]) - np.array([eye[3].x, eye[3].y]))
        return (A + B) / (2.0 * C)

    def analyze_motion(self, landmarks):
        """
        Analyze motion between consecutive frames.

        Args:
            landmarks (dlib.full_object_detection): Facial landmarks.

        Returns:
            float: Normalized motion score.
        """
        # Simulated motion analysis, return a placeholder for demonstration
        return 0.5

    def estimate_pose(self, landmarks):
        """
        Estimate the head pose based on facial landmarks.

        Args:
            landmarks (dlib.full_object_detection): Facial landmarks.

        Returns:
            float: 1.0 if the pose is within the expected range, else 0.0.
        """
        # Simulated pose analysis, return a placeholder for demonstration
        return 1.0

    def analyze_frame(self, frame):
        """
        Analyze a single frame for liveness detection.

        Args:
            frame (numpy.ndarray): The input video frame.

        Returns:
            list: A list of tuples containing face bounding box, label, and confidence score.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.detector(gray)
        if len(faces) == 0:
            return []

        results = []
        for face in faces:
            landmarks = self.predictor(gray, face)

            # Perform detection steps
            texture_score = self.passive_detection(frame, face)
            reflection = self.detect_reflection(frame, face)
            blink = self.blink_detection(landmarks)
            motion = self.analyze_motion(landmarks)
            pose = self.estimate_pose(landmarks)

            # Combine scores for the final result
            combined_score = (
                0.5 * texture_score +
                0.1 * motion +
                0.1 * (1 - reflection) +
                0.2 * blink +
                0.1 * pose
            )

            # Smooth the score using a moving average
            self.scores_history.append(combined_score)
            averaged_score = np.mean(self.scores_history)

            # Determine label
            label = "Real" if averaged_score > 0.5 else "Fake"
            results.append((face, label, averaged_score))
        return results

    def start_detection(self):
        """
        Start the liveness detection process.
        """
        cap = cv2.VideoCapture(1)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            results = self.analyze_frame(frame)

            if len(results) == 0:
                cv2.putText(frame, "No Faces Detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            else:
                for face, label, score in results:
                    color = (0, 255, 0) if label == "Real" else (0, 0, 255)
                    cv2.rectangle(frame, (face.left(), face.top()), (face.right(), face.bottom()), color, 2)

                    text = f"{label} ({score:.2f})"
                    cv2.putText(frame, text, (face.left(), face.top() - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            cv2.imshow("Liveness Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    detector = LivenessDetection()
    detector.start_detection()
