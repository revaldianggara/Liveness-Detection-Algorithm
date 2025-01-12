import cv2
import dlib
import numpy as np
import time
from collections import deque

class LivenessDetection:
    def __init__(self, active_mode=True):
        """
        Initialize the liveness detection system.

        Args:
            active_mode (bool): Enables motion-based validation if True.
        """
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
        self.active_mode = active_mode
        self.previous_landmarks = None
        self.results_history = deque(maxlen=15)  # Increase history length
        self.scores_history = deque(maxlen=15)  # For averaging scores

    def log_result(self, result):
        """
        Log the detection result with a timestamp.

        Args:
            result (str): The result to log ("Real" or "Fake").
        """
        with open("liveness_detection_log.txt", "a") as f:
            f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - {result}\n")

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

            label = "Real" if averaged_score > 0.5 else "Fake"
            results.append((face, label, averaged_score))
        return results

    def passive_detection(self, frame, face):
        """
        Perform passive texture analysis using Laplacian variance.

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
            landmarks (dlib.full_object_detection): Current facial landmarks.

        Returns:
            float: Normalized motion score.
        """
        if self.previous_landmarks is None:
            self.previous_landmarks = landmarks
            return 1.0

        motion_score = 0.0
        for i in range(68):
            current = np.array([landmarks.part(i).x, landmarks.part(i).y])
            previous = np.array([self.previous_landmarks.part(i).x, self.previous_landmarks.part(i).y])
            delta = np.linalg.norm(current - previous)
            if delta > 3:  # Increase threshold to reduce sensitivity
                motion_score += 1

        self.previous_landmarks = landmarks
        return min(motion_score / 68.0, 1.0)

    def estimate_pose(self, landmarks):
        """
        Estimate the head pose based on facial landmarks.

        Args:
            landmarks (dlib.full_object_detection): Facial landmarks.

        Returns:
            float: 1.0 if the pose is within the expected range, else 0.0.
        """
        nose = landmarks.part(33)
        chin = landmarks.part(8)
        left_eye = landmarks.part(36)
        right_eye = landmarks.part(45)
        left_mouth = landmarks.part(48)
        right_mouth = landmarks.part(54)

        vertical_dist = abs(nose.y - chin.y)
        horizontal_dist = abs(left_eye.x - right_eye.x)

        ratio = vertical_dist / max(horizontal_dist, 1)
        return 1.0 if 0.8 < ratio < 1.2 else 0.0

    def smooth_results(self, label):
        """
        Smooth detection results using majority voting.

        Args:
            label (str): Current frame's result ("Real" or "Fake").

        Returns:
            str: Smoothed result ("Real" or "Fake").
        """
        self.results_history.append(label)
        return "Real" if self.results_history.count("Real") > len(self.results_history) // 2 else "Fake"

    def start_detection(self):
        """
        Start the liveness detection process using exaternal webcam.
        """
        cap = cv2.VideoCapture(1)
        last_detection_time = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            current_time = time.time()
            if current_time - last_detection_time >= 1:
                results = self.analyze_frame(frame)
                last_detection_time = current_time

            results = self.analyze_frame(frame)

            if len(results) == 0:
                cv2.putText(frame, "Unidentified", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            else:
                for face, label, score in results:
                    smoothed_label = self.smooth_results(label)

                    self.log_result(f"{smoothed_label} (Score: {score:.2f})")

                    color = (0, 255, 0) if smoothed_label == "Real" else (0, 0, 255)
                    cv2.rectangle(frame, (face.left(), face.top()), (face.right(), face.bottom()), color, 2)

                    # text = f"{smoothed_label} ({score:.2f})"
                    text = f"{smoothed_label}"
                    cv2.putText(frame, text, (face.left(), face.top() - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            cv2.imshow("Liveness Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    detector = LivenessDetection(active_mode=False)
    detector.start_detection()