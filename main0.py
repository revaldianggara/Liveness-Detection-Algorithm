import cv2
import dlib
import numpy as np
import random
import time
from collections import deque

class LivenessDetection:
    def __init__(self, active_mode=True):
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
        self.active_mode = active_mode
        self.previous_landmarks = None
        self.results_history = deque(maxlen=15)
        self.scores_history = deque(maxlen=15)
        self.current_prompt = None
        self.prompt_time = time.time()

    def log_result(self, result):
        with open("liveness_detection_log.txt", "a") as f:
            f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - {result}\n")

    def generate_prompt(self):
        prompts = ["Nod your head", "Shake your head"]
        self.current_prompt = random.choice(prompts)
        self.prompt_time = time.time()

    def analyze_frame(self, frame):
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

            # Active liveness detection
            active_score = self.active_liveness(landmarks)

            # Combine scores for the final result
            combined_score = (
                0.3 * texture_score +
                0.1 * motion +
                0.1 * (1 - reflection) +
                0.2 * blink +
                0.1 * pose +
                0.2 * active_score
            )

            # Smooth the score using a moving average
            self.scores_history.append(combined_score)
            averaged_score = np.mean(self.scores_history)

            label = "Real" if averaged_score > 0.6 else "Fake"
            results.append((face, label, averaged_score))
        return results

    def passive_detection(self, frame, face):
        face_roi = frame[face.top():face.bottom(), face.left():face.right()]
        gray_face = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        laplacian = cv2.Laplacian(gray_face, cv2.CV_64F).var()

        return min(max(laplacian / 100.0, 0.0), 1.0)

    def detect_reflection(self, frame, face):
        face_roi = frame[face.top():face.bottom(), face.left():face.right()]
        hsv_face = cv2.cvtColor(face_roi, cv2.COLOR_BGR2HSV)
        brightness = cv2.mean(hsv_face[:, :, 2])[0]

        return min(brightness / 255.0, 1.0)

    def blink_detection(self, landmarks):
        left_eye = [landmarks.part(i) for i in range(36, 42)]
        right_eye = [landmarks.part(i) for i in range(42, 48)]

        left_ear = self.calculate_ear(left_eye)
        right_ear = self.calculate_ear(right_eye)

        return 1.0 if (left_ear < 0.25 or right_ear < 0.25) else 0.0

    def calculate_ear(self, eye):
        A = np.linalg.norm(np.array([eye[1].x, eye[1].y]) - np.array([eye[5].x, eye[5].y]))
        B = np.linalg.norm(np.array([eye[2].x, eye[2].y]) - np.array([eye[4].x, eye[4].y]))
        C = np.linalg.norm(np.array([eye[0].x, eye[0].y]) - np.array([eye[3].x, eye[3].y]))
        return (A + B) / (2.0 * C)

    def analyze_motion(self, landmarks):
        if self.previous_landmarks is None:
            self.previous_landmarks = landmarks
            return 1.0

        motion_score = 0.0
        for i in range(68):
            current = np.array([landmarks.part(i).x, landmarks.part(i).y])
            previous = np.array([self.previous_landmarks.part(i).x, self.previous_landmarks.part(i).y])
            delta = np.linalg.norm(current - previous)
            if delta > 3:
                motion_score += 1

        self.previous_landmarks = landmarks
        return min(motion_score / 68.0, 1.0)

    def estimate_pose(self, landmarks):
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

    def active_liveness(self, landmarks):
        if time.time() - self.prompt_time > 5:
            self.generate_prompt()

        # Detect head nod or shake
        nose = landmarks.part(33)
        chin = landmarks.part(8)

        if self.current_prompt == "Nod your head":
            # Check vertical movement
            if abs(nose.y - chin.y) > 10:
                return 1.0
        elif self.current_prompt == "Shake your head":
            # Check horizontal movement
            if abs(nose.x - chin.x) > 10:
                return 1.0

        return 0.0

    def smooth_results(self, label):
        self.results_history.append(label)
        return "Real" if self.results_history.count("Real") > len(self.results_history) // 2 else "Fake"

    def start_detection(self):
        cap = cv2.VideoCapture(1)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if self.current_prompt is None:
                self.generate_prompt()

            cv2.putText(frame, self.current_prompt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            results = self.analyze_frame(frame)

            if len(results) == 0:
                cv2.putText(frame, "No face detected", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            else:
                for face, label, score in results:
                    smoothed_label = self.smooth_results(label)

                    color = (0, 255, 0) if smoothed_label == "Real" else (0, 0, 255)
                    cv2.rectangle(frame, (face.left(), face.top()), (face.right(), face.bottom()), color, 2)

                    text = f"{smoothed_label} ({score:.2f})"
                    cv2.putText(frame, text, (face.left(), face.top() - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            cv2.imshow("Liveness Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    detector = LivenessDetection(active_mode=True)
    detector.start_detection()
