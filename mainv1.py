import cv2
import dlib
import numpy as np
import time

class LivenessDetection:
    def __init__(self, active_mode=True):
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
        self.active_mode = active_mode
        self.previous_landmarks = None  # motion-based validation
        self.results_history = []  # smooth results over time

    def log_result(self, result):
        with open("liveness_detection_log.txt", "a") as f:
            f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - {result}\n")

    def analyze_frame(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.detector(gray)
        if len(faces) == 0:
            return []  # No face detected

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
                0.3 * texture_score +
                0.3 * (1 - reflection) +  # Invert reflection for contribution
                0.2 * blink +
                0.2 * pose
            )
            label = "Real" if combined_score > 0.5 else "Fake"

            results.append((face, label, combined_score))
        return results

    def passive_detection(self, frame, face):
        # Analyze Laplacian variance for texture analysis
        face_roi = frame[face.top():face.bottom(), face.left():face.right()]
        gray_face = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        laplacian = cv2.Laplacian(gray_face, cv2.CV_64F).var()

        # Normalize to confidence score
        return min(max(laplacian / 100.0, 0.0), 1.0)

    def detect_reflection(self, frame, face):
        face_roi = frame[face.top():face.bottom(), face.left():face.right()]
        hsv_face = cv2.cvtColor(face_roi, cv2.COLOR_BGR2HSV)
        brightness = cv2.mean(hsv_face[:, :, 2])[0]

        # Return normalized brightness score (lower score for high brightness)
        return min(brightness / 255.0, 1.0)

    def blink_detection(self, landmarks):
        # Check EAR for blink detection
        left_eye = [landmarks.part(i) for i in range(36, 42)]
        right_eye = [landmarks.part(i) for i in range(42, 48)]

        left_ear = self.calculate_ear(left_eye)
        right_ear = self.calculate_ear(right_eye)

        # EAR threshold for blinking
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
            if delta > 2:  # Threshold for motion
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

    def smooth_results(self, label):
        self.results_history.append(label) # Maintain a history of results for smoothing
        if len(self.results_history) > 5:
            self.results_history.pop(0)

        # Use majority voting
        return "Real" if self.results_history.count("Real") > 2 else "Fake"

    def start_detection(self):
        cap = cv2.VideoCapture(1)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            results = self.analyze_frame(frame)

            if len(results) == 0:
                cv2.putText(frame, "No face detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
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
    detector = LivenessDetection(active_mode=False)
    detector.start_detection()
