import cv2
import dlib
import numpy as np
import time
from collections import deque
import argparse
import random


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
        self.results_history = deque(maxlen=15)  # History for smoothing results
        self.scores_history = deque(maxlen=15)  # For averaging scores
        self.motion_scores_history = deque(maxlen=15)  # For tracking motion-based validation
        self.current_prompt = None  # The current action prompt
        self.prompt_timer = time.time()  # Timer to manage prompts

    def generate_prompt(self):
        """
        Generate a random prompt for the user.
        """
        prompts = ["Blink your eyes", "Nod your head", "Shake your head"]
        self.current_prompt = random.choice(prompts)

    def validate_prompt(self, landmarks):
        """
        Validate if the user performs the prompted action.

        Args:
            landmarks (dlib.full_object_detection): Current facial landmarks.

        Returns:
            bool: True if the user performed the correct action, False otherwise.
        """
        if self.current_prompt == "Blink your eyes":
            return self.blink_detection(landmarks) == 1.0
        elif self.current_prompt == "Nod your head":
            motion_score = self.analyze_motion(landmarks)
            return motion_score > 0.3  # Detect significant head motion for nodding
        elif self.current_prompt == "Shake your head":
            pose_score = self.estimate_pose(landmarks)
            return pose_score < 0.8  # Detect head pose change for shaking
        return False

    def blink_detection(self, landmarks):
        """
        Detect blinks using Eye Aspect Ratio (EAR).
        """
        left_eye = [landmarks.part(i) for i in range(36, 42)]
        right_eye = [landmarks.part(i) for i in range(42, 48)]
        left_ear = self.calculate_ear(left_eye)
        right_ear = self.calculate_ear(right_eye)
        return 1.0 if (left_ear < 0.25 or right_ear < 0.25) else 0.0

    def calculate_ear(self, eye):
        """
        Calculate Eye Aspect Ratio (EAR).
        """
        A = np.linalg.norm(np.array([eye[1].x, eye[1].y]) - np.array([eye[5].x, eye[5].y]))
        B = np.linalg.norm(np.array([eye[2].x, eye[2].y]) - np.array([eye[4].x, eye[4].y]))
        C = np.linalg.norm(np.array([eye[0].x, eye[0].y]) - np.array([eye[3].x, eye[3].y]))
        return (A + B) / (2.0 * C)

    def analyze_motion(self, landmarks):
        """
        Analyze motion between consecutive frames.
        """
        if self.previous_landmarks is None:
            self.previous_landmarks = landmarks
            return 0.0

        motion_score = 0.0
        for i in range(68):
            current = np.array([landmarks.part(i).x, landmarks.part(i).y])
            previous = np.array([self.previous_landmarks.part(i).x, self.previous_landmarks.part(i).y])
            delta = np.linalg.norm(current - previous)
            if delta > 3:
                motion_score += 1

        self.previous_landmarks = landmarks
        return motion_score / 68.0

    def estimate_pose(self, landmarks):
        """
        Estimate head pose based on facial landmarks.
        """
        nose = landmarks.part(33)
        chin = landmarks.part(8)
        left_eye = landmarks.part(36)
        right_eye = landmarks.part(45)
        vertical_dist = abs(nose.y - chin.y)
        horizontal_dist = abs(left_eye.x - right_eye.x)
        ratio = vertical_dist / max(horizontal_dist, 1)
        return 1.0 if 0.8 < ratio < 1.2 else 0.0

    def start_detection(self):
        """
        Start the liveness detection process.
        """
        cap = cv2.VideoCapture(1)
        self.generate_prompt()  # Generate the first prompt

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.detector(gray)

            if len(faces) == 0:
                cv2.putText(frame, "Unidentified", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            else:
                for face in faces:
                    landmarks = self.predictor(gray, face)

                    if self.active_mode and self.current_prompt:
                        if self.validate_prompt(landmarks):
                            cv2.putText(frame, "Correct Action!", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                            self.generate_prompt()  # Generate a new prompt
                        else:
                            cv2.putText(frame, "Perform Action!", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                    # Draw rectangle around the face
                    cv2.rectangle(frame, (face.left(), face.top()), (face.right(), face.bottom()), (0, 255, 0), 2)

            # Display the current prompt
            if self.current_prompt:
                cv2.putText(frame, f"Action: {self.current_prompt}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

            cv2.imshow("Liveness Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Liveness Detection System")
    parser.add_argument("--active_mode", type=bool, default=False, help="Enable Active Mode for motion-based validation")
    args = parser.parse_args()

    detector = LivenessDetection(active_mode=args.active_mode)
    detector.start_detection()
