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
        self.previous_landmarks = None
        self.results_history = deque(maxlen=15) 
        self.scores_history = deque(maxlen=15) 
        self.current_prompt = None  # The current action prompt
        self.prompts = deque(["Blink your eyes", "Nod your head", "Shake your head"])  # Prompt rotation
        self.prompt_timer = time.time()  # manage prompts
        self.last_prompt_time = time.time() # manage next prompts

    def generate_prompt(self):
        """
        Generate the next prompt for the user in rotation order.
        """
        # self.prompts.rotate(-1)  # Move to the next prompt in the deque
        # self.current_prompt = self.prompts[0]  # Update the current prompt
        cooldown = 1  # in seconds
        current_time = time.time()
        if current_time - self.last_prompt_time > cooldown:  # Check if cooldown has passed
            self.prompts.rotate(-1)  # Move to the next prompt in the deque
            self.current_prompt = self.prompts[0]  # Update the current prompt
            self.last_prompt_time = current_time  # Update the last prompt time
            # print(f"Generated new prompt: {self.current_prompt}") 
            # print(f"Prompt queue: {list(self.prompts)}") 

    def validate_prompt(self, landmarks):
        """
        Validate if the user performs the prompted action.

        Args:
            landmarks (dlib.full_object_detection): Current facial landmarks.

        Returns:
            bool: True if the user performed the correct action, False otherwise.
        """
        if landmarks is None:
            return False  # Return False if landmarks are invalid or missing

        if self.current_prompt == "Blink your eyes":
            return self.blink_detection(landmarks) == 1.0
        elif self.current_prompt == "Nod your head":
            return self.analyze_vertical_motion(landmarks) > 0.3
        elif self.current_prompt == "Shake your head":
            return self.analyze_horizontal_motion(landmarks) > 0.3
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

    def analyze_vertical_motion(self, landmarks):
        """
        Analyze vertical motion (for nodding) between consecutive frames.
        """
        if self.previous_landmarks is None:
            self.previous_landmarks = landmarks
            return 0.0

        vertical_motion = 0.0
        for i in range(68):
            current_y = landmarks.part(i).y
            previous_y = self.previous_landmarks.part(i).y
            vertical_motion += abs(current_y - previous_y)

        self.previous_landmarks = landmarks
        return vertical_motion / 68.0

    def analyze_horizontal_motion(self, landmarks):
        """
        Analyze horizontal motion (for shaking) between consecutive frames.
        """
        if self.previous_landmarks is None:
            self.previous_landmarks = landmarks
            return 0.0

        horizontal_motion = 0.0
        for i in range(68):
            current_x = landmarks.part(i).x
            previous_x = self.previous_landmarks.part(i).x
            horizontal_motion += abs(current_x - previous_x)

        self.previous_landmarks = landmarks
        return horizontal_motion / 68.0

    def analyze_frame(self, frame):
        """
        Analyze a single frame for static aspects of liveness detection.

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
            texture_score = self.passive_detection(frame, face)
            reflection = self.detect_reflection(frame, face)
            combined_score = 0.7 * texture_score + 0.3 * (1 - reflection)
            self.scores_history.append(combined_score)
            averaged_score = np.mean(self.scores_history)
            label = "Real" if averaged_score > 0.5 else "Fake"
            results.append((face, label, averaged_score))
        return results

    def passive_detection(self, frame, face):
        """
        Perform passive texture analysis (e.g., Laplacian variance).
        """
        face_roi = frame[face.top():face.bottom(), face.left():face.right()]
        gray_face = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        laplacian = cv2.Laplacian(gray_face, cv2.CV_64F).var()
        return min(max(laplacian / 100.0, 0.0), 1.0)

    def detect_reflection(self, frame, face):
        """
        Detect reflections in the face region.
        """
        face_roi = frame[face.top():face.bottom(), face.left():face.right()]
        hsv_face = cv2.cvtColor(face_roi, cv2.COLOR_BGR2HSV)
        brightness = cv2.mean(hsv_face[:, :, 2])[0]
        return min(brightness / 255.0, 1.0)

    def start_detection(self):
        """
        Start the liveness detection process.
        """
        cap = cv2.VideoCapture(1)
        self.generate_prompt()  # generate prompt

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            results = self.analyze_frame(frame)
            faces = self.detector(gray)

            # print(f"Current Prompt: {self.current_prompt}") 

            for face, label, score in results:
                color = (0, 255, 0) if label == "Real" else (0, 0, 255)
                cv2.rectangle(frame, (face.left(), face.top()), (face.right(), face.bottom()), color, 2)
                text = f"{label} ({score:.2f})" # score and label
                cv2.putText(frame, text, (face.left(), face.top() - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            if self.active_mode:
                for face in faces:
                    landmarks = self.predictor(gray, face)
                    if self.current_prompt and self.validate_prompt(landmarks):
                        cv2.putText(frame, "Correct Action!", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        self.generate_prompt()  # Generate the next prompt
                    else:
                        cv2.putText(frame, "Perform Action!", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # Display the current prompt if in Active Mode
            if self.active_mode and self.current_prompt:
                cv2.putText(frame, f"Action: {self.current_prompt}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

            # Show current mode
            mode_text = "Active Mode" if self.active_mode else "Passive Mode"
            cv2.putText(frame, f"Mode: {mode_text}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

            cv2.imshow("Liveness Detection", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break  # Quit the program
            elif key == ord('a'):
                self.active_mode = True  # Switch to Active Mode
                self.generate_prompt()  # Generate a new prompt
            elif key == ord('p'):
                self.active_mode = False  # Switch to Passive Mode

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    detector = LivenessDetection()
    detector.start_detection()
