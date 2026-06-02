import math
from collections import deque


class PostureVerifier:
    def __init__(self, confirmation_frames=3):
        self.confirmation_frames = confirmation_frames
        self.fall_counter = 0
        self.normal_counter = 0
        self.hip_y_history = deque(maxlen=10)

        # Parameters
        self.RATIO_THRESHOLD = 1.2
        self.ANGLE_MIN = 45
        self.ANGLE_MAX = 135
        self.VELOCITY_THRESHOLD = 0.02

        self.confidence = 0.0

    def _torso_angle(self, landmarks):
        """Calculate torso angle - works with both MediaPipe and YOLO"""
        if len(landmarks) > 20:  # MediaPipe
            s = landmarks[11]  # left shoulder
            h = landmarks[23]  # left hip
            return abs(math.degrees(math.atan2(h.y - s.y, h.x - s.x)))
        else:  # YOLO
            if len(landmarks) < 13:
                return 90.0
            shoulder = (landmarks[5] + landmarks[6]) / 2
            hip = (landmarks[11] + landmarks[12]) / 2
            dx = hip[0] - shoulder[0]
            dy = hip[1] - shoulder[1]
            return abs(math.degrees(math.atan2(dy, dx)))

    def _velocity(self, landmarks):
        if len(landmarks) > 20:  # MediaPipe
            hip_y = landmarks[23].y
        else:  # YOLO
            hip_y = (landmarks[11][1] + landmarks[12][1]) / 2

        self.hip_y_history.append(hip_y)
        if len(self.hip_y_history) < 3:
            return 0.0
        return self.hip_y_history[-1] - self.hip_y_history[-3]

    def _compute_confidence(self, box, landmarks):
        _, _, w, h = box
        if h == 0:
            return 0.0

        # Ratio score
        ratio = w / float(h)
        ratio_score = min(max((ratio - 0.8) / (2.0 - 0.8), 0.0), 1.0)

        # Angle score
        angle = self._torso_angle(landmarks)
        deviation = min(abs(angle - 90), 90)
        angle_score = deviation / 90.0

        # Velocity score
        velocity = self._velocity(landmarks)
        velocity_score = min(max(velocity / (self.VELOCITY_THRESHOLD * 3), 0.0), 1.0)

        confidence = (ratio_score * 0.4) + (angle_score * 0.4) + (velocity_score * 0.2)
        return round(min(confidence, 1.0), 3)

    def evaluate_posture(self, box, landmarks):
        if box is None or landmarks is None:
            self.fall_counter = 0
            self.confidence = 0.0
            return False, 0.0

        _, _, w, h = box
        if h == 0:
            return False, 0.0

        ratio_ok = (w / float(h)) > self.RATIO_THRESHOLD
        angle = self._torso_angle(landmarks)
        inclined = angle < self.ANGLE_MIN or angle > self.ANGLE_MAX
        dropping = self._velocity(landmarks) > self.VELOCITY_THRESHOLD

        suspicious = (ratio_ok and inclined) or (dropping and inclined)

        if suspicious:
            self.fall_counter += 1
            self.normal_counter = 0
        else:
            self.normal_counter += 1
            if self.normal_counter >= 3:
                self.fall_counter = 0

        is_fall = self.fall_counter >= self.confirmation_frames
        self.confidence = self._compute_confidence(box, landmarks)

        return is_fall, self.confidence


# Backward compatibility
_verifier = PostureVerifier()

def evaluate_posture(box, landmarks):
    is_fall, confidence = _verifier.evaluate_posture(box, landmarks)
    return is_fall, confidence