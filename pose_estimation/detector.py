import time
import cv2
import mediapipe as mp


class PoseDetector():
    def __init__(self,static_image_mode: bool = False,model_complexity: int = 1,
                 smooth_landmarks: bool = True,enable_segmentation: bool = False,
                smooth_segmentation: bool = True,min_detection_confidence: float = 0.5,
                min_tracking_confidence: float = 0.5):
        self.mp_draw = mp.solutions.drawing_utils
        self.mp_pose = mp.solutions.pose
        self.model = self.mp_pose.Pose(static_image_mode,model_complexity,smooth_landmarks,
                                       enable_segmentation,smooth_segmentation,
                                       min_detection_confidence,min_tracking_confidence)

    def detect_pose(self,frame,is_draw=True):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.model.process(frame_rgb)
        print(result.pose_landmarks)
        if result.pose_landmarks:
            if is_draw:
                self.mp_draw.draw_landmarks(frame, result.pose_landmarks,
                                       self.mp_pose.POSE_CONNECTIONS)
        return frame






