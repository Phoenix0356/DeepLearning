import cv2
import time
from detector import PoseDetector

if __name__ == "__main__":
    cap = cv2.VideoCapture("D:/DeepLearning/videos/kunkun.mp4")
    prev_time = 0
    while True:
        flag, frame = cap.read()
        detector = PoseDetector()
        frame = detector.detect_pose(frame)

        current_time = time.time()
        fps = 1 / (current_time - prev_time)
        prev_time = current_time

        cv2.putText(frame, str(int(fps)), (70, 50), cv2.FONT_HERSHEY_PLAIN, 3,
                    (255, 0, 0), 3)
        cv2.namedWindow('image', cv2.WINDOW_NORMAL)
        cv2.imshow("image", frame)
        cv2.resizeWindow('image', 800, 600)
        cv2.waitKey(1)