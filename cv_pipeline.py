import torch
import torch.nn.functional as F
import cv2
import numpy as np

class VisionGoalPipeline:
    def __init__(self, model_path='model/yolov8n-football.pt'):
        self.device = torch.device('cpu')
        self.model = torch.hub.load('ultralytics/yolov8', 'custom', path=model_path)
        self.model.to(self.device).eval()
        self.orb = cv2.ORB_create(nfeatures=500)
        self.prev_gray = None

    def preprocess(self, frame):
        # 1. Gaussian blur
        frame_tensor = torch.from_numpy(frame).float().permute(2,0,1).unsqueeze(0) / 255.0
        blurred = F.avg_pool2d(frame_tensor, kernel_size=3, stride=1, padding=1)
        blurred = blurred.squeeze(0).permute(1,2,0).numpy() * 255
        blurred = blurred.astype(np.uint8)

        # 2. HSV color filtering (green field mask)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_RGB2HSV)
        lower_green = np.array([35, 50, 50])
        upper_green = np.array([85, 255, 255])
        mask = cv2.inRange(hsv, lower_green, upper_green)
        field_only = cv2.bitwise_and(blurred, blurred, mask=mask)
        return field_only, blurred

    def detect_objects(self, frame):
        results = self.model(frame, imgsz=320, conf=0.6)
        detections = []
        for det in results.xyxy[0]:
            x1, y1, x2, y2, conf, cls = det.tolist()
            label = self.model.names[int(cls)]
            detections.append({
                'box': [int(x1), int(y1), int(x2), int(y2)],
                'label': label,
                'conf': conf
            })
        return detections

    def track_ball(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        if self.prev_gray is None:
            self.prev_gray = gray
            return None

        # ORB features on ball region (assume detected)
        kp1, des1 = self.orb.detectAndCompute(self.prev_gray, None)
        kp2, des2 = self.orb.detectAndCompute(gray, None)

        if des1 is None or des2 is None:
            self.prev_gray = gray
            return None

        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)

        if len(matches) > 10:
            src_pts = np.float32([kp1[m.queryIdx].pt for m in matches[:10]]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches[:10]]).reshape(-1, 1, 2)
            M, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC)
            # Estimate ball center movement
            ball_center_prev = np.array([[150, 100]], dtype='float32').reshape(-1, 1, 2)
            ball_center_now = cv2.perspectiveTransform(ball_center_prev, M)
            motion = ball_center_now - ball_center_prev
            self.prev_gray = gray
            return motion.flatten()
        return None
