import os
import cv2

observationdir = '/home/tione/notebook/SpacialLogic-Demo/observation'
clipbasedir = '/home/tione/notebook/SpacialLogic-Demo/clips'

for sub1 in os.listdir(observationdir):  # e.g., 327
    sub1path = os.path.join(observationdir, sub1)
    for sub2 in os.listdir(sub1path):  # e.g., 648642
        sub2path = os.path.join(sub1path, sub2)
        for sub3 in os.listdir(sub2path):  # e.g., head_colors.mp4
            sub3path = os.path.join(sub2path, sub3)
            clipdir = os.path.join(clipbasedir, sub1, sub2)
            os.makedirs(clipdir, exist_ok=True)

            cap = cv2.VideoCapture(sub3path)
            frame_idx = 0
            saved_idx = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                if frame_idx % 10 == 0:
                    frame_path = os.path.join(clipdir, f'{saved_idx:05d}.jpg')
                    cv2.imwrite(frame_path, frame)
                    saved_idx += 1
                frame_idx += 1
            cap.release()