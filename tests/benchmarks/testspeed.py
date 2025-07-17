import time
import cv2
import sys
import os
import celux
import torch

VIDEO_PATH = r"D:\dev\Projects\Repos\CeLux\tests\data\default\BigBuckBunny.mp4"

def benchmark_opencv():
    cap = cv2.VideoCapture(VIDEO_PATH)
    start = time.time()
    frame_count = 0
    while cap.isOpened():
        ret, _ = cap.read()
        if not ret:
            break
        frame_count += 1
    cap.release()
    return frame_count / (time.time() - start)

def benchmark_celux():
    filters = celux.Curves(red="1.0")
    reader = celux.VideoReader(VIDEO_PATH)
    start = time.time()
    frame_count = 0
    for _ in reader:
        _.to(device=torch.device("cuda" if torch.cuda.is_available() else "cpu")).to(torch.float16).mul(1.25).to(torch.uint8).contiguous()
        frame_count += 1
    return frame_count / (time.time() - start)

if __name__ == "__main__":
    print(f"OpenCV FPS: {benchmark_opencv():.2f}")
    print(f"CeLux FPS: {benchmark_celux():.2f}")
