import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import mediapipe as mp
import time

def load_video(video_path):
    """
    Load a video file and return video capture object
    """
    cap = cv2.VideoCapture(video_path)
    
    # Check if video opened successfully
    if not cap.isOpened():
        raise ValueError("Error: Could not open video.")
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"Video loaded successfully:")
    print(f"- Resolution: {width}x{height}")
    print(f"- FPS: {fps}")
    print(f"- Total frames: {frame_count}")
    print(f"- Duration: {frame_count/fps:.2f} seconds")
    
    return cap

def main():
    video_path = "barbell_biceps_curl_1.mp4"  # Replace with your video path
    try:
        cap = load_video(video_path)
        if cap is not None:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                cv2.imshow('Video', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            cap.release()
            cv2.destroyAllWindows()
    except ValueError as e:
        print(e)

if __name__ == "__main__":
    main()