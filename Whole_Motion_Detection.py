import cv2
import numpy as np
import mediapipe as mp
import os
import csv

# Initialize Mediapipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

# Optical Flow (Lucas-Kanade) parameters
lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# List of video files to process
video_files = ['barbell_biceps_curl_1.mp4', 'deadlift_10.mp4']  # Update with your paths

for video_file in video_files:
    # Check if video exists
    if not os.path.exists(video_file):
        print(f"Video file {video_file} not found. Skipping...")
        continue

    # Open video
    cap = cv2.VideoCapture(video_file)
    if not cap.isOpened():
        print(f"Error opening {video_file}. Skipping...")
        continue

    # Get video properties for output
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Set up video writers for each output
    base_name = os.path.basename(video_file).split('.')[0]
    out_main = cv2.VideoWriter(f"output_{base_name}_main.mp4", cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))
    out_flow = cv2.VideoWriter(f"output_{base_name}_farneback.mp4", cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))
    out_gmm = cv2.VideoWriter(f"output_{base_name}_gmm.mp4", cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

    # Set up CSV file for numerical data
    csv_file = f"analysis_{base_name}.csv"
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Frame', 'LK_Dots', 'Farneback_Avg_Magnitude', 'GMM_Foreground_Pixels', 'Mediapipe_Elbow_Angle', 'Mediapipe_Knee_Angle'])

    # Read first frame
    ret, prev_frame = cap.read()
    if not ret:
        print(f"Error reading first frame of {video_file}. Skipping...")
        cap.release()
        continue

    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    hsv = np.zeros_like(prev_frame)
    hsv[..., 1] = 255

    # Detect initial points for Lucas-Kanade
    p0 = cv2.goodFeaturesToTrack(prev_gray, maxCorners=100, qualityLevel=0.3, minDistance=7)

    # Background Subtraction (GMM)
    bg_subtractor = cv2.createBackgroundSubtractorMOG2()

    print(f"Processing {video_file}...")
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_count += 1

        # 1. Optical Flow (Lucas-Kanade)
        lk_dots = 0
        if p0 is not None:
            p1, st, err = cv2.calcOpticalFlowPyrLK(prev_gray, frame_gray, p0, None, **lk_params)
            if p1 is not None:
                good_new = p1[st == 1]
                good_old = p0[st == 1]
                lk_dots = len(good_new)
                for i, (new, old) in enumerate(zip(good_new, good_old)):
                    a, b = new.ravel()
                    c, d = old.ravel()
                    frame = cv2.line(frame, (int(a), int(b)), (int(c), int(d)), (0, 255, 0), 2)
                    frame = cv2.circle(frame, (int(a), int(b)), 5, (0, 0, 255), -1)
                p0 = good_new.reshape(-1, 1, 2)

        # 2. Dense Optical Flow (Farneback)
        flow = cv2.calcOpticalFlowFarneback(prev_gray, frame_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        farneback_avg_mag = np.mean(mag)
        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        flow_vis = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        # 3. Background Subtraction (GMM)
        fg_mask = bg_subtractor.apply(frame)
        fg_result = cv2.bitwise_and(frame, frame, mask=fg_mask)
        gmm_foreground_pixels = np.sum(fg_mask > 0)

        # 4. Pose Estimation (Mediapipe)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)
        elbow_angle = None
        knee_angle = None
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            landmarks = results.pose_landmarks.landmark
            
            # Determine which angle to calculate based on video filename
            if 'biceps_curl' in video_file.lower():
                # Elbow angle for bicep curl (left shoulder, elbow, wrist)
                shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y]
                elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW].y]
                wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST].y]
                angle = np.arctan2(wrist[1] - elbow[1], wrist[0] - elbow[0]) - np.arctan2(shoulder[1] - elbow[1], shoulder[0] - elbow[0])
                elbow_angle = np.abs(angle * 180.0 / np.pi)
                cv2.putText(frame, f"Elbow Angle: {elbow_angle:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            elif 'deadlift' in video_file.lower():
                # Knee angle for deadlift (left hip, knee, ankle)
                hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP].y]
                knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE].y]
                ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE].x, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE].y]
                angle = np.arctan2(ankle[1] - knee[1], ankle[0] - knee[0]) - np.arctan2(hip[1] - knee[1], hip[0] - knee[0])
                knee_angle = np.abs(angle * 180.0 / np.pi)
                cv2.putText(frame, f"Knee Angle: {knee_angle:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        # Write numerical data to CSV
        with open(csv_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([frame_count, lk_dots, farneback_avg_mag, gmm_foreground_pixels, elbow_angle, knee_angle])

        # Write to output videos
        out_main.write(frame)  # Lucas-Kanade + Mediapipe
        out_flow.write(flow_vis)  # Farneback
        out_gmm.write(fg_result)  # GMM

        # Display results
        cv2.imshow('Frame', frame)
        cv2.imshow('Dense Optical Flow', flow_vis)
        cv2.imshow('Background Subtraction', fg_result)

        # Update previous frame
        prev_gray = frame_gray.copy()

        # Exit on 'q' press
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    # Release resources for this video
    cap.release()
    out_main.release()
    out_flow.release()
    out_gmm.release()
    print(f"Finished processing {video_file}. Outputs saved as:")
    print(f" - Main (Lucas-Kanade + Mediapipe): output_{base_name}_main.mp4")
    print(f" - Farneback: output_{base_name}_farneback.mp4")
    print(f" - GMM: output_{base_name}_gmm.mp4")
    print(f" - Numerical Analysis: {csv_file}")

# Clean up
cv2.destroyAllWindows()
pose.close()