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

# Create output folder
output_folder = "Boxing_Foul_Analysis_Outputs"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# List of video files to process (boxing foul vs non-foul clips)
video_files = [
    'Boxing_Vids\foul_1.mp4',  # Foul: punch below the belt
    'Boxing_Vids\non_foul_5.mp4'  # Non-foul: punch above the belt
]  # Update with your boxing clip paths

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

    # Set up video writers for each output in the output folder
    base_name = os.path.basename(video_file).split('.')[0]
    out_main_path = os.path.join(output_folder, f"output_{base_name}_main.mp4")
    out_flow_path = os.path.join(output_folder, f"output_{base_name}_farneback.mp4")
    out_gmm_path = os.path.join(output_folder, f"output_{base_name}_gmm.mp4")
    
    out_main = cv2.VideoWriter(out_main_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))
    out_flow = cv2.VideoWriter(out_flow_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))
    out_gmm = cv2.VideoWriter(out_gmm_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

    # Set up CSV file for numerical data in the output folder
    csv_file = os.path.join(output_folder, f"analysis_{base_name}.csv")
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Frame', 'LK_Dots', 'Farneback_Avg_Magnitude', 'GMM_Foreground_Pixels', 
                         'Left_Wrist_Y', 'Right_Wrist_Y', 'Hip_Y_Avg', 'Is_Foul'])

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

        # 4. Pose Estimation (Mediapipe) - Foul detection
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)
        left_wrist_y = None
        right_wrist_y = None
        hip_y_avg = None
        is_foul = "No"

        if results.pose_landmarks:
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            landmarks = results.pose_landmarks.landmark
            
            # Wrist positions (for punch location)
            left_wrist_y = landmarks[mp_pose.PoseLandmark.LEFT_WRIST].y
            right_wrist_y = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].y
            
            # Average hip height (belt line approximation)
            left_hip_y = landmarks[mp_pose.PoseLandmark.LEFT_HIP].y
            right_hip_y = landmarks[mp_pose.PoseLandmark.RIGHT_HIP].y
            hip_y_avg = (left_hip_y + right_hip_y) / 2

            # Foul detection: Check if either wrist is below the average hip height
            if (left_wrist_y > hip_y_avg or right_wrist_y > hip_y_avg) and \
               (abs(landmarks[mp_pose.PoseLandmark.LEFT_WRIST].x - landmarks[mp_pose.PoseLandmark.RIGHT_HIP].x) < 0.2 or \
                abs(landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].x - landmarks[mp_pose.PoseLandmark.LEFT_HIP].x) < 0.2):
                is_foul = "Yes"  # Punch below belt and near opponent's body

            # Display foul status on video
            cv2.putText(frame, f"Foul: {is_foul}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255) if is_foul == "Yes" else (255, 0, 0), 2)

        # Write numerical data to CSV
        with open(csv_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([frame_count, lk_dots, farneback_avg_mag, gmm_foreground_pixels, 
                             left_wrist_y, right_wrist_y, hip_y_avg, is_foul])

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
    print(f"Finished processing {video_file}. Outputs saved in {output_folder}:")
    print(f" - Main (Lucas-Kanade + Mediapipe): {os.path.basename(out_main_path)}")
    print(f" - Farneback: {os.path.basename(out_flow_path)}")
    print(f" - GMM: {os.path.basename(out_gmm_path)}")
    print(f" - Numerical Analysis: {os.path.basename(csv_file)}")

# Clean up
cv2.destroyAllWindows()
pose.close()