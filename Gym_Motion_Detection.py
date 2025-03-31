import cv2
import numpy as np
import mediapipe as mp
import os
import csv
import time
from collections import deque

# Initialize Mediapipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

# Optical Flow parameters
lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
hs_params = dict(iterations=10, lmbda=0.01)

# Activity recognition parameters
WINDOW_SIZE = 10  # Number of frames for pattern analysis
ANGLE_THRESHOLD_CURL = 30  # Minimum elbow angle change for curl
ANGLE_THRESHOLD_DEADLIFT = 45  # Minimum knee angle change for deadlift

# Create output folder
output_folder = "Gym_Motion_Analysis_Outputs"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Video files
video_files = [
    'C:\\Users\\Ryan\\OneDrive\\Desktop\\Motion\\Motion_Detection\\Gym_Vids\\barbell_biceps_curl_1.mp4',
    'C:\\Users\\Ryan\\OneDrive\\Desktop\\Motion\\Motion_Detection\\Gym_Vids\\deadlift_10.mp4'
]

def calculate_angle(p1, p2, p3):
    """Calculate angle between three points"""
    a = np.array(p1)
    b = np.array(p2)
    c = np.array(p3)
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)
    return np.degrees(angle)

def horn_schunck(prev_gray, frame_gray, iterations=10, lmbda=0.01):
    u = np.zeros_like(prev_gray, dtype=np.float32)
    v = np.zeros_like(prev_gray, dtype=np.float32)
    Ix = cv2.Sobel(prev_gray, cv2.CV_32F, 1, 0, ksize=3)
    Iy = cv2.Sobel(prev_gray, cv2.CV_32F, 0, 1, ksize=3)
    It = frame_gray.astype(np.float32) - prev_gray.astype(np.float32)
    
    for _ in range(iterations):
        u_avg = cv2.blur(u, (3, 3))
        v_avg = cv2.blur(v, (3, 3))
        num = Ix * u_avg + Iy * v_avg + It
        denom = lmbda + Ix**2 + Iy**2
        u = u_avg - Ix * (num / denom)
        v = v_avg - Iy * (num / denom)
    
    return np.stack([u, v], axis=2)

def classify_activity(curl_angles, deadlift_angles, video_name):
    """Classify activity based on angle patterns"""
    if len(curl_angles) < WINDOW_SIZE and len(deadlift_angles) < WINDOW_SIZE:
        return "Unknown"
    
    if 'biceps_curl' in video_name.lower():
        if curl_angles:
            angle_range = max(curl_angles) - min(curl_angles)
            if angle_range > ANGLE_THRESHOLD_CURL:
                return "Dumbbell Curl"
    elif 'deadlift' in video_name.lower():
        if deadlift_angles:
            angle_range = max(deadlift_angles) - min(deadlift_angles)
            if angle_range > ANGLE_THRESHOLD_DEADLIFT:
                return "Deadlift"
    return "No Activity Detected"

def main():
    for video_file in video_files:
        if not os.path.exists(video_file):
            print(f"Video file {video_file} not found. Skipping...")
            continue

        cap = cv2.VideoCapture(video_file)
        if not cap.isOpened():
            print(f"Error opening {video_file}. Skipping...")
            continue

        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))

        base_name = os.path.basename(video_file).split('.')[0]
        # Store filenames for each VideoWriter
        out_lk_main_path = os.path.join(output_folder, f"output_{base_name}_lk_main.mp4")
        out_hs_main_path = os.path.join(output_folder, f"output_{base_name}_hs_main.mp4")
        out_flow_path = os.path.join(output_folder, f"output_{base_name}_farneback.mp4")
        out_gmm_path = os.path.join(output_folder, f"output_{base_name}_gmm.mp4")
        out_flow_comp_path = os.path.join(output_folder, f"output_{base_name}_flow_comparison.mp4")
        out_main_comp_path = os.path.join(output_folder, f"output_{base_name}_main_comparison.mp4")
        out_bg_comp_path = os.path.join(output_folder, f"output_{base_name}_bg_comparison.mp4")

        out_lk_main = cv2.VideoWriter(out_lk_main_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))
        out_hs_main = cv2.VideoWriter(out_hs_main_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))
        out_flow = cv2.VideoWriter(out_flow_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))
        out_gmm = cv2.VideoWriter(out_gmm_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))
        out_flow_comp = cv2.VideoWriter(out_flow_comp_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width * 2, frame_height))
        out_main_comp = cv2.VideoWriter(out_main_comp_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width * 2, frame_height))
        out_bg_comp = cv2.VideoWriter(out_bg_comp_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width * 2, frame_height))

        csv_file = os.path.join(output_folder, f"analysis_{base_name}.csv")
        with open(csv_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Frame', 'Activity', 'LK_Dots', 'HS_Avg_Magnitude', 'Farneback_Avg_Magnitude', 
                             'GMM_Foreground_Pixels', 'FD_Foreground_Pixels', 'Elbow_Angle', 'Knee_Angle', 
                             'GMM_Time', 'FD_Time'])

        ret, prev_frame = cap.read()
        if not ret:
            print(f"Error reading first frame of {video_file}. Skipping...")
            cap.release()
            continue

        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        hsv = np.zeros_like(prev_frame)
        hsv[..., 1] = 255
        p0 = cv2.goodFeaturesToTrack(prev_gray, maxCorners=100, qualityLevel=0.3, minDistance=7)
        bg_subtractor = cv2.createBackgroundSubtractorMOG2()
        prev_frame_fd = prev_gray.copy()

        # Activity recognition buffers
        curl_angles = deque(maxlen=WINDOW_SIZE)
        deadlift_angles = deque(maxlen=WINDOW_SIZE)

        print(f"Processing {video_file}...")
        frame_count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame_count += 1

            # 1. Lucas-Kanade Optical Flow
            lk_start = time.time()
            lk_dots = 0
            lk_frame = frame.copy()
            if p0 is not None:
                p1, st, err = cv2.calcOpticalFlowPyrLK(prev_gray, frame_gray, p0, None, **lk_params)
                if p1 is not None:
                    good_new = p1[st == 1]
                    good_old = p0[st == 1]
                    lk_dots = len(good_new)
                    for new, old in zip(good_new, good_old):
                        a, b = new.ravel()
                        c, d = old.ravel()
                        lk_frame = cv2.line(lk_frame, (int(a), int(b)), (int(c), int(d)), (0, 255, 0), 2)
                        lk_frame = cv2.circle(lk_frame, (int(a), int(b)), 5, (0, 0, 255), -1)
                    p0 = good_new.reshape(-1, 1, 2)
            lk_time = time.time() - lk_start

            # 2. Horn-Schunck Optical Flow
            hs_start = time.time()
            hs_frame = frame.copy()
            hs_flow = horn_schunck(prev_gray, frame_gray, **hs_params)
            hs_mag, hs_ang = cv2.cartToPolar(hs_flow[..., 0], hs_flow[..., 1])
            hs_avg_mag = np.mean(hs_mag)
            hs_hsv = np.zeros_like(frame)
            hs_hsv[..., 1] = 255
            hs_hsv[..., 0] = hs_ang * 180 / np.pi / 2
            hs_hsv[..., 2] = cv2.normalize(hs_mag, None, 0, 255, cv2.NORM_MINMAX)
            hs_vis = cv2.cvtColor(hs_hsv, cv2.COLOR_HSV2BGR)
            hs_time = time.time() - hs_start

            # Flow Comparison
            flow_comp = np.hstack((lk_frame, hs_vis))
            cv2.putText(flow_comp, "Lucas-Kanade", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(flow_comp, "Horn-Schunck", (frame_width + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            # 3. Dense Optical Flow (Farneback)
            fb_start = time.time()
            fb_flow = cv2.calcOpticalFlowFarneback(prev_gray, frame_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            fb_mag, fb_ang = cv2.cartToPolar(fb_flow[..., 0], fb_flow[..., 1])
            farneback_avg_mag = np.mean(fb_mag)
            hsv[..., 0] = fb_ang * 180 / np.pi / 2
            hsv[..., 2] = cv2.normalize(fb_mag, None, 0, 255, cv2.NORM_MINMAX)
            flow_vis = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            fb_time = time.time() - fb_start

            # 4. Background Subtraction (GMM)
            gmm_start = time.time()
            fg_mask_gmm = bg_subtractor.apply(frame)
            fg_result_gmm = cv2.bitwise_and(frame, frame, mask=fg_mask_gmm)
            gmm_foreground_pixels = np.sum(fg_mask_gmm > 0)
            gmm_time = time.time() - gmm_start

            # 5. Frame Differencing
            fd_start = time.time()
            fd_diff = cv2.absdiff(prev_frame_fd, frame_gray)
            _, fg_mask_fd = cv2.threshold(fd_diff, 30, 255, cv2.THRESH_BINARY)
            fg_result_fd = cv2.bitwise_and(frame, frame, mask=fg_mask_fd)
            fd_foreground_pixels = np.sum(fg_mask_fd > 0)
            fd_time = time.time() - fd_start

            # Background Comparison
            bg_comp = np.hstack((fg_result_gmm, fg_result_fd))
            cv2.putText(bg_comp, "GMM", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(bg_comp, "Frame Differencing", (frame_width + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            # 6. Pose Estimation and Activity Recognition
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(frame_rgb)
            elbow_angle = None
            knee_angle = None
            activity = "Unknown"

            if results.pose_landmarks:
                mp_drawing.draw_landmarks(lk_frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                mp_drawing.draw_landmarks(hs_frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                landmarks = results.pose_landmarks.landmark

                # Dumbbell Curl (elbow angle)
                shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].x, 
                           landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y]
                elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW].x, 
                        landmarks[mp_pose.PoseLandmark.LEFT_ELBOW].y]
                wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST].x, 
                        landmarks[mp_pose.PoseLandmark.LEFT_WRIST].y]
                elbow_angle = calculate_angle(shoulder, elbow, wrist)
                curl_angles.append(elbow_angle)

                # Deadlift (knee angle)
                hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP].x, 
                      landmarks[mp_pose.PoseLandmark.LEFT_HIP].y]
                knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE].x, 
                       landmarks[mp_pose.PoseLandmark.LEFT_KNEE].y]
                ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE].x, 
                        landmarks[mp_pose.PoseLandmark.LEFT_ANKLE].y]
                knee_angle = calculate_angle(hip, knee, ankle)
                deadlift_angles.append(knee_angle)

                # Classify activity
                activity = classify_activity(curl_angles, deadlift_angles, video_file)

                # Display results
                cv2.putText(lk_frame, f"Activity: {activity}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                cv2.putText(hs_frame, f"Activity: {activity}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                if elbow_angle:
                    cv2.putText(lk_frame, f"Elbow: {elbow_angle:.1f}", (10, 60), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                    cv2.putText(hs_frame, f"Elbow: {elbow_angle:.1f}", (10, 60), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                if knee_angle:
                    cv2.putText(lk_frame, f"Knee: {knee_angle:.1f}", (10, 90), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                    cv2.putText(hs_frame, f"Knee: {knee_angle:.1f}", (10, 90), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

            # Main Comparison
            main_comp = np.hstack((lk_frame, hs_frame))
            cv2.putText(main_comp, "LK + Mediapipe", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(main_comp, "HS + Mediapipe", (frame_width + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            # Write to CSV
            with open(csv_file, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([frame_count, activity, lk_dots, hs_avg_mag, farneback_avg_mag, 
                                gmm_foreground_pixels, fd_foreground_pixels, elbow_angle, knee_angle, 
                                gmm_time, fd_time])

            # Write to videos
            out_lk_main.write(lk_frame)
            out_hs_main.write(hs_frame)
            out_flow.write(flow_vis)
            out_gmm.write(fg_result_gmm)
            out_flow_comp.write(flow_comp)
            out_main_comp.write(main_comp)
            out_bg_comp.write(bg_comp)

            # Display
            cv2.imshow('LK + Mediapipe', lk_frame)
            cv2.imshow('HS + Mediapipe', hs_frame)
            cv2.imshow('Dense Optical Flow (Farneback)', flow_vis)
            cv2.imshow('Background Subtraction (GMM)', fg_result_gmm)
            cv2.imshow('Flow Comparison (LK vs HS)', flow_comp)
            cv2.imshow('Main Comparison (LK vs HS + Mediapipe)', main_comp)
            cv2.imshow('Background Comparison (GMM vs FD)', bg_comp)

            prev_gray = frame_gray.copy()
            prev_frame_fd = frame_gray.copy()

            if cv2.waitKey(30) & 0xFF == ord('q'):
                break

        # Summary
        with open(csv_file, mode='r') as file:
            reader = csv.reader(file)
            next(reader)
            data = list(reader)
            lk_dots_avg = np.mean([float(row[2]) for row in data])
            hs_avg_mag_avg = np.mean([float(row[3]) for row in data])
            gmm_pixels_avg = np.mean([float(row[5]) for row in data])
            fd_pixels_avg = np.mean([float(row[6]) for row in data])
            gmm_time_avg = np.mean([float(row[9]) for row in data])
            fd_time_avg = np.mean([float(row[10]) for row in data])
            activities = set([row[1] for row in data])

        print(f"\nAnalysis Summary for {video_file}:")
        print(f" - Detected Activities: {activities}")
        print(f" - Lucas-Kanade Avg Dots: {lk_dots_avg:.2f}")
        print(f" - Horn-Schunck Avg Magnitude: {hs_avg_mag_avg:.2f}")
        print(f" - GMM Avg Foreground Pixels: {gmm_pixels_avg:.2f}, Avg Time: {gmm_time_avg:.4f}s")
        print(f" - Frame Differencing Avg Foreground Pixels: {fd_pixels_avg:.2f}, Avg Time: {fd_time_avg:.4f}s")
        print(f"Outputs saved in {output_folder}:")
        print(f" - LK + Mediapipe: {os.path.basename(out_lk_main_path)}")
        print(f" - HS + Mediapipe: {os.path.basename(out_hs_main_path)}")
        print(f" - Farneback: {os.path.basename(out_flow_path)}")
        print(f" - GMM: {os.path.basename(out_gmm_path)}")
        print(f" - Flow Comparison: {os.path.basename(out_flow_comp_path)}")
        print(f" - Main Comparison: {os.path.basename(out_main_comp_path)}")
        print(f" - Background Comparison: {os.path.basename(out_bg_comp_path)}")
        print(f" - Numerical Analysis: {os.path.basename(csv_file)}")

        cap.release()
        out_lk_main.release()
        out_hs_main.release()
        out_flow.release()
        out_gmm.release()
        out_flow_comp.release()
        out_main_comp.release()
        out_bg_comp.release()

    cv2.destroyAllWindows()
    pose.close()

    # Activity Recognition Review
    print("\n=== Activity Recognition Review ===")
    print("Activity Classification:")
    print(f"- Dumbbell Curl: Detected if elbow angle changes > {ANGLE_THRESHOLD_CURL}° over {WINDOW_SIZE} frames")
    print(f"- Deadlift: Detected if knee angle changes > {ANGLE_THRESHOLD_DEADLIFT}° over {WINDOW_SIZE} frames")
    print("Performance Notes:")
    print("- Accuracy depends on clear landmark detection and consistent motion.")
    print("- Limitations: Noise, occlusion, or side angles may affect results.")
    print("Potential Improvements:")
    print("- Use a trained ML model (e.g., LSTM) with labeled data.")
    print("- Incorporate optical flow features for motion direction.")
    print("- Analyze multiple joints for more robust classification.")

if __name__ == "__main__":
    main()