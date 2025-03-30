import cv2
import numpy as np
import mediapipe as mp
import os
import csv
import time

# Initialize Mediapipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Optical Flow (Lucas-Kanade) parameters
lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# Horn-Schunck parameters
hs_params = dict(iterations=10, lmbda=0.01)  # Lambda controls smoothness

# Create output folder
output_folder = "Hand_Gesture_Analysis_Outputs"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# List of video files to process (hand gesture clips)
video_files = [
    'Gesture_Vids\\fist.mp4',
    'Gesture_Vids\\four.mp4',
    'Gesture_Vids\\me.mp4',
    'Gesture_Vids\\one.mp4',
    'Gesture_Vids\\small.mp4'
]  # Update with your gesture clip paths

# Horn-Schunck Optical Flow implementation
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

    # Ensure portrait orientation (height > width)
    if frame_height < frame_width:  # If landscape, swap dimensions
        frame_width, frame_height = frame_height, frame_width
        print(f"Detected landscape input for {video_file}. Adjusting to portrait.")

    # Set up video writers
    base_name = os.path.basename(video_file).split('.')[0]
    out_lk_main_path = os.path.join(output_folder, f"output_{base_name}_lk_main.mp4")  # LK + Mediapipe
    out_hs_main_path = os.path.join(output_folder, f"output_{base_name}_hs_main.mp4")  # HS + Mediapipe
    out_flow_path = os.path.join(output_folder, f"output_{base_name}_farneback.mp4")
    out_gmm_path = os.path.join(output_folder, f"output_{base_name}_gmm.mp4")
    out_flow_comp_path = os.path.join(output_folder, f"output_{base_name}_flow_comparison.mp4")  # LK vs HS flow
    out_main_comp_path = os.path.join(output_folder, f"output_{base_name}_main_comparison.mp4")  # LK + Mediapipe vs HS + Mediapipe
    out_bg_comp_path = os.path.join(output_folder, f"output_{base_name}_bg_comparison.mp4")  # GMM vs FD
    
    out_lk_main = cv2.VideoWriter(out_lk_main_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))
    out_hs_main = cv2.VideoWriter(out_hs_main_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))
    out_flow = cv2.VideoWriter(out_flow_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))
    out_gmm = cv2.VideoWriter(out_gmm_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))
    out_flow_comp = cv2.VideoWriter(out_flow_comp_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width * 2, frame_height))
    out_main_comp = cv2.VideoWriter(out_main_comp_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width * 2, frame_height))
    out_bg_comp = cv2.VideoWriter(out_bg_comp_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width * 2, frame_height))

    # Set up CSV file
    csv_file = os.path.join(output_folder, f"analysis_{base_name}.csv")
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Frame', 'LK_Dots', 'HS_Avg_Magnitude', 'Farneback_Avg_Magnitude', 
                         'GMM_Foreground_Pixels', 'FD_Foreground_Pixels', 
                         'Index_Finger_Angle', 'Middle_Finger_Angle', 'Ring_Finger_Angle', 
                         'Pinky_Angle', 'Gesture', 'GMM_Time', 'FD_Time'])

    # Read first frame
    ret, prev_frame = cap.read()
    if not ret:
        print(f"Error reading first frame of {video_file}. Skipping...")
        cap.release()
        continue

    # Rotate frame if landscape to match portrait output
    if prev_frame.shape[1] > prev_frame.shape[0]:  # Landscape input
        prev_frame = cv2.rotate(prev_frame, cv2.ROTATE_90_COUNTERCLOCKWISE)  # Adjusted for orientation

    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    hsv = np.zeros_like(prev_frame)
    hsv[..., 1] = 255

    # Detect initial points for Lucas-Kanade
    p0 = cv2.goodFeaturesToTrack(prev_gray, maxCorners=100, qualityLevel=0.3, minDistance=7)

    # Background Subtraction (GMM and Frame Differencing)
    bg_subtractor = cv2.createBackgroundSubtractorMOG2()
    prev_frame_fd = prev_gray.copy()

    print(f"Processing {video_file}...")
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Rotate frame if landscape to match portrait output
        if frame.shape[1] > frame.shape[0]:  # Landscape input
            frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)  # Adjusted for orientation

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_count += 1

        # 1. Optical Flow (Lucas-Kanade)
        lk_start = time.time()
        lk_dots = 0
        lk_frame = frame.copy()  # Frame for LK + Mediapipe
        if p0 is not None:
            p1, st, err = cv2.calcOpticalFlowPyrLK(prev_gray, frame_gray, p0, None, **lk_params)
            if p1 is not None:
                good_new = p1[st == 1]
                good_old = p0[st == 1]
                lk_dots = len(good_new)
                for i, (new, old) in enumerate(zip(good_new, good_old)):
                    a, b = new.ravel()
                    c, d = old.ravel()
                    lk_frame = cv2.line(lk_frame, (int(a), int(b)), (int(c), int(d)), (0, 255, 0), 2)
                    lk_frame = cv2.circle(lk_frame, (int(a), int(b)), 5, (0, 0, 255), -1)
                p0 = good_new.reshape(-1, 1, 2)
        lk_time = time.time() - lk_start

        # 2. Optical Flow (Horn-Schunck)
        hs_start = time.time()
        hs_frame = frame.copy()  # Frame for HS + Mediapipe
        hs_flow = horn_schunck(prev_gray, frame_gray, **hs_params)
        hs_mag, hs_ang = cv2.cartToPolar(hs_flow[..., 0], hs_flow[..., 1])
        hs_avg_mag = np.mean(hs_mag)
        hs_hsv = np.zeros_like(frame)
        hs_hsv[..., 1] = 255
        hs_hsv[..., 0] = hs_ang * 180 / np.pi / 2
        hs_hsv[..., 2] = cv2.normalize(hs_mag, None, 0, 255, cv2.NORM_MINMAX)
        hs_vis = cv2.cvtColor(hs_hsv, cv2.COLOR_HSV2BGR)
        hs_time = time.time() - hs_start

        # Flow Comparison Visualization (LK vs HS)
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

        # 5. Background Subtraction (Frame Differencing)
        fd_start = time.time()
        fd_diff = cv2.absdiff(prev_frame_fd, frame_gray)
        _, fg_mask_fd = cv2.threshold(fd_diff, 30, 255, cv2.THRESH_BINARY)
        fg_result_fd = cv2.bitwise_and(frame, frame, mask=fg_mask_fd)
        fd_foreground_pixels = np.sum(fg_mask_fd > 0)
        fd_time = time.time() - fd_start

        # Background Subtraction Comparison Visualization (GMM vs FD)
        bg_comp = np.hstack((fg_result_gmm, fg_result_fd))
        cv2.putText(bg_comp, "GMM", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(bg_comp, "Frame Differencing", (frame_width + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # 6. Hand Detection (Mediapipe Hands) for both LK and HS frames
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)
        index_angle = None
        middle_angle = None
        ring_angle = None
        pinky_angle = None
        gesture = "Unknown"

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw on LK frame
                mp_drawing.draw_landmarks(lk_frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                # Draw on HS frame
                mp_drawing.draw_landmarks(hs_frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Calculate finger angles
                landmarks = hand_landmarks.landmark
                def calc_angle(p1, p2, p3):
                    v1 = [p1.x - p2.x, p1.y - p2.y]
                    v2 = [p3.x - p2.x, p3.y - p2.y]
                    dot = v1[0] * v2[0] + v1[1] * v2[1]
                    mag1 = np.sqrt(v1[0]**2 + v1[1]**2)
                    mag2 = np.sqrt(v2[0]**2 + v2[1]**2)
                    angle = np.arccos(dot / (mag1 * mag2)) * 180.0 / np.pi
                    return angle

                index_mcp = landmarks[mp_hands.HandLandmark.INDEX_FINGER_MCP]
                index_pip = landmarks[mp_hands.HandLandmark.INDEX_FINGER_PIP]
                index_tip = landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                index_angle = calc_angle(index_mcp, index_pip, index_tip)

                middle_mcp = landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
                middle_pip = landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_PIP]
                middle_tip = landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
                middle_angle = calc_angle(middle_mcp, middle_pip, middle_tip)

                ring_mcp = landmarks[mp_hands.HandLandmark.RING_FINGER_MCP]
                ring_pip = landmarks[mp_hands.HandLandmark.RING_FINGER_PIP]
                ring_tip = landmarks[mp_hands.HandLandmark.RING_FINGER_TIP]
                ring_angle = calc_angle(ring_mcp, ring_pip, ring_tip)

                pinky_mcp = landmarks[mp_hands.HandLandmark.PINKY_MCP]
                pinky_pip = landmarks[mp_hands.HandLandmark.PINKY_PIP]
                pinky_tip = landmarks[mp_hands.HandLandmark.PINKY_TIP]
                pinky_angle = calc_angle(pinky_mcp, pinky_pip, pinky_tip)

                # Gesture classification
                if index_angle < 60 and middle_angle < 60 and ring_angle < 60 and pinky_angle < 60:
                    gesture = "Fist"
                elif index_angle > 120 and middle_angle > 120 and ring_angle > 120 and pinky_angle > 120:
                    gesture = "Four"
                elif index_angle < 60 and middle_angle > 120 and ring_angle < 60 and pinky_angle < 60:
                    gesture = "Me"
                elif index_angle > 120 and middle_angle < 60 and ring_angle < 60 and pinky_angle < 60:
                    gesture = "One"
                elif index_angle < 60 and middle_angle < 60 and ring_angle < 60 and pinky_angle > 120:
                    gesture = "Small"

                # Display gesture on both frames
                cv2.putText(lk_frame, f"Gesture: {gesture}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                cv2.putText(hs_frame, f"Gesture: {gesture}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        # Main Comparison Visualization (LK + Mediapipe vs HS + Mediapipe)
        main_comp = np.hstack((lk_frame, hs_frame))
        cv2.putText(main_comp, "LK + Mediapipe", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(main_comp, "HS + Mediapipe", (frame_width + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Write numerical data to CSV
        with open(csv_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([frame_count, lk_dots, hs_avg_mag, farneback_avg_mag, 
                             gmm_foreground_pixels, fd_foreground_pixels, 
                             index_angle, middle_angle, ring_angle, pinky_angle, gesture, 
                             gmm_time, fd_time])

        # Write to output videos
        out_lk_main.write(lk_frame)  # LK + Mediapipe
        out_hs_main.write(hs_frame)  # HS + Mediapipe
        out_flow.write(flow_vis)
        out_gmm.write(fg_result_gmm)
        out_flow_comp.write(flow_comp)
        out_main_comp.write(main_comp)
        out_bg_comp.write(bg_comp)

        # Display results
        cv2.imshow('LK + Mediapipe', lk_frame)
        cv2.imshow('HS + Mediapipe', hs_frame)
        cv2.imshow('Dense Optical Flow (Farneback)', flow_vis)
        cv2.imshow('Background Subtraction (GMM)', fg_result_gmm)
        cv2.imshow('Flow Comparison (LK vs HS)', flow_comp)
        cv2.imshow('Main Comparison (LK vs HS + Mediapipe)', main_comp)
        cv2.imshow('Background Comparison (GMM vs FD)', bg_comp)

        # Update previous frame
        prev_gray = frame_gray.copy()
        prev_frame_fd = frame_gray.copy()

        # Exit on 'q' press
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    # Analysis Summary for Terminal
    with open(csv_file, mode='r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip header
        data = list(reader)
        lk_dots_avg = np.mean([float(row[1]) for row in data])
        hs_avg_mag_avg = np.mean([float(row[2]) for row in data])
        gmm_pixels_avg = np.mean([float(row[4]) for row in data])
        fd_pixels_avg = np.mean([float(row[5]) for row in data])
        gmm_time_avg = np.mean([float(row[11]) for row in data])
        fd_time_avg = np.mean([float(row[12]) for row in data])

    print(f"\nAnalysis Summary for {video_file}:")
    print(f" - Lucas-Kanade Avg Dots: {lk_dots_avg:.2f}")
    print(f" - Horn-Schunck Avg Magnitude: {hs_avg_mag_avg:.2f}")
    print(f" - GMM Avg Foreground Pixels: {gmm_pixels_avg:.2f}, Avg Time: {gmm_time_avg:.4f}s")
    print(f" - Frame Differencing Avg Foreground Pixels: {fd_pixels_avg:.2f}, Avg Time: {fd_time_avg:.4f}s")
    print(f"Outputs saved in {output_folder}:")
    print(f" - LK + Mediapipe: {os.path.basename(out_lk_main_path)}")
    print(f" - HS + Mediapipe: {os.path.basename(out_hs_main_path)}")
    print(f" - Farneback: {os.path.basename(out_flow_path)}")
    print(f" - GMM: {os.path.basename(out_gmm_path)}")
    print(f" - Flow Comparison (LK vs HS): {os.path.basename(out_flow_comp_path)}")
    print(f" - Main Comparison (LK vs HS + Mediapipe): {os.path.basename(out_main_comp_path)}")
    print(f" - Background Comparison (GMM vs FD): {os.path.basename(out_bg_comp_path)}")
    print(f" - Numerical Analysis: {os.path.basename(csv_file)}")

    # Release resources
    cap.release()
    out_lk_main.release()
    out_hs_main.release()
    out_flow.release()
    out_gmm.release()
    out_flow_comp.release()
    out_main_comp.release()
    out_bg_comp.release()

# Clean up
cv2.destroyAllWindows()
hands.close()

# Activity Recognition Review
print("\n=== Gesture Recognition Review ===")
print("Hand Gesture Estimation and Classification Performance:")
print("- Mediapipe Hands was used to estimate finger angles for gesture recognition.")
print("- For gestures like 'Fist', 'Four', 'Me', 'One', and 'Small', accuracy depends on clear hand visibility.")
print("  - Advantage: Robust detection of finger joints in good lighting.")
print("  - Limitation: Occlusion, fast motion, or poor lighting may reduce accuracy.")
print("Visualizations (main output videos) support these conclusions by showing hand landmarks and gesture labels.")
print("\nMethod Comparison:")
print("- Lucas-Kanade vs Horn-Schunck: LK tracks sparse points (faster, less detail), HS provides dense flow (slower, smoother).")
print("- GMM vs Frame Differencing: GMM adapts to background (more accurate), FD is simpler and faster but less robust.")
print("See 'Flow Comparison', 'Main Comparison', and 'Background Comparison' videos for visual evidence.")