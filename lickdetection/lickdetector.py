import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import sync_data

def select_roi(video_path):
    """
    Loads the first frame of a video and allows the user to select a Region of Interest (ROI).

    Args:
        video_path (str): The path to the video file.

    Returns:
        tuple: (x, y, w, h) of the selected ROI, or None if no ROI is selected.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return None

    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read the first frame.")
        return None

    cap.release()

    # Select ROI
    print("Select ROI by drawing a rectangle and pressing ENTER or SPACE. Press 'c' to cancel.")
    roi = cv2.selectROI("Select ROI", frame, fromCenter=False, showCrosshair=True)
    cv2.destroyWindow("Select ROI")

    x, y, w, h = roi
    if w > 0 and h > 0:
        return (x, y, w, h)
    else:
        return None

def detect_licks(video_path, roi, threshold=30, min_movement_percent=20, cooldown_frames=2, show_video_with_licks=False, playback_speed=0.5, synced_gpio_file=None):
    """
    Detects mouse licks in a video using frame differencing within a specified ROI.
    Optionally displays the video with lick indicators and controls playback speed.

    Args:
        video_path (str): The path to the video file.
        roi (tuple): (x, y, w, h) of the Region of Interest.
        threshold (int): Pixel intensity difference threshold for movement detection.
        min_movement_percent (float): Minimum percentage (0-100) of changed pixels within ROI to count as a lick.
        cooldown_frames (int): Number of frames to ignore after detecting a lick to prevent multiple counts.
        show_video_with_licks (bool): If True, displays the video with visual lick indicators.
        playback_speed (float): Controls the video playback speed (1.0 for normal, <1.0 for slower, >1.0 for faster).
        sync_pulse_periods (list): A list of [start_time, end_time] tuples for active sync pulses.

    Returns:
        list: lick_timestamps
    """
    syncpulse = synced_gpio_file['syncpulse']
    state = synced_gpio_file['state']
    trialtype = synced_gpio_file['trialtype']
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return None, None, None, None

    x, y, w, h = roi
    roi_area = w * h if w > 0 and h > 0 else 0
    # Translate percentage to a pixel count threshold based on ROI size
    min_changed_pixels = (roi_area * (min_movement_percent / 100.0)) if roi_area > 0 else float('inf')
    
    # Read the first frame
    ret, prev_frame = cap.read()
    if not ret:
        print("Error: Could not read the first frame.")
        cap.release()
        return None, None, None, None

    prev_roi_gray = cv2.cvtColor(prev_frame[y:y+h, x:x+w], cv2.COLOR_BGR2GRAY)
    prev_roi_blur = cv2.GaussianBlur(prev_roi_gray, (21, 21), 0)

    lick_timestamps = []
    in_cooldown = 0
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Calculate delay for cv2.waitKey
    wait_key_delay = 1 if fps == 0 else max(1, int(1000 / (fps * playback_speed)))

    frame_num = 1
    index = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        current_time = (frame_num / fps) if fps > 0 else 0
        frame_num += 1
        current_roi = frame[y:y+h, x:x+w]
        current_roi_gray = cv2.cvtColor(current_roi, cv2.COLOR_BGR2GRAY)
        current_roi_blur = cv2.GaussianBlur(current_roi_gray, (21, 21), 0)

        # Calculate difference
        frame_diff = cv2.absdiff(prev_roi_blur, current_roi_blur)
        
        # Threshold the difference image
        _, thresh = cv2.threshold(frame_diff, threshold, 255, cv2.THRESH_BINARY)
        
        # Dilate the thresholded image to fill in small holes
        dilated = cv2.dilate(thresh, None, iterations=2)
        
        # Find contours (for visualization) and compute overall changed pixel count for detection
        contours, _ = cv2.findContours(dilated.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Use percentage of changed pixels across the ROI as detection criterion
        changed_pixels = cv2.countNonZero(dilated)
        movement_detected = changed_pixels > min_changed_pixels

        # Track largest contour for optional visualization
        largest_contour_area = 0
        largest_contour = None
        if contours:
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > largest_contour_area:
                    largest_contour_area = area
                    largest_contour = contour
        
        if movement_detected and in_cooldown == 0:
            lick_timestamps.append(current_time)
            in_cooldown = cooldown_frames
            if show_video_with_licks and largest_contour is not None:
                # Draw bounding box around the largest moving object
                M = cv2.moments(largest_contour)
                if M['m00'] != 0:
                    cx = int(M['m10']/M['m00'])
                    cy = int(M['m01']/M['m00'])
                    cv2.circle(current_roi, (cx, cy), 3, (0, 255, 0), -1) # Green circle

        elif in_cooldown > 0:
            in_cooldown -= 1

        prev_roi_blur = current_roi_blur

        if show_video_with_licks:
            # Draw ROI rectangle on the full frame
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2) # Blue ROI
            # Put text for frame number and lick count
            cv2.putText(frame, f"Frame: {frame_num}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, f"Licks: {len(lick_timestamps)}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            
            # Check and display sync pulse information
            this_state = str(state[index])
            this_sync = str(syncpulse[index])
            this_trial_type = str(trialtype[index])
            if this_trial_type != 'None':
                cv2.putText(frame, 'Trial Type: ' + this_trial_type, (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)
            if this_sync != 'None':
                cv2.putText(frame, this_sync, (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA) # Yellow text
            if this_state != 'None':
                cv2.putText(frame, this_state, (10, 190), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.imshow("Lick Detection Video", frame)
            if cv2.waitKey(wait_key_delay) & 0xFF == ord('q'): # Press 'q' to quit video playback
                break
        index += 1

    cap.release()
    cv2.destroyAllWindows()

    video_duration_seconds = total_frames / fps if fps > 0 else 0
    lick_rate_bpm = (len(lick_timestamps) / video_duration_seconds) * 60 if video_duration_seconds > 0 else 0

    return lick_timestamps, lick_rate_bpm, video_duration_seconds, fps


def plot_licks(lick_timestamps, video_duration_seconds):
    """
    Plots the detected lick timestamps over the video duration.

    Args:
        lick_timestamps (list): A list of timestamps (in seconds) where licks were detected.
        video_duration_seconds (float): The total duration of the video in seconds.
    """
    plt.figure(figsize=(12, 4))
    plt.eventplot(lick_timestamps, orientation='horizontal', colors='blue')
    plt.xlabel("Time (seconds)")
    plt.yticks([]) # Hide y-axis ticks
    plt.title("Mouse Lick Events Over Time")
    plt.xlim(0, video_duration_seconds)
    plt.grid(True, axis='x', linestyle='--')
    plt.show()

if __name__ == "__main__":
    protocol = 'OdorWater_VariableDelay_FreeRewards'
    print('protocol: ' + protocol)
    mouse_id = "CC" + input("MouseID: CC")
    date = "2025" + input("Date (eg. 0917): 2025")

    root_bpod_dir = Path(r"\\140.247.90.110\homes2\Carol\BpodData")
    session_dir = root_bpod_dir / mouse_id / protocol / "Session Data"
    pattern = f"{mouse_id}_{protocol}_{date}_*.mat"
    # choose the latest match if multiple
    matches = list(session_dir.glob(pattern))
    bpod_data_path = max(matches, key=lambda p: p.stat().st_mtime) if matches else None
    if bpod_data_path is None:
        print("No session file found")

    root_video_dir = Path(r"\\140.247.90.110\homes2\Carol\VideoData")
    video_path = next(Path(root_video_dir).glob(f"{mouse_id}_{date}_*_cam1.avi"))
    if not video_path:
         print("No video found")

    gpio_file_path = next(Path(root_video_dir).glob(f"{mouse_id}_{date}_*_gpio1.csv"))
    
    synced_gpio_file = sync_data.sync_bpod_video(gpio_file_path=gpio_file_path, bpod_data_path=bpod_data_path)

    selected_roi = select_roi(video_path)

    if selected_roi:
        print(f"Selected ROI: {selected_roi}")
        
        show_video = input("Video playback with lick detection? (y/n): ").lower()
        show_video_with_licks = (show_video == 'y')

        playback_speed = 1.0
        if show_video_with_licks:
            try:
                speed_input = float(input("Enter playback speed:"))
                if speed_input > 0:
                    playback_speed = speed_input
                else:
                    print("Playback speed must be positive. Using default (0.5).")
            except ValueError:
                print("Invalid input for playback speed. Using default (0.5).")


        # Threshold is now a default value in detect_licks, not user-prompted
        # min_movement_area = 150, cooldown_frames = 4
        lick_timestamps, lick_rate, video_duration, fps = detect_licks(video_path, selected_roi, 
                                                                  show_video_with_licks=show_video_with_licks, 
                                                                  playback_speed=playback_speed, 
                                                                  synced_gpio_file=synced_gpio_file)
        
        if lick_timestamps is not None:
            plot_licks(lick_timestamps, video_duration)
            print(lick_timestamps)

    else:
        print("No ROI selected or an error occurred during ROI selection.")
