import cv2
import numpy as np
import matplotlib.pyplot as plt

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

def detect_licks(video_path, roi, threshold=30, min_movement_area=150, cooldown_frames=4, show_video_with_licks=False, playback_speed=1.0, sync_pulse_periods=None):
    """
    Detects mouse licks in a video using frame differencing within a specified ROI.
    Optionally displays the video with lick indicators and controls playback speed.

    Args:
        video_path (str): The path to the video file.
        roi (tuple): (x, y, w, h) of the Region of Interest.
        threshold (int): Pixel intensity difference threshold for movement detection.
        min_movement_area (int): Minimum area of movement (pixels) to be considered a lick.
        cooldown_frames (int): Number of frames to ignore after detecting a lick to prevent multiple counts.
        show_video_with_licks (bool): If True, displays the video with visual lick indicators.
        playback_speed (float): Controls the video playback speed (1.0 for normal, <1.0 for slower, >1.0 for faster).
        sync_pulse_periods (list): A list of [start_time, end_time] tuples for active sync pulses.

    Returns:
        tuple: (lick_timestamps, lick_rate_bpm, video_duration_seconds)
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return None, None, None

    x, y, w, h = roi
    
    # Read the first frame
    ret, prev_frame = cap.read()
    if not ret:
        print("Error: Could not read the first frame.")
        cap.release()
        return None, None, None

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
        
        # Find contours of moving objects
        contours, _ = cv2.findContours(dilated.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        movement_detected = False
        largest_contour_area = 0
        largest_contour = None
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > min_movement_area:
                movement_detected = True
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
            if sync_pulse_periods:
                is_sync_active = False
                for start_t, end_t in sync_pulse_periods:
                    if start_t <= current_time < end_t:
                        is_sync_active = True
                        break
                if is_sync_active:
                    cv2.putText(frame, "Sync Pulse ON", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA) # Yellow text

            cv2.imshow("Lick Detection Video", frame)
            if cv2.waitKey(wait_key_delay) & 0xFF == ord('q'): # Press 'q' to quit video playback
                break

    cap.release()
    cv2.destroyAllWindows()

    video_duration_seconds = total_frames / fps if fps > 0 else 0
    lick_rate_bpm = (len(lick_timestamps) / video_duration_seconds) * 60 if video_duration_seconds > 0 else 0

    return lick_timestamps, lick_rate_bpm, video_duration_seconds

def parse_gpio_data(gpio_file_path):
    """
    Parses GPIO data from a CSV file to identify sync pulse periods.

    Args:
        gpio_file_path (str): Path to the GPIO CSV file.

    Returns:
        list: A list of [start_time, end_time] tuples (in seconds) for active sync pulses.
    """
    sync_pulse_periods = []
    current_pulse_start = None
    last_time_in_seconds = 0 # To handle potential end-of-file pulse

    with open(gpio_file_path, 'r') as f:
        for line in f:
            parts = line.strip().split(',') # Assuming CSV format
            if len(parts) >= 3:
                gpio_val_str = parts[0]
                time_str = parts[2]

                try:
                    gpio_val = int(gpio_val_str)
                    
                    # Parse MM:SS.ms timestamp
                    min_sec_ms = time_str.split(':')
                    if len(min_sec_ms) == 2:
                        minutes = int(min_sec_ms[0])
                        seconds_ms = float(min_sec_ms[1])
                        current_time_in_seconds = minutes * 60 + seconds_ms
                        last_time_in_seconds = current_time_in_seconds # Update last_time
                    else:
                        continue # Skip malformed timestamps

                    # Check for active pulse (value > 0 indicates pulse ON)
                    if gpio_val > 0: 
                        if current_pulse_start is None:
                            current_pulse_start = current_time_in_seconds
                    else:
                        if current_pulse_start is not None:
                            # End of pulse, store the period
                            sync_pulse_periods.append([current_pulse_start, current_time_in_seconds])
                            current_pulse_start = None
                except ValueError:
                    continue # Skip lines with parsing errors

    # If a pulse was active until the end of the file, use the last recorded timestamp as end time
    if current_pulse_start is not None:
        sync_pulse_periods.append([current_pulse_start, last_time_in_seconds])

    return sync_pulse_periods

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
    video_path = input("Enter the path to the mouse licking video: ")

    selected_roi = select_roi(video_path)

    if selected_roi:
        print(f"Selected ROI: {selected_roi}")
        
        show_video = input("Do you want to show video playback with lick detection? (y/n): ").lower()
        show_video_with_licks = (show_video == 'y')

        playback_speed = 1.0
        if show_video_with_licks:
            try:
                speed_input = float(input("Enter playback speed (e.g., 1.0 for normal, 0.5 for half, 2.0 for double): "))
                if speed_input > 0:
                    playback_speed = speed_input
                else:
                    print("Playback speed must be positive. Using default (1.0).")
            except ValueError:
                print("Invalid input for playback speed. Using default (1.0).")

        gpio_file_path = input("Enter the path to the GPIO file (e.g., /path/to/sync_pulses.csv) or leave blank if not applicable: ")
        sync_pulse_periods = []
        if gpio_file_path:
            sync_pulse_periods = parse_gpio_data(gpio_file_path)
            if not sync_pulse_periods:
                print("Warning: No sync pulse periods found in the provided GPIO file.")

        # Threshold is now a default value in detect_licks, not user-prompted
        # min_movement_area = 150, cooldown_frames = 4
        lick_timestamps, lick_rate, video_duration = detect_licks(video_path, selected_roi, 
                                                                show_video_with_licks=show_video_with_licks, 
                                                                playback_speed=playback_speed, 
                                                                sync_pulse_periods=sync_pulse_periods)
        
        if lick_timestamps is not None:
            print(f"Detected lick rate: {lick_rate:.2f} licks per minute")
            print(f"Total licks: {len(lick_timestamps)}")
            print(f"Lick timestamps (seconds): {lick_timestamps}")
            plot_licks(lick_timestamps, video_duration)
    else:
        print("No ROI selected or an error occurred during ROI selection.")
