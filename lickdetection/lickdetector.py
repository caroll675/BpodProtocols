import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import sync_data
import pickle # Added import for pickle


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
    roi = cv2.selectROI("Select ROI", frame, fromCenter=False, showCrosshair=True)
    cv2.destroyWindow("Select ROI")

    x, y, w, h = roi
    if w > 0 and h > 0:
        return (x, y, w, h)
    else:
        return None

def detect_licks(video_path, roi, threshold=30, min_movement_percent=25, cooldown_frames=2, show_video_with_licks=False, playback_speed=0.5, synced_gpio_file=None):
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
        return None, None, None, None, None # Added None for lick_frames

    x, y, w, h = roi
    roi_area = w * h if w > 0 and h > 0 else 0
    # Translate percentage to a pixel count threshold based on ROI size
    min_changed_pixels = (roi_area * (min_movement_percent / 100.0)) if roi_area > 0 else float('inf')
    
    # Get video properties first
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Calculate middle frame position
    middle_frame = total_frames // 2
    
    # Set video position to middle frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, middle_frame)
    
    # Read the middle frame as reference
    ret, prev_frame = cap.read()
    if not ret:
        print("Error: Could not read the middle frame.")
        cap.release()
        return None, None, None, None, None # Added None for lick_frames
    
    # Reset video position to beginning for processing
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    prev_roi_gray = cv2.cvtColor(prev_frame[y:y+h, x:x+w], cv2.COLOR_BGR2GRAY)
    prev_roi_blur = cv2.GaussianBlur(prev_roi_gray, (21, 21), 0)

    lick_timestamps = []
    # Initialize lick_frames as a NumPy array of zeros with the same length as synced_gpio_file
    lick_frames_array = np.zeros(len(synced_gpio_file), dtype=int)
    in_cooldown = 0

    # Calculate delay for cv2.waitKey
    wait_key_delay = 1 if fps == 0 else max(1, int(1000 / (fps * playback_speed)))

    frame_num = 0 # Start frame_num from 0 for correct indexing with synced_gpio_file
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Ensure index is within bounds of synced_gpio_file
        if frame_num >= len(synced_gpio_file):
            break

        current_time = (frame_num / fps) if fps > 0 else 0
        current_frame = frame_num # This is the index for lick_frames_array
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
            lick_frames_array[current_frame] = 1 # Mark lick at current_frame
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
            this_state = str(state[current_frame]) # Use current_frame as index
            this_sync = str(syncpulse[current_frame]) # Use current_frame as index
            this_trial_type = str(trialtype[current_frame]) # Use current_frame as index
            if this_trial_type != 'None':
                cv2.putText(frame, 'Trial Type: ' + this_trial_type, (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)
            if this_sync != 'None':
                cv2.putText(frame, this_sync, (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA) # Yellow text
            if this_state != 'None':
                cv2.putText(frame, this_state, (10, 190), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.imshow("Lick Detection Video", frame)
            if cv2.waitKey(wait_key_delay) & 0xFF == ord('q'): # Press 'q' to quit video playback
                break
        # index += 1 # Remove this, as frame_num is used as the index

    cap.release()
    cv2.destroyAllWindows()

    video_duration_seconds = total_frames / fps if fps > 0 else 0
   
    return lick_timestamps, lick_frames_array, video_duration_seconds, fps




if __name__ == "__main__":
    protocol = 'OdorWater_VariableDelay_FreeRewards'
    print('protocol: ' + protocol)
    mouse_id = "CC" + input("MouseID: CC")
    date = "2025" + input("Date (eg. 0917): 2025")
    ########################################################
    # Temporary file for saving/loading lick detection data
    temp_data_file = f'{mouse_id}_{date}_temp.pkl'

    lick_timestamps = None
    video_duration = None
    fps = None
    lick_frames_array = None
    selected_roi = None
    synced_gpio_file = None

    if Path(temp_data_file).exists():
        print(f"Loading temporary data from {temp_data_file}")
        with open(temp_data_file, 'rb') as f:
            temp_data = pickle.load(f)
            lick_timestamps = temp_data.get('lick_timestamps')
            video_duration = temp_data.get('video_duration')
            fps = temp_data.get('fps')
            synced_gpio_file = temp_data.get('synced_gpio_file')
            selected_roi = temp_data.get('selected_roi')
            lick_frames = temp_data.get('lick_frames_array') 
    else:
        root_bpod_dir = Path(r"\\140.247.90.110\homes2\Carol\BpodData")
        session_dir = root_bpod_dir / mouse_id / protocol / "Session Data"
        pattern = f"{mouse_id}_{protocol}_{date}_*.mat"
        # latest match
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

        enter_roi_manually = input("Enter roi manually? (y/n): ").lower()
        if enter_roi_manually == 'y':
            roi_input = input("Enter ROI as (x, y, w, h): ")
            selected_roi = tuple(map(int, roi_input.strip("()").split(",")))

        else:
            selected_roi = select_roi(video_path)

        if selected_roi:
            print(f"Selected ROI: {selected_roi}")
            
            show_video = input("Video playback with lick detection? (y/n): ").lower()
            show_video_with_licks = (show_video == 'y')

            playback_speed = 0.5
            if show_video_with_licks:
                try:
                    speed_input = float(input("Enter playback speed:"))
                    if speed_input > 0:
                        playback_speed = speed_input
                    else:
                        print("Playback speed must be positive. Using default (0.5).")
                except ValueError:
                    print("Invalid input for playback speed. Using default (0.5).")


            lick_timestamps, lick_frames, video_duration, fps = detect_licks(video_path, selected_roi, 
                                                                      show_video_with_licks=show_video_with_licks, 
                                                                      playback_speed=playback_speed, 
                                                                      synced_gpio_file=synced_gpio_file)

            # Save data temporarily
            with open(temp_data_file, 'wb') as f:
                pickle.dump({
                    'lick_timestamps': lick_timestamps,
                    'video_duration': video_duration,
                    'fps': fps,
                    'synced_gpio_file': synced_gpio_file,
                    'selected_roi': selected_roi,
                    'lick_frames_array': lick_frames # Save as array
                }, f)
            print(f"Data saved to {temp_data_file}")    
    synced_gpio_file['lickframe'] = lick_frames
    # no none rows 
    synced_gpio_file = synced_gpio_file[synced_gpio_file['trialtype']!='None']
    unique_trial_type = np.sort(synced_gpio_file['trialtype'].unique())
    unique_trial_type = unique_trial_type[1:]

    all_lick_rates_per_trial_type = {} # Dictionary to store organized lick rates

    window = 6
    for tt in unique_trial_type:
        all_lick_rates_per_trial_type[tt] = {}
        tt_df = synced_gpio_file[synced_gpio_file['trialtype'] == tt].copy() # .copy() to avoid SettingWithCopyWarning
        unique_trial_start_time = tt_df['trialstarttime'].unique()
        
        for trial_idx, t_start in enumerate(unique_trial_start_time):
            single_trial_df = tt_df[tt_df['trialstarttime']==t_start]
            
            lick_rates_per_second = []
            # Iterate through the trial in 1-second (frames_per_second) chunks
            half_fps = int(fps/window)
            for i in range(half_fps, len(single_trial_df)-half_fps, half_fps):
                # Extract a 1-second chunk of lick_frames
                chunk = single_trial_df['lickframe'].iloc[i-half_fps : i]
                # Count licks (number of 0.5 s) in this chunk
                licks_in_second = chunk.sum()*window
                lick_rates_per_second.append(licks_in_second)
            
            all_lick_rates_per_trial_type[tt][trial_idx] = lick_rates_per_second
    
    # plot lick rate over time for each trial type 
    # plot mean and sem to represent trials in each trial type
    plt.figure(figsize=(20, 8))
    color_map = {
        # 0: np.array([41, 114, 112]) / 255,
        1: np.array([230, 109, 80]) / 255,
        2: np.array([231, 198, 107]) / 255,
        3: np.array([138, 176, 124]) / 255,
        4: np.array([41, 157, 143]) / 255
    }

    max_y = 0
    for tt, trials_dict in all_lick_rates_per_trial_type.items():

        max_len = max(len(rates) for rates in trials_dict.values())

        # Pad each trial's lick rate list to max_len with NaNs
        padded_dict = {
            trial_idx: rates + [np.nan] * (max_len - len(rates))
            for trial_idx, rates in trials_dict.items()
        }

        df = pd.DataFrame(padded_dict)

        df.index.name = "second" # index name is second, column name is trial_idx

        mean_series = df.mean(axis=1, skipna=True) 
        sem_series  = df.sem(axis=1, skipna=True)

        if max(mean_series + sem_series) > max_y:
            max_y = max(mean_series + sem_series)

        # Time axis in seconds
        t = mean_series.index.to_numpy()
        plt.plot(t, mean_series, label=f"TrialType {tt}", color=color_map[tt])
        plt.fill_between(t, mean_series - sem_series, mean_series + sem_series, color=color_map[tt], alpha=0.2)


    odor_start_time = 1.5
    odor_end_time = 2
    ymin = 0
    ymax = max_y + 1
    vline_times = [odor_start_time, odor_end_time, odor_end_time+0.75, odor_end_time+1.5, odor_end_time+3, odor_end_time+6]
    vline_times = [i*window for i in vline_times]
    for t_vline in vline_times:
        plt.vlines(t_vline, ymin, ymax, color='purple', linestyle='-', linewidth=1, alpha=0.7)
    plt.axvspan(odor_start_time*window, odor_end_time*window, ymin=ymin, ymax=ymax, facecolor='purple', alpha=0.1)

    x_ticks_labels = ['odor', '0.75', '1.5', '3', '6']
    x_tick_positions = [odor_start_time, odor_end_time+0.75, odor_end_time+1.5, odor_end_time+3, odor_end_time+6]
    x_tick_positions = [i*window for i in x_tick_positions]
    plt.xticks(x_tick_positions, x_ticks_labels)
    plt.grid(True, axis='x', linestyle='--')
    
    # ticklabels = ['odor', '0.75', '1.5', '3', '6']
    # plt.xticks(x_ticks, ticklabels)
    plt.xlim([0, 15*window])
    plt.legend()
    plt.xlabel("Time (s)")
    plt.ylabel("Licks / s")
    plt.title(f'{mouse_id}_{date}')
    fig_root_dir = Path(r"\\140.247.90.110\homes2\Carol\LickData")
    fig_file_name = f'{mouse_id}_{date}.pdf'
    full_fig_path = fig_root_dir / fig_file_name
    full_fig_path.parent.mkdir(parents=True, exist_ok=True) # Ensure directory exists
    plt.savefig(full_fig_path)
    # plt.show()

