import cv2
import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import sync_data
import pickle # Added import for pickle

# cuda_available = False # Initialize globally

# try:
#     import cv2.cuda
#     if cv2.cuda.getCudaEnabledDeviceCount() > 0:
#         cuda_available = True
#         print('use cuda')
# except ImportError:
#     pass # cv2.cuda is not available

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

#  def detect_licks(video_path, roi, threshold=30, min_movement_percent=25, cooldown_frames=2, show_video_with_licks=False, playback_speed=0.5, synced_gpio_file=None):
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
    
    # Read the first frame
    ret, prev_frame = cap.read()

    if not ret:
        print("Error: Could not read the first frame.")
        cap.release()
        return None, None, None, None, None # Added None for lick_frames

    prev_roi_gray = cv2.cvtColor(prev_frame[y:y+h, x:x+w], cv2.COLOR_BGR2GRAY)
    prev_roi_blur = cv2.GaussianBlur(prev_roi_gray, (21, 21), 0)

    lick_timestamps = []
    # Initialize lick_frames as a NumPy array of zeros with the same length as synced_gpio_file
    lick_frames_array = np.zeros(len(synced_gpio_file), dtype=int)
    in_cooldown = 0
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

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
                cv2.putText(frame, 'TrialType: ' + this_trial_type, (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)
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

def detect_licks(
    video_path,
    roi,
    threshold=30,
    min_movement_percent=25,
    cooldown_frames=2,
    show_video_with_licks=False,
    playback_speed=0.5,
    synced_gpio_file=None,
    save_video_path=None,
    save_codec=None,
    save_fps=None,
):
    """
    Detect mouse licks using frame differencing within an ROI (Region of Interest).
    Uses PyTorch with CUDA (Compute Unified Device Architecture) on the GPU (Graphics Processing Unit)
    if available to accelerate grayscale, blur, absdiff, threshold, dilation, and counting.

    Now supports saving an edited (annotated) video with overlays.

    Returns:
        (lick_timestamps, lick_frames_array, video_duration_seconds, fps)
    """
    import cv2
    import numpy as np
    from pathlib import Path

    if synced_gpio_file is None:
        print("Error: synced_gpio_file is required.")
        return None, None, 0, 0

    x, y, w, h = roi
    if w <= 0 or h <= 0:
        print("Error: ROI has non-positive width/height.")
        return None, None, 0, 0

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return None, None, 0, 0

    # Video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    wait_key_delay = 1 if fps == 0 else max(1, int(1000 / (fps * playback_speed)))

    # Read first frame
    ret, prev_frame = cap.read()
    if not ret:
        print("Error: Could not read the first frame.")
        cap.release()
        return None, None, 0, 0

    # Compute detection threshold in pixels from percentage
    roi_area = w * h
    min_changed_pixels = roi_area * (min_movement_percent / 100.0)

    # Video writer (edited output)
    writer = None
    if save_video_path is not None:
        save_path = Path(save_video_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        # Choose codec if not provided
        if save_codec is None:
            ext = save_path.suffix.lower()
            if ext in (".mp4", ".m4v"):
                fourcc_str = "mp4v"  # widely supported
            elif ext in (".avi",):
                fourcc_str = "XVID"
            else:
                fourcc_str = "mp4v"
        else:
            fourcc_str = save_codec
        out_fps = float(save_fps) if save_fps is not None else (fps if fps > 0 else 30.0)
        writer = cv2.VideoWriter(
            str(save_path),
            cv2.VideoWriter_fourcc(*fourcc_str),
            out_fps,
            (frame_w, frame_h),
        )
        if not writer.isOpened():
            print(f"Warning: Could not open VideoWriter for {save_path}. Video will not be saved.")
            writer = None

    # GPU path via PyTorch if available
    use_torch_cuda = False
    F = None
    try:
        import torch
        import torch.nn.functional as F  # noqa
        use_torch_cuda = torch.cuda.is_available()
    except Exception:
        use_torch_cuda = False

    overlay_needed = show_video_with_licks or (writer is not None)

    if use_torch_cuda:
        device = torch.device("cuda")
        k = 21  # Gaussian kernel size, match your (21,21)
        # OpenCV sigma approximation when sigma=0
        sigma = 0.3 * ((k - 1) * 0.5 - 1) + 0.8

        def _gaussian_kernel_2d(ksize, sigma, device):
            ax = torch.arange(ksize, device=device, dtype=torch.float32) - (ksize // 2)
            g1 = torch.exp(-(ax**2) / (2 * sigma**2))
            g1 = g1 / g1.sum()
            g2 = torch.outer(g1, g1)
            return (g2 / g2.sum()).unsqueeze(0).unsqueeze(0)  # [1,1,k,k]

        gauss = _gaussian_kernel_2d(k, sigma, device)

        def _to_gray_tensor(bgr_roi_np):
            # BGR uint8 -> gray float tensor on GPU [1,1,H,W] in [0,1]
            t = torch.from_numpy(bgr_roi_np).to(device=device, dtype=torch.float32)  # H,W,3
            b, g, r = t[..., 0], t[..., 1], t[..., 2]
            gray = 0.114 * b + 0.587 * g + 0.299 * r
            return (gray / 255.0).unsqueeze(0).unsqueeze(0)

        prev_roi = prev_frame[y:y+h, x:x+w]
        prev_t = _to_gray_tensor(prev_roi)
        prev_blur_t = F.conv2d(prev_t, gauss, padding=k // 2)

        print("GPU acceleration: ON (PyTorch CUDA)")
    else:
        prev_roi_gray = cv2.cvtColor(prev_frame[y:y+h, x:x+w], cv2.COLOR_BGR2GRAY)
        prev_roi_blur = cv2.GaussianBlur(prev_roi_gray, (21, 21), 0)
        print("GPU acceleration: OFF (CPU OpenCV path)")

    lick_timestamps = []
    lick_frames_array = np.zeros(len(synced_gpio_file), dtype=int)
    in_cooldown = 0
    frame_num = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_num >= len(synced_gpio_file):
            break

        current_time = (frame_num / fps) if fps > 0 else 0.0
        current_frame_idx = frame_num
        frame_num += 1

        # We'll draw on this if saving or showing
        vis_frame = frame.copy() if overlay_needed else None

        roi_frame = frame[y:y+h, x:x+w]

        if use_torch_cuda:
            cur_t = _to_gray_tensor(roi_frame)
            cur_blur_t = F.conv2d(cur_t, gauss, padding=k // 2)
            diff_t = (cur_blur_t - prev_blur_t).abs()

            # Threshold in [0,1] scaled back to 0..255 like OpenCV
            mask = (diff_t * 255.0) > threshold

            # Dilation (two iterations)
            dil1 = F.max_pool2d(mask.float(), kernel_size=3, stride=1, padding=1)
            dil2 = F.max_pool2d(dil1, kernel_size=3, stride=1, padding=1)

            changed_pixels = int(dil2.sum().item())
            movement_detected = changed_pixels > min_changed_pixels
            prev_blur_t = cur_blur_t

            if overlay_needed:
                dilated_np = (dil2.squeeze().detach().to("cpu").numpy() * 255).astype(np.uint8)
            else:
                dilated_np = None
        else:
            roi_gray = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY)
            roi_blur = cv2.GaussianBlur(roi_gray, (21, 21), 0)
            frame_diff = cv2.absdiff(prev_roi_blur, roi_blur)
            _, thresh_img = cv2.threshold(frame_diff, threshold, 255, cv2.THRESH_BINARY)
            dilated_np = cv2.dilate(thresh_img, None, iterations=2)
            changed_pixels = cv2.countNonZero(dilated_np)
            movement_detected = changed_pixels > min_changed_pixels
            prev_roi_blur = roi_blur

        # Determine largest contour (for green dot) only if we are drawing
        largest_contour = None
        if overlay_needed and dilated_np is not None:
            contours, _ = cv2.findContours(dilated_np.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)

        # Lick detection with cooldown
        drew_dot = False
        if movement_detected and in_cooldown == 0:
            lick_timestamps.append(current_time)
            lick_frames_array[current_frame_idx] = 1
            in_cooldown = cooldown_frames

            # Draw centroid dot in full-frame coordinates if we have a contour
            if overlay_needed and largest_contour is not None:
                M = cv2.moments(largest_contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    if vis_frame is not None:
                        cv2.circle(vis_frame, (x + cx, y + cy), 3, (0, 255, 0), -1)
                        drew_dot = True
        elif in_cooldown > 0:
            in_cooldown -= 1

        # Overlays (ROI, counters, trial info)
        if overlay_needed and vis_frame is not None:
            cv2.rectangle(vis_frame, (x, y), (x + w, y + h), (255, 0, 0), 2)  # ROI box
            cv2.putText(vis_frame, f"Frame: {frame_num}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(vis_frame, f"Licks: {len(lick_timestamps)}", (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

            # Pull per-frame metadata from synced_gpio_file safely
            try:
                this_state = str(synced_gpio_file['state'].iloc[current_frame_idx])
                this_sync = str(synced_gpio_file['syncpulse'].iloc[current_frame_idx])
                this_trial_type = str(synced_gpio_file['trialtype'].iloc[current_frame_idx])
            except Exception:
                this_state = this_sync = this_trial_type = 'None'

            if this_trial_type != 'None':
                cv2.putText(vis_frame, f"TrialType: {this_trial_type}", (10, 110),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)
            if this_sync != 'None':
                cv2.putText(vis_frame, this_sync, (10, 150),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)
            if this_state != 'None':
                cv2.putText(vis_frame, this_state, (10, 190),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)

            if show_video_with_licks:
                cv2.imshow("Lick Detection Video", vis_frame)
                if cv2.waitKey(wait_key_delay) & 0xFF == ord('q'):
                    break

        # Write edited frame if saving
        if writer is not None:
            writer.write(vis_frame if vis_frame is not None else frame)

    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()

    video_duration_seconds = (total_frames / fps) if fps > 0 else 0.0
    return lick_timestamps, lick_frames_array, video_duration_seconds, fps



# def plot_lick_raster(synced_gpio_file):
#     # no none rows 
#     synced_gpio_file = synced_gpio_file[synced_gpio_file['trialtype']!='None']
#     unique_trial_type = np.sort(synced_gpio_file['trialtype'].unique())

#     all_lick_events_per_trial_type = {} # Dictionary to store organized lick events

#     for tt in unique_trial_type:
#         all_lick_events_per_trial_type[tt] = {}
#         tt_df = synced_gpio_file[synced_gpio_file['trialtype'] == tt].copy() # .copy() to avoid SettingWithCopyWarning
#         unique_trial_start_time = tt_df['trialstarttime'].unique()
        
#         for trial_idx, t_start in enumerate(unique_trial_start_time):
#             single_trial_df = tt_df[tt_df['trialstarttime']==t_start]
#             # Get frame numbers where licks occurred within this trial
#             lick_frames_in_trial = single_trial_df[single_trial_df['lickframe'] == 1].index.values
            
#             if len(lick_frames_in_trial) > 0:
#                 # Get the first frame number of the trial to use as an offset
#                 trial_start_frame = single_trial_df.index.min()
                
#                 # Calculate lick times relative to the start of the trial
#                 relative_lick_times = (lick_frames_in_trial - trial_start_frame) / fps
#                 all_lick_events_per_trial_type[tt][trial_idx] = relative_lick_times.tolist() # Store as list


#     # Plot individual lick events for each trial type 
#     plt.figure(figsize=(20, 8))
#     all_lick_data = [] # To store (relative_lick_time, plot_y_idx, tt) for eventplot for event individual lick
#     y_tick_labels = []
#     y_tick_positions = []

#     color_map = {
#         0: np.array([41, 114, 112]) / 255,
#         1: np.array([230, 109, 80]) / 255,
#         2: np.array([231, 198, 107]) / 255,
#         3: np.array([138, 176, 124]) / 255,
#         4: np.array([41, 157, 143]) / 255
#     }
    

#     event_positions = []
#     event_colors    = []
#     y_tick_labels   = []

#     for tt in unique_trial_type:
#         trials_dict = all_lick_events_per_trial_type[tt]
#         color = color_map.get(tt, "black")
#         for trial_in_type_idx, lick_times in sorted(trials_dict.items()):
#             if not lick_times:
#                 continue
#             event_positions.append(list(lick_times))
#             event_colors.append(color)
#             y_tick_labels.append(f"TT{tt} T{trial_in_type_idx + 1}")

#     # Y tick positions are simply 0..N-1 
#     y_tick_positions = list(range(len(event_positions)))
#     print(len(y_tick_positions))
    
#     # Plot
#     if event_positions:
#         plt.eventplot(event_positions, orientation="horizontal", colors=event_colors)

#     # plt.yticks(y_tick_positions, y_tick_labels)
#     plt.xlabel("Time (s)")
#     plt.ylabel("Trial Index")
#     plt.title(f"{mouse_id}_{date}")

#     odor_start_time = 1.5
#     odor_end_time = 2
#     ymin = -0.5 # Extend across all trials
#     ymax = max(y_tick_positions) - 0.5
#     vline_times = [odor_start_time, odor_end_time, odor_end_time+0.75, odor_end_time+1.5, odor_end_time+3, odor_end_time+6]
#     for t_vline in vline_times:
#         plt.vlines(t_vline, ymin, ymax, color='purple', linestyle='-', linewidth=1, alpha=0.7)
#     plt.axvspan(odor_start_time, odor_end_time, ymin=0, ymax=1, facecolor='purple', alpha=0.1)

#     x_ticks_labels = ['odor', '0.75', '1.5', '3', '6']
#     x_tick_positions = [odor_start_time, odor_end_time+0.75, odor_end_time+1.5, odor_end_time+3, odor_end_time+6]
    
#     plt.xlim([0, 15])

#     plt.xticks(x_tick_positions, x_ticks_labels)
#     plt.grid(True, axis='x', linestyle='--')
#     plt.gca().invert_yaxis() # Display trials from top to bottom

#     fig_root_dir = Path(r"\\140.247.90.110\homes2\Carol\LickData")
#     fig_file_name = f'{mouse_id}_{date}_raster.png'
#     full_fig_path = fig_root_dir / fig_file_name
#     full_fig_path.parent.mkdir(parents=True, exist_ok=True)
#     plt.savefig(full_fig_path)
#     plt.show()


def plot_lick_raster(
    synced_gpio_file,
    fps,
    mouse_id,
    date,
    fig_root_dir=Path(r"\\140.247.90.110\homes2\Carol\LickData")
):
    """
    Plots a raster of lick times per trial type.
    Expects `synced_gpio_file` to already contain a column 'lickframe' of 0/1 flags
    (one per video frame), and columns 'trialtype' and 'trialstarttime'.

    Args:
        synced_gpio_file (pd.DataFrame): frame-aligned metadata with 'lickframe' flags.
        fps (float): Frames Per Second for converting frames to seconds.
        mouse_id (str): Mouse identifier (for plot title & filename).
        date (str): Session date (for plot title & filename).
        fig_root_dir (Path): Directory to save the figure.

    Returns:
        Path: full path to the saved figure.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from pathlib import Path

    df = synced_gpio_file.copy()
    # keep rows with valid trialtype
    df = df[df['trialtype'] != 'None']

    if 'lickframe' not in df.columns:
        raise ValueError("synced_gpio_file must have a 'lickframe' column (0/1 per frame).")

    # sort trial types in a robust way even if they're strings or ints
    unique_trial_types = sorted(df['trialtype'].unique(), key=lambda v: str(v))

    # build a dict: trialtype -> {trial_idx: [relative lick times...]}
    all_lick_events = {}
    for tt in unique_trial_types:
        tt_df = df[df['trialtype'] == tt].copy()
        trial_starts = tt_df['trialstarttime'].unique()
        all_lick_events[str(tt)] = {}
        for trial_idx, t_start in enumerate(trial_starts):
            single_trial = tt_df[tt_df['trialstarttime'] == t_start]
            lick_frames_in_trial = single_trial.index[single_trial['lickframe'] == 1].to_numpy()
            if lick_frames_in_trial.size > 0:
                trial_start_frame = int(single_trial.index.min())
                rel_times = (lick_frames_in_trial - trial_start_frame) / float(fps)
                all_lick_events[str(tt)][trial_idx] = rel_times.tolist()

    # prepare plotting lists
    event_positions = []
    event_colors = []
    y_tick_labels = []

    # color palette (cycled over trial types)
    palette = np.array([
        [41, 114, 112],
        [230, 109, 80],
        [231, 198, 107],
        [138, 176, 124],
        [41, 157, 143],
    ], dtype=float) / 255.0

    for i, tt in enumerate(unique_trial_types):
        trials_for_tt = all_lick_events[str(tt)]
        color = palette[i % len(palette)]
        for trial_in_type_idx, lick_times in sorted(trials_for_tt.items()):
            if lick_times:
                event_positions.append(list(lick_times))
                event_colors.append(color)
                y_tick_labels.append(f"TT{tt} T{trial_in_type_idx + 1}")

    y_positions = list(range(len(event_positions)))

    plt.figure(figsize=(20, 8))
    if event_positions:
        plt.eventplot(event_positions, orientation="horizontal", colors=event_colors)

    plt.xlabel("Time (s)")
    plt.ylabel("Trial Index")
    plt.title(f"{mouse_id}_{date}")

    # same reference timings you used before
    odor_start_time = 1.5
    odor_end_time = 2.0
    ymin = -0.5
    ymax = (max(y_positions) - 0.5) if y_positions else 0.5

    vline_times = [odor_start_time, odor_end_time,
                   odor_end_time + 0.75, odor_end_time + 1.5,
                   odor_end_time + 3, odor_end_time + 6]
    for t_v in vline_times:
        plt.vlines(t_v, ymin, ymax, color='purple', linestyle='-', linewidth=1, alpha=0.7)
    plt.axvspan(odor_start_time, odor_end_time, ymin=0, ymax=1, facecolor='purple', alpha=0.1)

    x_ticks_labels = ['odor', '0.75', '1.5', '3', '6']
    x_tick_positions = [odor_start_time,
                        odor_end_time + 0.75,
                        odor_end_time + 1.5,
                        odor_end_time + 3,
                        odor_end_time + 6]
    plt.xlim([0, 15])
    plt.xticks(x_tick_positions, x_ticks_labels)
    plt.grid(True, axis='x', linestyle='--')
    plt.gca().invert_yaxis()

    fig_root_dir = Path(fig_root_dir)
    fig_root_dir.mkdir(parents=True, exist_ok=True)
    fig_path = fig_root_dir / f'{mouse_id}_{date}_raster.png'
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.show()

    return fig_path


# if __name__ == "__main__":

if __name__ == "__main__":
    import sys  # for sys.exit
    from pathlib import Path
    import pickle

    # protocol = 'OdorWater_VariableDelay_FreeRewards'
    protocol = 'CombinedStimOdorTask'
    print('protocol: ' + protocol)
    mouse_id = "CC" + input("MouseID: CC")
    date = "2025" + input("Date (eg. 0917): 2025")

    # temp cache for long runs
    temp_data_folder = Path(r'C:\Users\carol\Github\BpodProtocols\lickdetection\tmpfile')
    temp_data_folder.mkdir(parents=True, exist_ok=True)
    temp_data_file = temp_data_folder / f'{mouse_id}_{date}_temp.pkl'

    lick_timestamps = None
    video_duration = None
    fps = None
    lick_frames_array = None
    selected_roi = None
    synced_gpio_file = None

    if temp_data_file.exists():
        print(f"Loading temporary data from {temp_data_file}")
        with open(temp_data_file, 'rb') as f:
            temp_data = pickle.load(f)
        lick_timestamps   = temp_data.get('lick_timestamps')
        video_duration    = temp_data.get('video_duration')
        fps               = temp_data.get('fps')
        synced_gpio_file  = temp_data.get('synced_gpio_file')
        selected_roi      = temp_data.get('selected_roi')
        lick_frames_array = temp_data.get('lick_frames_array')  # <-- consistent name
    else:
        root_bpod_dir  = Path(r"\\140.247.90.110\homes2\Carol\BpodData")
        session_dir    = root_bpod_dir / mouse_id / protocol / "Session Data"
        pattern        = f"{mouse_id}_{protocol}_{date}_*.mat"
        matches        = list(session_dir.glob(pattern))
        bpod_data_path = max(matches, key=lambda p: p.stat().st_mtime) if matches else None
        if bpod_data_path is None:
            print("No session file found")
            sys.exit(1)

        root_video_dir = Path(r"\\140.247.90.110\homes2\Carol\VideoData")

        # Safe 'next' with default None
        video_path = next((p for p in root_video_dir.glob(f"{mouse_id}_{date}_*_cam1.avi")), None)
        if video_path is None:
            print("No video found")
            sys.exit(1)

        gpio_file_path = next((p for p in root_video_dir.glob(f"{mouse_id}_{date}_*_gpio1.csv")), None)
        if gpio_file_path is None:
            print("No GPIO file found")
            sys.exit(1)

        # Build synced dataframe
        synced_gpio_file = sync_data.sync_bpod_video(
            gpio_file_path=gpio_file_path,
            bpod_data_path=bpod_data_path
        )

        # ROI (Region of Interest) selection
        enter_roi_manually = input("Enter roi manually? (y/n): ").lower()
        if enter_roi_manually == 'y':
            roi_input = input("Enter ROI as (x, y, w, h): ")
            selected_roi = tuple(map(int, roi_input.strip("()").split(",")))
        else:
            selected_roi = select_roi(str(video_path))

        if not selected_roi:
            print("No ROI selected; exiting.")
            sys.exit(1)

        print(f"Selected ROI: {selected_roi}")

        # Show annotated playback?
        show_video = input("Video playback with lick detection? (y/n): ").lower()
        show_video_with_licks = (show_video == 'y')

        playback_speed = 0.5
        if show_video_with_licks:
            try:
                speed_input = float(input("Enter playback speed: "))
                if speed_input > 0:
                    playback_speed = speed_input
                else:
                    print("Playback speed must be positive. Using default (0.5).")
            except ValueError:
                print("Invalid input for playback speed. Using default (0.5).")

        # Optional: save the annotated video to disk
        out_dir = Path(r"C:\Users\carol\Github\BpodProtocols\lickdetection\out")
        out_dir.mkdir(parents=True, exist_ok=True)
        save_path = out_dir / f"{mouse_id}_{date}_annotated.mp4"

        # Run detection (returns 4-tuple consistently)
        lick_timestamps, lick_frames_array, video_duration, fps = detect_licks(
            str(video_path),
            selected_roi,
            show_video_with_licks=show_video_with_licks,
            playback_speed=playback_speed,
            synced_gpio_file=synced_gpio_file,
            save_video_path=str(save_path),  # comment out if you don't want saving
            save_codec="mp4v",
            save_fps=30.0,
        )

        # Save temp cache unless you were watching the playback (same behavior as before)
        if show_video != 'y':
            with open(temp_data_file, 'wb') as f:
                pickle.dump({
                    'lick_timestamps': lick_timestamps,
                    'video_duration' : video_duration,
                    'fps'            : fps,
                    'synced_gpio_file': synced_gpio_file,
                    'selected_roi'   : selected_roi,
                    'lick_frames_array': lick_frames_array,  # <-- consistent name
                }, f)
            print(f"Data saved to {temp_data_file}")

    # Attach lick flags and plot
    if lick_frames_array is None:
        print("Error: lick_frames_array is missing; cannot plot.")
        sys.exit(1)

    synced_gpio_file = synced_gpio_file.copy()
    synced_gpio_file['lickframe'] = lick_frames_array

    fig_path = plot_lick_raster(synced_gpio_file, fps, mouse_id, date)
    print(f"Saved raster: {fig_path}")

    # protocol = 'OdorWater_VariableDelay_FreeRewards'
    protocol = 'CombinedStimOdorTask'
    print('protocol: ' + protocol)
    mouse_id = "CC" + input("MouseID: CC")
    date = "2025" + input("Date (eg. 0917): 2025")
    ########################################################
    # Temporary file for saving/loading lick detection data
    temp_data_folder = Path(r'C:\Users\carol\Github\BpodProtocols\lickdetection\tmpfile')
    temp_data_folder.mkdir(parents=True, exist_ok=True)
    temp_data_file = temp_data_folder / f'{mouse_id}_{date}_temp.pkl'

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

            # no temp file -> select roi -> detect licks
            lick_timestamps, lick_frames, video_duration, fps = detect_licks(video_path, selected_roi, 
                                                                      show_video_with_licks=show_video_with_licks, 
                                                                      playback_speed=playback_speed, 
                                                                      synced_gpio_file=synced_gpio_file)
            # Save data temporarily
            if show_video != 'y':
                # will not interupt manually
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

    #  either temp file exist or detect licks from scratch -> plot lick raster
    synced_gpio_file['lickframe'] = lick_frames
    plot_lick_raster(synced_gpio_file)