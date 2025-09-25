import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

# Input file paths
video_path_str = r"\\140.247.90.110\homes2\Carol\VideoData\CC4_20250918_124923_cam1.avi"
gpio_file_str = r"\\140.247.90.110\homes2\Carol\VideoData\CC4_20250918_124923_gpio1.csv"

# Convert to Path objects
video_path = Path(video_path_str)
gpio_file = Path(gpio_file_str)

# Define output file paths
output_video_path = video_path.parent / f"{video_path.stem}_trimmed.avi"
output_gpio_file = gpio_file.parent / f"{gpio_file.stem}_trimmed.csv"

# --- Video Trimming ---
cap = cv2.VideoCapture(str(video_path))

if not cap.isOpened():
    print(f"Error: Could not open video file {video_path}")
else:
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Calculate target frames for 50 minutes
    trim_duration_minutes = 50
    target_frames_count = int(fps * trim_duration_minutes * 60)

    fourcc = cv2.VideoWriter_fourcc(*'MJPG') # Or 'XVID', 'MP4V', depending on installed codecs
    out = cv2.VideoWriter(str(output_video_path), fourcc, fps, (width, height))

    if not out.isOpened():
        print(f"Error: Could not open video writer for {output_video_path}")
    else:
        frame_num = 0
        while frame_num < target_frames_count:
            ret, frame = cap.read()
            if not ret:
                break
            out.write(frame)
            frame_num += 1
            if frame_num % (fps * 60 * 5) == 0: # Print progress every 5 minutes
                print(f"Processed {frame_num / (fps * 60):.0f} minutes of video.")

        print(f"Video trimmed to {frame_num / (fps * 60):.2f} minutes and saved to {output_video_path}")
    
    cap.release()
    out.release()
    cv2.destroyAllWindows()

# --- GPIO File Trimming ---
try:
    gpio_df = pd.read_csv(gpio_file)
    trimmed_gpio_df = gpio_df.iloc[:target_frames_count]
    trimmed_gpio_df.to_csv(output_gpio_file, index=False)
    print(f"GPIO file trimmed to {target_frames_count} frames and saved to {output_gpio_file}")
except FileNotFoundError:
    print(f"Error: Could not find GPIO file {gpio_file}")
except Exception as e:
    print(f"An error occurred while trimming GPIO file: {e}")