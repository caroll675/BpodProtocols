import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import argparse
import re
import time

def parse_time_to_seconds(time_str):
    """
    Parse time string in format 'MM:SS' or 'HH:MM:SS' to total seconds.
    
    Examples:
        '16:35' -> 995 seconds (16 minutes 35 seconds)
        '1:05:23' -> 3923 seconds (1 hour 5 minutes 23 seconds)
    """
    # Remove any whitespace
    time_str = time_str.strip()
    
    # Split by colon
    parts = time_str.split(':')
    
    if len(parts) == 2:
        # Format: MM:SS
        minutes, seconds = map(int, parts)
        return minutes * 60 + seconds
    elif len(parts) == 3:
        # Format: HH:MM:SS
        hours, minutes, seconds = map(int, parts)
        return hours * 3600 + minutes * 60 + seconds
    else:
        raise ValueError(f"Invalid time format: {time_str}. Use 'MM:SS' or 'HH:MM:SS'")

def check_gpu_availability():
    """Check if GPU/CUDA is available for video processing."""
    try:
        # Check if CUDA is available
        if cv2.cuda.getCudaEnabledDeviceCount() > 0:
            print(f"GPU detected: {cv2.cuda.getCudaEnabledDeviceCount()} CUDA device(s) available")
            return True
        else:
            print("No CUDA devices found")
            return False
    except Exception as e:
        print(f"Error checking GPU availability: {e}")
        return False

def setup_gpu_video_capture(video_path, use_gpu=True):
    """Setup video capture with GPU acceleration if available."""
    if use_gpu and check_gpu_availability():
        try:
            # Try to use GPU-accelerated video capture
            cap = cv2.VideoCapture(str(video_path), cv2.CAP_CUDA)
            if cap.isOpened():
                print("Using GPU-accelerated video capture")
                return cap, True
        except Exception as e:
            print(f"GPU video capture failed: {e}")
    
    # Fallback to CPU
    cap = cv2.VideoCapture(str(video_path))
    print("Using CPU video capture")
    return cap, False

def setup_gpu_video_writer(output_path, fps, width, height, use_gpu=True):
    """Setup video writer with GPU acceleration if available."""
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    
    if use_gpu and check_gpu_availability():
        try:
            # Try GPU-accelerated video writer
            out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height), cv2.CAP_CUDA)
            if out.isOpened():
                print("Using GPU-accelerated video writer")
                return out, True
        except Exception as e:
            print(f"GPU video writer failed: {e}")
    
    # Fallback to CPU
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    print("Using CPU video writer")
    return out, False

def main():
    parser = argparse.ArgumentParser(description='Trim video from start time to end time with GPU acceleration')
    parser.add_argument('--video', '-v', type=str, help='Path to input video file')
    parser.add_argument('--gpio', '-g', type=str, help='Path to input GPIO file')
    parser.add_argument('--start', '-s', type=str, help='Start time (MM:SS or HH:MM:SS)')
    parser.add_argument('--end', '-e', type=str, help='End time (MM:SS or HH:MM:SS)')
    parser.add_argument('--gpu', action='store_true', help='Force GPU acceleration (if available)')
    parser.add_argument('--cpu', action='store_true', help='Force CPU processing (disable GPU)')
    parser.add_argument('--benchmark', action='store_true', help='Show processing speed benchmark')
    
    args = parser.parse_args()
    
    # Use command line arguments or default values
    video_path_str = r"\\140.247.90.110\homes2\Carol\VideoData\CC4_20251007_114439_cam1.avi"
    gpio_file_str = r"\\140.247.90.110\homes2\Carol\VideoData\CC4_20251007_114439_gpio1.csv"
    
    # Parse start and end times
    start_time_str = "15:08"
    end_time_str = "1:21:32"
    
    # Determine GPU usage
    use_gpu = True
    if args.cpu:
        use_gpu = False
        print("GPU disabled by user request")
    elif args.gpu:
        use_gpu = True
        print("GPU acceleration requested")
    
    try:
        start_seconds = parse_time_to_seconds(start_time_str)
        end_seconds = parse_time_to_seconds(end_time_str)
    except ValueError as e:
        print(f"Error parsing time: {e}")
        return

    # Convert to Path objects
    video_path = Path(video_path_str)
    gpio_file = Path(gpio_file_str)

    # Define output file paths
    output_video_path = video_path.parent / f"{video_path.stem}_trimmed_{start_time_str.replace(':', '')}to{end_time_str.replace(':', '')}.avi"
    output_gpio_file = gpio_file.parent / f"{gpio_file.stem}_trimmed_{start_time_str.replace(':', '')}to{end_time_str.replace(':', '')}.csv"

    print(f"Trimming video from {start_time_str} to {end_time_str}")
    print(f"Start: {start_seconds} seconds, End: {end_seconds} seconds")
    
    # Start timing for benchmark
    start_time_processing = time.time()
    
    # --- Video Trimming ---
    cap, gpu_capture = setup_gpu_video_capture(video_path, use_gpu)

    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    total_duration = total_frames / fps

    print(f"Video info: {fps:.2f} FPS, {total_duration/60:.2f} minutes total duration")

    # Calculate frame numbers for start and end times
    start_frame = int(start_seconds * fps)
    end_frame = int(end_seconds * fps)
    
    # Ensure end frame doesn't exceed video length
    end_frame = min(end_frame, total_frames)
    
    # Ensure start frame is valid
    start_frame = max(0, start_frame)
    
    if start_frame >= end_frame:
        print("Error: Start time must be before end time")
        return

    # Calculate number of frames to process
    frames_to_process = end_frame - start_frame
    
    print(f"Processing frames {start_frame} to {end_frame} ({frames_to_process} frames)")
    print(f"Duration: {(end_seconds - start_seconds)/60:.2f} minutes")

    # Set video position to start frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    # Setup GPU-accelerated video writer
    out, gpu_writer = setup_gpu_video_writer(output_video_path, fps, width, height, use_gpu)

    if not out.isOpened():
        print(f"Error: Could not open video writer for {output_video_path}")
        return
    
    frames_written = 0
    frame_start_time = time.time()
    
    # GPU-accelerated frame processing if available
    if gpu_writer and check_gpu_availability():
        print("Using GPU-accelerated frame processing")
        gpu_frame = cv2.cuda_GpuMat()
    
    while frames_written < frames_to_process:
        ret, frame = cap.read()
        if not ret:
            break
            
        # GPU-accelerated processing if available
        if gpu_writer and check_gpu_availability():
            try:
                # Upload frame to GPU
                gpu_frame.upload(frame)
                # Download processed frame back to CPU for writing
                processed_frame = gpu_frame.download()
                out.write(processed_frame)
            except Exception as e:
                # Fallback to CPU if GPU processing fails
                out.write(frame)
        else:
            out.write(frame)
            
        frames_written += 1
        
        # Print progress every 5 minutes
        if frames_written % (fps * 60 * 5) == 0:
            elapsed_minutes = frames_written / (fps * 60)
            current_time = time.time()
            elapsed_seconds = current_time - frame_start_time
            fps_actual = frames_written / elapsed_seconds if elapsed_seconds > 0 else 0
            print(f"Processed {elapsed_minutes:.0f} minutes of trimmed video. Processing speed: {fps_actual:.1f} FPS")

    actual_duration = frames_written / fps
    processing_time = time.time() - start_time_processing
    
    print(f"Video trimmed to {actual_duration/60:.2f} minutes and saved to {output_video_path}")
    
    # Benchmark information
    if args.benchmark:
        print(f"\n=== BENCHMARK RESULTS ===")
        print(f"Processing method: {'GPU-accelerated' if (gpu_capture or gpu_writer) else 'CPU'}")
        print(f"Total processing time: {processing_time:.2f} seconds")
        print(f"Average processing speed: {frames_written/processing_time:.1f} FPS")
        print(f"Real-time factor: {(frames_written/processing_time)/fps:.2f}x")
        print(f"Video duration: {actual_duration:.2f} seconds")
        print(f"Processing efficiency: {actual_duration/processing_time:.2f}x real-time")
    
    cap.release()
    out.release()
    cv2.destroyAllWindows()

    # --- GPIO File Trimming ---
    try:
        gpio_df = pd.read_csv(gpio_file)
        
        # Trim GPIO data to match video time range
        # GPIO data should be synchronized with video frames
        trimmed_gpio_df = gpio_df.iloc[start_frame:end_frame]
        trimmed_gpio_df.to_csv(output_gpio_file, index=False)
        print(f"GPIO file trimmed to {len(trimmed_gpio_df)} frames (from frame {start_frame} to {end_frame}) and saved to {output_gpio_file}")
    except FileNotFoundError:
        print(f"Error: Could not find GPIO file {gpio_file}")
    except Exception as e:
        print(f"An error occurred while trimming GPIO file: {e}")

if __name__ == "__main__":
    main()