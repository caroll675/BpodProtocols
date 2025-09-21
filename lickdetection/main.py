import lickdetector

if __name__ == "__main__":
    video_path = input("Enter the path to the mouse licking video: ")

    selected_roi = lickdetector.select_roi(video_path)

    if selected_roi:
        print(f"Selected ROI: {selected_roi}")
        lick_timestamps, lick_rate, video_duration = lickdetector.detect_licks(video_path, selected_roi)
        
        if lick_timestamps is not None:
            print(f"Detected lick rate: {lick_rate:.2f} licks per minute")
            print(f"Total licks: {len(lick_timestamps)}")
            print(f"Lick timestamps (seconds): {lick_timestamps}")
            lickdetector.plot_licks(lick_timestamps, video_duration)
    else:
        print("No ROI selected or an error occurred during ROI selection.")
