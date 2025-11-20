import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
import pickle
import sys



def generate_lick_rate_figure(mouse_id, date, synced_gpio_file, lick_frames, fps, selected_roi):
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


if __name__ == "__main__":
    # path for all pickle files in C:\Users\carol\Github\BpodProtocols\lickdetection
    pickle_files = list(Path(r"C:\Users\carol\Github\BpodProtocols\lickdetection").glob("*.pkl"))
    pickle_files.sort()

    for temp_data_file in pickle_files:
        if Path(temp_data_file).exists():
            print(f"Loading data from {temp_data_file}")
            with open(temp_data_file, 'rb') as f:
                temp_data = pickle.load(f)
                lick_timestamps = temp_data.get('lick_timestamps')
                video_duration = temp_data.get('video_duration')
                fps = temp_data.get('fps')
                synced_gpio_file = temp_data.get('synced_gpio_file')
                selected_roi = temp_data.get('selected_roi')
                lick_frames = temp_data.get('lick_frames_array') 

            # Extract mouse_id and date from the filename
            filename = Path(temp_data_file).stem
            parts = filename.split('_')
            if len(parts) >= 2:
                mouse_id = parts[0]
                date = parts[1]
            else:
                mouse_id = "UnknownMouse"
                date = "UnknownDate"
            
            if synced_gpio_file is not None and lick_frames is not None and fps is not None and selected_roi is not None:
                generate_lick_rate_figure(mouse_id, date, synced_gpio_file, lick_frames, fps, selected_roi)
                print(f"Figure generated for {temp_data_file}")
            else:
                print(f"Skipping {temp_data_file} due to missing data.")
        else:
            print(f"Error: Pickle file not found: {temp_data_file}")