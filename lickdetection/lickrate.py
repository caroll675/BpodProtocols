# calculate the lickrates based on the pickle file generated from the lickdetection_raster.py 
# count the number of licks very 0.2 sec 
# move the 0.2 sec window forward before the first reward frame and move the window backward after the first reward frame 
# so the lickrate before and after the reward is not contaminated by the reward information 
# align the data on the trial start time stamp 
# plot the lickrates for differnet trialtypes (1 2 3 4) of variable reward delays, ignore trial type 0 in this script (no trial start cue, no reward delays)


# compute_lickrates.py
# Calculates 0.2 s lick rates aligned to trial start from the pickle produced by lickdetection_raster.py.
# Windows: forward [t, t+bin) before first Reward; backward (t-bin, t] after first Reward.
# Output: ONE PNG with all trial types (1..4). No CSV.

import pickle
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# helper functions

def first_reward_time_in_trial(trial_df: pd.DataFrame, fps: float) -> float:
    """Return first Reward time (s) relative to this trial's start, or np.nan if none."""
    if trial_df.empty:
        return np.nan
    start_frame = int(trial_df.index.min())
    reward_frames = trial_df.index[trial_df['state'] == 'Reward']
    if len(reward_frames) == 0:
        return np.nan
    first_reward_frame = int(reward_frames.min())
    return (first_reward_frame - start_frame) / float(fps)

def sliding_count_prepost(lick_times_s: np.ndarray, t_grid: np.ndarray, bin_size: float, reward_time: float):
    """
    Count licks per grid time t using:
      - BEFORE reward:  [t, t+bin)  (backward window), count from reward time 
      - AFTER  reward:  (t-bin, t]  (forward window), count from reward time
    Returns counts and rates (licks/s).
    """
    lt = np.asarray(lick_times_s, dtype=float)
    counts = np.zeros_like(t_grid, dtype=float)

    if np.isnan(reward_time):
        print('missing reward info')

    for i, t in enumerate(t_grid):
        if i+1 < len(t_grid):
            left, right = t, t+bin_size
            if right < reward_time:
                counts[i+1] = np.count_nonzero((lt > left) & (lt <= right))
            else:
                counts[i+1] = np.count_nonzero((lt > left) & (lt <= right))
    return counts, counts / bin_size

def main(mouse_id, date):
    bin_size = 0.2 # sec
    t_max = 15 # sec 
    pkl_path = Path(r'C:\Users\carol\Github\BpodProtocols\lickdetection\tmpfile') / f"{mouse_id}_{date}_temp.pkl"
    if not pkl_path.exists():
        raise FileNotFoundError(f"Pickle file not found: {pkl_path}")

    with open(pkl_path, "rb") as f:
        blob = pickle.load(f)

    # Pull pieces we need
    synced = blob['synced_gpio_file'].copy()
    lick_frames_array = np.asarray(blob['lick_frames_array']) # binary 
    fps = float(blob['fps'])

    synced['lickframe'] = lick_frames_array

    # Trial type as numeric; ignore 0 / None; ignore frames with nothing
    synced['trialtype_num'] = pd.to_numeric(synced['trialtype'], errors='coerce')
    df = synced[synced['trialtype_num'].isin([1, 2, 3, 4])].copy()

    # Time grid (aligned to trial start)
    t_grid = np.arange(0.0, t_max + 1e-12, bin_size)

    # Collect per-trial lick-rate time courses by trial type
    results = {}  # tt -> dict('rates': [n_trials x n_bins], 'reward_times': [n_trials])

    for tt in [1, 2, 3, 4]:
        tt_df = df[df['trialtype_num'] == tt]
        if tt_df.empty:
            print('missing trial type' + str(tt))
            continue

        trial_ids = pd.unique(tt_df['trialstarttime'])
        trial_rates = []
        reward_times = []

        for tstart in trial_ids:
            tri = tt_df[tt_df['trialstarttime'] == tstart]
            if tri.empty:
                continue

            start_frame = int(tri.index.min())
            lick_idx = tri.index[tri['lickframe'] == 1].to_numpy()
            lick_times_s = (lick_idx - start_frame) / fps

            rtime = first_reward_time_in_trial(tri, fps)
            reward_times.append(rtime)

            _, rates = sliding_count_prepost(lick_times_s, t_grid, bin_size, rtime)
            trial_rates.append(rates)

        if len(trial_rates) == 0:
            continue

        trial_rates = np.vstack(trial_rates)  # [n_trials, n_bins]
        results[tt] = {
            "rates": trial_rates,
            "reward_times": np.array(reward_times, dtype=float)
        }

    # Plot: all trial types on one figure
    out_dir = Path(r'C:\Users\carol\Github\BpodProtocols\lickdetection\fig')
    out_dir.mkdir(parents=False, exist_ok=True)
    plt.figure(figsize=(25, 6))

    colors = {
        1: (171/255, 92/255, 154/255),
        2: (138/255, 110/255, 168/255),
        3: (136/255, 172/255, 215/255),
        4: (142/255, 206/255, 216/255),
    }

    unique_rt = []
    for tt in [1, 2, 3, 4]:
        if tt not in results:
            continue
        rates = results[tt]["rates"]                 # [n_trials, n_bins]
        mean_rate = np.nanmean(rates, axis=0)
        sem_rate  = np.nanstd(rates, axis=0, ddof=1) / np.sqrt(rates.shape[0])
        c = colors.get(tt, None)
        plt.plot(t_grid, mean_rate, label='Odor ' + str(tt), linewidth=2, color=c)
        # mark the dots 
        plt.plot(t_grid, mean_rate, marker='o', color=c)
        # Shaded uncertainty band
        plt.fill_between(t_grid, mean_rate - sem_rate, mean_rate + sem_rate, alpha=0.25, color=c)

        # show reward vlines
        rt = results[tt]["reward_times"]  
        unique_rt.append(rt[0])
        plt.axvline(rt[0], linestyle="--", alpha=0.8, color=c)

    print(unique_rt)
    plt.title(f"{mouse_id}_{date}")
    plt.axvline(1.5, linestyle="--", alpha=0.8, color='k')
    x_ticks_labels = ['0', '1.25', '2', '3.5', '6.5']
    x_tick_positions = [1.5, unique_rt[0], unique_rt[1], unique_rt[2], unique_rt[3]]
    plt.xticks(x_tick_positions, x_ticks_labels)
    plt.xlabel("Time from odor onset (s)")
    plt.ylabel("Lick rate (licks/s)")
    plt.xlim(0, t_max)
    plt.legend()
    fig_path = out_dir / f"{mouse_id}_{date}.svg"
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved: {fig_path}")

if __name__ == "__main__":
    mouse_id = "CC" + input("MouseID: CC")
    date = "2025" + input("Date (eg. 0917): 2025")
    main(mouse_id, date)
