import scipy.io as sio
import numpy as np
import pandas as pd

def sync_bpod_video(gpio_file_path, bpod_data_path):
    """
    Returns:
        csv file: synced data
    """
    parse_gpio_data = pd.read_csv(gpio_file_path, header=None)
    parse_gpio_data.columns = ['pulse1', 'pulse2', 'timestamp']
    # load the sync pulse start and end in 24 hour clock system
    sync_pulse_start_frame = parse_gpio_data[(parse_gpio_data['pulse1'].diff() > 0)].index + 1
    sync_pulse_end_frame = parse_gpio_data[(parse_gpio_data['pulse1'].diff() < 0)].index
    # one time I forgot to restart the video after I restarted the protocol, so I only need the last 110 sync pulse
    sync_pulse_start_frame = sync_pulse_start_frame[-110:]
    sync_pulse_end_frame = sync_pulse_end_frame[-110:]
    sync_pulse_start_timestamp = parse_gpio_data.loc[sync_pulse_start_frame, 'timestamp']
    sync_pulse_end_timestamp =  parse_gpio_data.loc[sync_pulse_end_frame, 'timestamp']
    parse_gpio_data['syncpulse'] = np.where(parse_gpio_data['pulse1'] > 0, 'syncpulse', 'None')

    # Load the .mat file. Use the optional 'struct_as_record=False' to load structs as objects,
    # which can sometimes simplify access.
    # squeeze_me=True can help flatten single-element arrays automatically.
    bpod_data = sio.loadmat(bpod_data_path, squeeze_me=True, struct_as_record=False)

    # Access the top-level 'SessionData' struct.
    # It often loads as a single-element object array.
    SessionData = bpod_data['SessionData']
    field_names = [attr for attr in dir(SessionData) if not attr.startswith('__')]
    # print(f"SessionData contains the following fields: {field_names}")

    # Access nested fields. The exact access depends on the structure.
    # With squeeze_me=True, you often don't need [0, 0].
    try:
        TrialSettings = SessionData.TrialSettings[0]
        trialSettings_field_names = [attr for attr in dir(TrialSettings) if not attr.startswith('__')]
        # print(f"TrialSettings contains the following fields: {trialSettings_field_names}")
        LED = TrialSettings.TrialStartSignal
        OdorDelay = TrialSettings.OdorDelay
        Odor = TrialSettings.OdorDuration
        RewardDelay = TrialSettings.RewardDelay
        RewardValveTime = 0.0883

    except AttributeError as e:
        print(f"Error accessing field: {e}. Check the exact structure of your .mat file.")


    TrialTypes = SessionData.TrialTypes
    TrialStartTimestamp = SessionData.TrialStartTimestamp # in sec
    TrialEndTimestamp = SessionData.TrialEndTimestamp
    TrialDuration = TrialEndTimestamp - TrialStartTimestamp

    fps = 30
    state = np.full(len(parse_gpio_data), 'None', dtype=object)
    trial_type_col = np.full(len(parse_gpio_data), 'None', dtype=object)
    trial_start_time_col = np.full(len(parse_gpio_data), 'None', dtype=object)

    to_frames = lambda sec: int(round(sec * fps))

    def paint(start, seconds, label):
        s = int(start)
        e = min(s + to_frames(seconds), len(state))
        if e > s:
            state[s:e] = label
        return e  # next start


    for i, start_frame in enumerate(sync_pulse_start_frame):
        tt = TrialTypes[i]
        s = int(start_frame)

        trial_start_frame = s
        trial_duration_sec = TrialEndTimestamp[i] - TrialStartTimestamp[i]
        trial_end_frame = trial_start_frame + int(round(trial_duration_sec*fps))
        trial_type_col[trial_start_frame:trial_end_frame] = tt
        trial_start_time_col[trial_start_frame:trial_end_frame] = TrialStartTimestamp[i]

        if tt == 0:
            s = paint(s,  RewardValveTime, 'Reward')  
            s = paint(s, trial_duration_sec - RewardValveTime, 'ITI')
        else:
            s = paint(s, LED, 'LED')
            s = paint(s, OdorDelay, 'OdorDelay')
            s = paint(s, Odor, 'Odor')
            s = paint(s, RewardDelay[TrialTypes[i]-1], 'RewardDelay')
            s = paint(s, RewardValveTime, 'Reward')
            tasktime = LED + OdorDelay + RewardDelay[TrialTypes[i]-1] + RewardValveTime
            s = paint(s, trial_duration_sec - tasktime , 'ITI')


    parse_gpio_data['state'] = state
    parse_gpio_data['trialtype'] = trial_type_col
    parse_gpio_data['trialstarttime'] = trial_start_time_col
    parse_gpio_data.to_csv('tmp.csv')
    return parse_gpio_data