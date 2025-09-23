import scipy.io as sio
import numpy as np
import pandas as pd

gpio_file_path = '/Users/carol/Downloads/CC4/CC4_20250922_133604_gpio1.csv'
parse_gpio_data = pd.read_csv(gpio_file_path, header=None)
parse_gpio_data.columns = ['pulse1', 'pulse2', 'timestamp']
# load the sync pulse start and end in 24 hour clock system
sync_pulse_start_frame = parse_gpio_data[(parse_gpio_data['pulse1'].diff() > 0)].index + 1
sync_pulse_end_frame = parse_gpio_data[(parse_gpio_data['pulse1'].diff() < 0)].index
sync_pulse_start_timestamp = parse_gpio_data.loc[sync_pulse_start_frame, 'timestamp']
sync_pulse_end_timestamp =  parse_gpio_data.loc[sync_pulse_end_frame, 'timestamp']
parse_gpio_data['syncpulse'] = np.where(parse_gpio_data['pulse1'] > 0, 'syncpulse', None)


bpod_data_path = r'/Users/carol/Downloads/CC4/CC4_OdorWater_VariableDelay_FreeRewards_20250922_130442.mat'

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

except AttributeError as e:
    print(f"Error accessing field: {e}. Check the exact structure of your .mat file.")


TrialTypes = SessionData.TrialTypes
TrialStartTimestamp = SessionData.TrialStartTimestamp # in sec

fps = 30
trial_types = np.asarray(SessionData.TrialTypes)
state = np.full(len(parse_gpio_data), 'None', dtype=object)

to_frames = lambda sec: int(round(sec * fps))

def paint(start, seconds, label):
    s = int(start)
    e = min(s + to_frames(seconds), len(state))
    if e > s:
        state[s:e] = label
    return e  # next start

# If you only need the start frames, drop end_frame entirely
for i, start_frame in enumerate(sync_pulse_start_frame):
    s = int(start_frame)
    if trial_types[i] == 0:
        s = paint(s, 1, 'Reward')  # fixed 1s reward
    else:
        s = paint(s, LED, 'LED')
        s = paint(s, OdorDelay, 'OdorDelay')
        s = paint(s, Odor, 'Odor')
        s = paint(s, RewardDelay[trial_types[i]-1], 'RewardDelay')
        s = paint(s, 1, 'Reward')

parse_gpio_data['state'] = state

# add trial type
