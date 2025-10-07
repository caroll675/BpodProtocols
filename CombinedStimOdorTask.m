function CombinedStimOdorTask

% C. Chen 9/14/2025: Combined StimPatterns_FreeWater_7Patterns_Optotag and OdorWater_VariableDelay  

global BpodSystem

airON = 0;
while ~airON
    answer = questdlg('Is the air ON?', ...
	'Yes','No');
    switch answer
        case 'Yes'
            airON = 1;
        case 'No'
            disp('The task will not start until you turn on the air :)')
            airON = 0;
    end
end

COM_Ports = readtable(fullfile(BpodSystem.DataPath, '..\COM_Ports.txt')); % get COM ports from text file (ignored by git)

%% Setup (runs once before the first trial)

mouse = BpodSystem.Status.CurrentSubjectName;

BpodSystem.Data.TaskDescription = 'CombinedStimOdorTask';

% Task parameters
S = BpodSystem.ProtocolSettings; % contains valve order for this mouse in field OdorValvesOdor

% These parameters are specific to each mouse and loaded from file:
if isempty(fieldnames(S))
    fprintf(['\n******\nWARNING: No saved task parameters found for this mouse.' ...
        '\nGenerating new default parameters.\n******\n']);
    S.NumOdors = input('Number of odors: '); % 4 
    S.OdorValvesOrder = input('Odor valve order (use brackets): ');
    % S.OdorValvesOrder = randperm(S.NumOdors);
    assert(S.NumOdors == numel(S.OdorValvesOrder),'S.NumOdors must match numel(S.OdorValvesOrder)');
    SaveProtocolSettings(S);
end

%% These parameters are shared across animals:
S.Experimenter = 'Carol';
S.Mouse = mouse;

%% Odor parameters
S.TrialStartSignal = 0.25; % seconds - LED as a trial start cue 
S.OdorDelay = 1.25; % seconds - pre odor period after LED before odor presentation
S.OdorDuration = 0.5; % seconds

OdorChunkSize = 22; % trials; chunk size in which to balance trial types
S.NumOdorTrials = OdorChunkSize*5; % 110 trials in total, 10 free reward trials, 100 odor trials

S.RewardDelay = [0.75 1.5 3 6]; % one per odor
S.FracTrials_Odor = [5/OdorChunkSize 5/OdorChunkSize 5/OdorChunkSize 5/OdorChunkSize]; % fraction trials per odor (changed ChunkSize to 22)
S.FracTrials_Free = 1-sum(S.FracTrials_Odor); % fraction free reward trials = 2/22
assert(S.NumOdors == numel(S.RewardDelay),'RewardDelay must have same number of elements as there are odors'); % assert one reward delay per odor
S.RewardAmount = 4; % in uL; same for all odors

S.ITIMean_odor = 18.5;
S.ITIMin_odor = 17;
S.ITIMax_odor = 20;

%% Parameters from StimPatterns_FreeWater_7Pattern.m
S.NumPatterns = 3;

% Num trials
S.NumOptotagTrials1 = 60;
S.NumStimTrials1 = 3*15;
S.NumStimTrials2 = 3*15;
S.NumOptotagTrials2 = 60;

% ITI duration - note different values for optotag trials
S.ITIMean_stim = 12;
S.ITIMin_stim = 8;
S.ITIMax_stim = 20;
S.ForeperiodDuration = 0.5; % ????

S.StimPower_mW = 1; % input('Stim LED power (mW): ');
S.PulseDur = 0.005;

% optotag pulse options:
S.OptotagPulseFreq = 10;
S.OptotagPulseDur = 0.02;
S.OptotagPulseNum = 1;
S.OptotagLightPower_mW = 10;
S.ITIMin_optotag = 1; % different ITI for optotagging trials (uniform distribution)
S.ITIMax_optotag = 3;
% Duration of Optotag state (based on parameters in S)
OptotagStateDuration = ceil(S.OptotagPulseNum/S.OptotagPulseFreq); % seconds

%% display parameters
fprintf('\nSession parameters:\n')
S

%% Set up WavePlayer (Analog Output Module for controlling lasers)
W = BpodWavePlayer(COM_Ports.COM_Port{strcmp(COM_Ports.Module,'BpodWavePlayer')}); % Make sure the COM port is correct
SR = 10000; % Sampling rate for analog output
W.SamplingRate = SR;
W.OutputRange = '0V:5V';

% Stim patterns:

% 1) 2 sec ramping up
waveform_rampUp = [];
freq = 4;
for i = 1:24
    numOnes = S.PulseDur*SR;
    numZeros = round(SR/freq)-numOnes;
    waveform_rampUp = [waveform_rampUp 5*ones(1,numOnes) zeros(1,numZeros)];
    freq = freq*1.135;
end
W.loadWaveform(1,waveform_rampUp);

% 2) 2 sec ramping down
waveform_rampDown = fliplr(waveform_rampUp);
waveform_rampDown = [waveform_rampDown(87:end) zeros(1,86)]; % align to zero
W.loadWaveform(2,waveform_rampDown);


% 3) 3 sec at 20 Hz
waveform_3secSquare_20Hz = zeros(1,round(SR/20));
waveform_3secSquare_20Hz(1:(S.PulseDur * SR)) = 5;
waveform_3secSquare_20Hz = repmat(waveform_3secSquare_20Hz,1,60);
W.loadWaveform(3,waveform_3secSquare_20Hz);

% 4) Optotag message (one 20 ms pulse)
waveform_optotag = zeros(1,round(SR/S.OptotagPulseFreq));
waveform_optotag(1:(S.OptotagPulseDur * SR)) = 5;
waveform_optotag = repmat(waveform_optotag,1,S.OptotagPulseNum);
W.loadWaveform(4,waveform_optotag);

%% LED 
LED_waveform = [ones(1, SR*S.TrialStartSignal) * 5, zeros(1, SR*0.01)]; % 5V for TrialStartSignal duration, then 0V briefly
W.loadWaveform(5, LED_waveform); % Add waveform to channel 3, index 1
channel = 4;

% load waveforms to WavePlayer:
WavePlayerMessages = {};
redStim_idx = 1; % for triggering red LED
blueStim_idx = 2; % for triggering blue laser
for patternIdx = 1:S.NumPatterns
    WavePlayerMessages = [WavePlayerMessages {['P' 2^(redStim_idx-1) patternIdx-1]}]; % send waveform patternIdx to the LED_idx'th channel
end
WavePlayerMessages = [WavePlayerMessages {['P' 2^(blueStim_idx-1) S.NumPatterns]}]; % send optotag message
WavePlayerMessages = [WavePlayerMessages {['P' 2^(channel-1) S.NumPatterns+1]}]; % send LED message   
LoadSerialMessages('WavePlayer1', WavePlayerMessages);

% save waveforms to bpod output structure S (task parameters):
S.stimWaveforms = {};
S.stimWaveforms = {waveform_rampUp,waveform_rampDown,waveform_3secSquare_20Hz};


%% Stim trial types
TargetChunkSize = 3;
ActualChunkSize = S.NumPatterns*round(TargetChunkSize/S.NumPatterns);
TrialTypesChunk = repmat(1:S.NumPatterns,1,ActualChunkSize/S.NumPatterns);

NumChunks1 = ceil(S.NumStimTrials1/ActualChunkSize);
TrialTypes1 = [];
for i = 1:NumChunks1
    perm_idx = randperm(ActualChunkSize);
    TrialTypes1 = [TrialTypes1 TrialTypesChunk(perm_idx)];
end
TrialTypes1 = TrialTypes1(1:S.NumStimTrials1);

NumChunks2 = ceil(S.NumStimTrials2/ActualChunkSize);
TrialTypes2 = [];
for i = 1:NumChunks2
    perm_idx = randperm(ActualChunkSize);
    TrialTypes2 = [TrialTypes2 TrialTypesChunk(perm_idx)];
end
TrialTypes2 = TrialTypes2(1:S.NumStimTrials2);



%% Odor trial types: 0 = free reward; 1 = Odor1, 2 = Odor2, etc; 
% Also define omission trials
NumChunks_Odor = ceil(S.NumOdorTrials/OdorChunkSize); 
TrialTypes_Odor = [];
RewardDelays_Odor = [];
for chunkIdx = 1:NumChunks_Odor
    N_free = round(OdorChunkSize*S.FracTrials_Free); % 1
    tt_this = zeros(1,N_free); % trial types
    delay_this = zeros(1,N_free); % not sure about whether unexpected reward should have delays
    for odorIdx = 1:S.NumOdors
        N_odor = round(OdorChunkSize*S.FracTrials_Odor(odorIdx)); % 5
        tt_this = [tt_this odorIdx*ones(1,N_odor)];
        delay_this = [delay_this S.RewardDelay(odorIdx)*ones(1,N_odor)];
    end
    
    max_consec = Inf;
    while tt_this(1)==0 || max_consec > 3
        shuf_idx = randperm(OdorChunkSize);
        tt_this = tt_this(shuf_idx);
        delay_this = delay_this(shuf_idx);
        max_consec = max(diff([0 find(diff(tt_this)~=0) OdorChunkSize]));
    end

    
    TrialTypes_Odor = [TrialTypes_Odor tt_this];
    RewardDelays_Odor = [RewardDelays_Odor delay_this];
    
end


%% Pokes plot
state_colors = struct( ...
    'TrialStartSignal', [39 71 83]/255, ...  % black
    'OdorDelay',        [0.8 0.8 0.8], ...  % gray
    'CS1',              [230 109 80]/255, ...  % red
    'CS2',              [231 198 107]/255, ...  % yellow
    'CS3',              [138 176 124]/255, ...  % tender green
    'CS4',              [41 157 143]/255, ...  % teal
    'RewardDelay',      [0.85 0.85 0.85], ...  % gray
    'Reward',           [41 114 112]/255, ...  % dark green
    'ITI',              [0.92 0.92 0.92]);     % gray

OdorWaterTrialVisualizer('init', state_colors); % only plot available states
% PokesPlotLicksSlow('init', state_colors, []);

%% Set odors for each trial type in each mouse
% S.OdorValvesOrder is the order of odors for this mouse, 
% loaded in the line S = BpodSystem.ProtocolSettings;
ValveMessages = {['B' 0]}; % Valve 1 is blank
for i = 1:S.NumOdors
    ValveMessages = [ValveMessages {['B' 2^S.OdorValvesOrder(i)+1]}];
end
LoadSerialMessages('ValveModule1', ValveMessages);  % Set serial messages for valve module. Valve 1 is the default that is normally on

%% Start Protocol
RedLampOn = 0;
while ~RedLampOn
    answer = questdlg('Is the Red Lamp ON?', ...
	'Yes','No');
    switch answer
        case 'Yes'
            RedLampOn = 1;
        case 'No'
            disp('Please turn on red lamp')
            RedLampOn = 0;
    end
end

total_trial_ctr = 0;

%% Optotag1

fprintf('\nOptotag1 (%d trials)\n', S.NumOptotagTrials1);
for currentTrial = 1:S.NumOptotagTrials1
    
    total_trial_ctr = total_trial_ctr+1;
    
    % Calculate ITI for this trial
    ITIDuration = unifrnd(S.ITIMin_optotag,S.ITIMax_optotag);

    fprintf('\tTrial %d:\tITI=%0.1fs\n',currentTrial,ITIDuration);

    %--- Assemble state machine
    sma = NewStateMatrix();
    
    sma = AddState(sma,'Name','Foreperiod',...
        'Timer',S.ForeperiodDuration,...
        'StateChangeConditions',{'Tup','Optotag'},...
        'OutputActions',{'BNC1',1});
    sma = AddState(sma, 'Name', 'Optotag', ... 
        'Timer', OptotagStateDuration,...
        'StateChangeConditions', {'Tup', 'ITI'},...
        'OutputActions', {'WavePlayer1', S.NumPatterns+1, 'BNC1', 0}); 
    sma = AddState(sma, 'Name', 'ITI', ... 
        'Timer', ITIDuration,...
        'StateChangeConditions', {'Tup','exit'},...
        'OutputActions', {'BNC1',0});

    % Add the odor states so that pokes plot doesn't get messed up:
    for tt = 1:S.NumPatterns
        sma = AddState(sma, 'Name', sprintf('Stim%d',tt),'Timer', 0,'StateChangeConditions',{},'OutputActions', {}); 
    end
    
    SendStateMatrix(sma); % Send state machine to the Bpod state machine device

    RawEvents = RunStateMatrix; % Run the trial and return events

    % Update online plots
    if ~isempty(fieldnames(RawEvents))
        
        BpodSystem.Data = AddTrialEvents(BpodSystem.Data, RawEvents);
        BpodSystem.Data.TrialSettings(total_trial_ctr) = S;

        SaveBpodSessionData;

    end
    
    %--- This final block of code is necessary for the Bpod console's pause and stop buttons to work
    HandlePauseCondition; % Checks to see if the protocol is paused. If so, waits until user resumes.
    if BpodSystem.Status.BeingUsed == 0
        return
    end
end

%% StimTrials1
fprintf('\nStim trials1 (%d trials)\n', S.NumStimTrials1);
for currentTrial = 1:S.NumStimTrials1
    
    total_trial_ctr = total_trial_ctr+1;
    
    TrialType = TrialTypes1(currentTrial); 
    
    % Compute variables for this trial's state machine:
   
    Stim_state = sprintf('Stim%d',TrialType);

    % Which laser pattern to trigger
    LaserMessage = TrialType;
    
    % Calculate ITI for this trial
    ITIDuration = exprnd(S.ITIMean_stim-S.ITIMin_stim) + S.ITIMin_stim;
    if ITIDuration > S.ITIMax_stim
        ITIDuration = S.ITIMax_stim;
    end
    
    
    % Display trial type
    fprintf('\tTrial %d:\tTrialType%d\tITI=%0.1fs\n',...
        currentTrial,TrialType,ITIDuration);
    
    
    % Create state matrix
    sma = NewStateMatrix();
    sma = AddState(sma, 'Name', 'Foreperiod',...
        'Timer', S.ForeperiodDuration,...
        'StateChangeConditions', {'Tup', Stim_state},...
        'OutputActions', {'BNC1', 1}); % BNC1 for sync pulse
    for tt = 1:S.NumPatterns
        sma = AddState(sma, 'Name', sprintf('Stim%d',tt),...
            'Timer', 2,... % Assuming 2 seconds for stim duration, adjust if necessary
            'StateChangeConditions', {'Tup', 'ITI'},...
            'OutputActions', {'WavePlayer1', LaserMessage, 'BNC1', 0}); 
    end
    sma = AddState(sma, 'Name', 'ITI',...
        'Timer', ITIDuration,...
        'StateChangeConditions', {'Tup', 'exit'},...
        'OutputActions', {'BNC1', 0});
    sma = AddState(sma, 'Name', 'Optotag', ... 
        'Timer', OptotagStateDuration,...
        'StateChangeConditions', {'Tup', 'ITI'},...
        'OutputActions', {'WavePlayer1', S.NumPatterns+1, 'BNC1', 0}); 

    % Send state machine to Bpod device
    SendStateMatrix(sma);
    
    % Run the trial and return events
    RawEvents = RunStateMatrix;
    
    if ~isempty(fieldnames(RawEvents))
        % Save trial data
        BpodSystem.Data = AddTrialEvents(BpodSystem.Data, RawEvents);
        BpodSystem.Data.TrialSettings(total_trial_ctr) = S;
        BpodSystem.Data.TrialTypes1(currentTrial) = TrialType;
        SaveBpodSessionData;
        
    end

    % Handle pauses and exit if the user ended the session
    HandlePauseCondition;
    if BpodSystem.Status.BeingUsed == 0
        ModuleWrite('ValveModule1', ['B' 0]); % make sure the odor valves are closed
        return
    end
    
end

fprintf('Stim trials1 finished\n');
toc;

pause(10);

%% Turn off red lamps
RedLampOff = 0;
while ~RedLampOff
    answer = questdlg('Is the Red Lamp OFF?', ...
	'Yes','No');
    switch answer
        case 'Yes'
            RedLampOff = 1;
        case 'No'
            disp('Please turn off red lamp')
            RedLampOff = 0;
    end
end

pause(10);

%% Odor trials
tic
AccumulatedReward = 0;
for currentTrial = 1:S.NumOdorTrials
    total_trial_ctr = total_trial_ctr+1;
    TrialType = TrialTypes_Odor(currentTrial);
    RewardValveTime = GetValveTimes(S.RewardAmount, 1);
    AccumulatedReward = AccumulatedReward+S.RewardAmount;

    if TrialType==0
        CS_state = 'Free Reward';
    else
        CS_state = sprintf('CS%d',TrialType);
    end

    ValveMessage = TrialType+1;
    
    % Calculate ITI for this trial
    ITIDuration = (S.ITIMax_odor - S.ITIMin_odor) * rand() + S.ITIMin_odor;
    
    % Display trial type
    if TrialType==0
        fprintf('\tTrial %d: TrialType=%d Free TotalReward=%d ITI=%0.1fs\n',...
            currentTrial,TrialType,AccumulatedReward,ITIDuration);
    else
        fprintf('\tTrial %d: TrialType=%d Odor=%d TotalReward=%d ITI=%0.1fs\n',...
            currentTrial,TrialType,S.OdorValvesOrder(TrialType),AccumulatedReward,ITIDuration);
    end
    
    % Create state matrix
    sma = NewStateMatrix();
    if TrialType==0 % (reward -> ITI)
        sma = AddState(sma, 'Name', 'Reward',...
            'Timer', RewardValveTime,...
            'StateChangeConditions', {'Tup', 'ITI'},...
            'OutputActions', {'ValveState',1,'BNC1',1}); 
    else % (LED -> odor delay -> odor -> reward delay -> reward -> ITI)
        sma = AddState(sma, 'Name', 'TrialStartSignal',...
            'Timer', S.TrialStartSignal,...
            'StateChangeConditions', {'Tup', 'OdorDelay'},...
            'OutputActions', {'WavePlayer1', S.NumTrials+2, 'BNC1', 1}); 
        sma = AddState(sma, 'Name', 'OdorDelay',...
            'Timer', S.OdorDelay,...
            'StateChangeConditions', {'Tup', CS_state},...
                'OutputActions', {'BNC1', 0}); 
        for tt = 1:S.NumOdors % plotting purpose
            sma = AddState(sma, 'Name', sprintf('CS%d',tt),...
                'Timer', S.OdorDuration,...
                'StateChangeConditions', {'Tup', 'RewardDelay'},...
                'OutputActions', {'ValveModule1', ValveMessage,'BNC1', 0}); 
        end 
        sma = AddState(sma, 'Name', 'RewardDelay',...
            'Timer', RewardDelays_Odor(currentTrial),...
            'StateChangeConditions', {'Tup', 'Reward'},...
            'OutputActions', {'ValveModule1', 1,'BNC1', 0}); 
        sma = AddState(sma, 'Name', 'Reward',...
            'Timer', RewardValveTime,...
            'StateChangeConditions', {'Tup', 'ITI'},...
            'OutputActions', {'ValveState',1,'BNC1', 0});
    end  
    sma = AddState(sma, 'Name', 'ITI',...
        'Timer', ITIDuration,...
        'StateChangeConditions', {'Tup', 'exit'},...
        'OutputActions', {'BNC1', 0});

    % Send state machine to Bpod device
    SendStateMatrix(sma);
    
    % Run the trial and return events
    RawEvents = RunStateMatrix;
    
    if ~isempty(fieldnames(RawEvents))
        % Save trial data
        BpodSystem.Data = AddTrialEvents(BpodSystem.Data, RawEvents);
        BpodSystem.Data.TrialSettings(total_trial_ctr) = S;
        BpodSystem.Data.TrialTypes(total_trial_ctr) = TrialType;
        BpodSystem.Data.AccumulatedReward(total_trial_ctr) = AccumulatedReward; 
        if TrialType==0
            BpodSystem.Data.OdorID(total_trial_ctr) = nan;
        else
            BpodSystem.Data.OdorID(total_trial_ctr) = S.OdorValvesOrder(TrialType);
        end
        SaveBpodSessionData;
        
        % Update online plots
        OdorWaterTrialVisualizer('update', state_colors);
        % PokesPlotLicksSlow('update');
    end

    % Handle pauses and exit if the user ended the session
    HandlePauseCondition;
    if BpodSystem.Status.BeingUsed == 0
        fprintf('Protocol stopped by user at trial %d\n', currentTrial);
        ModuleWrite('ValveModule1', ['B' 0]); % make sure the odor valves are closed
        return
    end
end
toc;

fprintf('\nOdor trials finished\n');

pause(10);

%% Turn on red lamps
RedLampOn = 0;
while ~RedLampOn
    answer = questdlg('Is the Red Lamp ON?', ...
	'Yes','No');
    switch answer
        case 'Yes'
            RedLampOn = 1;
        case 'No'
            disp('Please turn on red lamp')
            RedLampOn = 0;
    end
end

pause(10);

%% StimTrials2
fprintf('\nStim trials2 (%d trials)\n', S.NumStimTrials2);
for currentTrial = 1:S.NumStimTrials2
    
    total_trial_ctr = total_trial_ctr+1;
    
    TrialType = TrialTypes2(currentTrial);
    
    % Compute variables for this trial's state machine:
   
    Stim_state = sprintf('Stim%d',TrialType);

    % Which laser pattern to trigger
    LaserMessage = TrialType;
    
    % Calculate ITI for this trial
    ITIDuration = exprnd(S.ITIMean_stim-S.ITIMin_stim) + S.ITIMin_stim;
    if ITIDuration > S.ITIMax_stim
        ITIDuration = S.ITIMax_stim;
    end
    
    
    % Display trial type
    fprintf('\tTrial %d:\tTrialType%d\tITI=%0.1fs\n',...
        currentTrial,TrialType,ITIDuration);
    
    
    % Create state matrix
    sma = NewStateMatrix();
    sma = AddState(sma, 'Name', 'Foreperiod',...
        'Timer', S.ForeperiodDuration,...
        'StateChangeConditions', {'Tup', Stim_state},...
        'OutputActions', {'BNC1', 1}); 
    for tt = 1:S.NumPatterns
        sma = AddState(sma, 'Name', sprintf('Stim%d',tt),...
            'Timer', 2,... % Assuming 2 seconds for stim duration, adjust if necessary
            'StateChangeConditions', {'Tup', 'ITI'},...
            'OutputActions', {'WavePlayer1', LaserMessage, 'BNC1', 0}); 
    end
    sma = AddState(sma, 'Name', 'ITI',...
        'Timer', ITIDuration,...
        'StateChangeConditions', {'Tup', 'exit'},...
        'OutputActions', {'BNC1', 0});
    sma = AddState(sma, 'Name', 'Optotag', ... 
        'Timer', OptotagStateDuration,...
        'StateChangeConditions', {'Tup', 'ITI'},...
        'OutputActions', {'WavePlayer1', LaserMessage, 'BNC1', 0}); 

    % Add reward state so pokes plot doesn't get messed up:
    sma = AddState(sma,'Name','Reward','Timer',0,'StateChangeConditions',{},'OutputActions',{}); 
    
    % Send state machine to Bpod device
    SendStateMatrix(sma);
    
    % Run the trial and return events
    RawEvents = RunStateMatrix;
    
    if ~isempty(fieldnames(RawEvents))
        % Save trial data
        BpodSystem.Data = AddTrialEvents(BpodSystem.Data, RawEvents);
        BpodSystem.Data.TrialSettings(total_trial_ctr) = S;
        BpodSystem.Data.TrialTypes2(currentTrial) = TrialType;
        SaveBpodSessionData;
        
    end

    % Handle pauses and exit if the user ended the session
    HandlePauseCondition;
    if BpodSystem.Status.BeingUsed == 0
        ModuleWrite('ValveModule1', ['B' 0]); % make sure the odor valves are closed
        return
    end
    
end

fprintf('Stim trials2 finished\n');
toc;

pause(10);

%% Optotag2

fprintf('\nOptotag2 (%d trials)\n', S.NumOptotagTrials2);
for currentTrial = 1:S.NumOptotagTrials2
    
    total_trial_ctr = total_trial_ctr+1;
    
    % Calculate ITI for this trial
    ITIDuration = unifrnd(S.ITIMin_optotag,S.ITIMax_optotag);

    fprintf('\tTrial %d:\tITI=%0.1fs\n',currentTrial,ITIDuration);

    %--- Assemble state machine
    sma = NewStateMatrix();
    
    sma = AddState(sma,'Name','Foreperiod',...
        'Timer',S.ForeperiodDuration,...
        'StateChangeConditions',{'Tup','Optotag'},...
        'OutputActions',{'BNC1',1});
    sma = AddState(sma, 'Name', 'Optotag', ... 
        'Timer', OptotagStateDuration,...
        'StateChangeConditions', {'Tup', 'ITI'},...
        'OutputActions', {'WavePlayer1', S.NumTrials+1, 'BNC1', 0}); 
    sma = AddState(sma, 'Name', 'ITI', ... 
        'Timer', ITIDuration,...
        'StateChangeConditions', {'Tup','exit'},...
        'OutputActions', {'BNC1',0});

    % Add the odor states so that pokes plot doesn't get messed up:
    for tt = 1:S.NumPatterns
        sma = AddState(sma, 'Name', sprintf('Stim%d',tt),'Timer', 0,'StateChangeConditions',{},'OutputActions', {}); 
    end
    
    SendStateMatrix(sma); % Send state machine to the Bpod state machine device

    RawEvents = RunStateMatrix; % Run the trial and return events

    % Update online plots
    if ~isempty(fieldnames(RawEvents))
        
        BpodSystem.Data = AddTrialEvents(BpodSystem.Data, RawEvents);
        BpodSystem.Data.TrialSettings(total_trial_ctr) = S;

        SaveBpodSessionData;

    end
    
    %--- This final block of code is necessary for the Bpod console's pause and stop buttons to work
    HandlePauseCondition; % Checks to see if the protocol is paused. If so, waits until user resumes.
    if BpodSystem.Status.BeingUsed == 0
        return
    end
end

fprintf('Optotag2 finished\n');
toc;


%% Turn off red lamps
RedLampOff = 0;
while ~RedLampOff
    answer = questdlg('Is the Red Lamp OFF?', ...
	'Yes','No');
    switch answer
        case 'Yes'
            RedLampOff = 1;
        case 'No'
            disp('Please turn on red lamp')
            RedLampOff = 0;
    end
end

clear W;

fprintf('\nProtocol finished\n')

end