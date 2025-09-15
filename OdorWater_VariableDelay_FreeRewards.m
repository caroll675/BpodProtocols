function OdorWater_VariableDelay_FreeRewards

% M. Campbell 8/2/2021: Protocol to deliver odors followed by laser pulses.
% M. Campbell 12/1/2021: Edited OdorLaser to create OdorLaserWater task.
% M. Campbell 10/24/2022: Edited OdorLaserWater to create OdorLaser_v2:
%   Initially used for photometry opto calibration experiments 
%   (calibrating opto stim to water reward, then repeating OdorLaser task)
%   Added omission trials
%   Made the ITI exponentially distributed
%   Added block of rewards before and after odor trials (for
%   calibration/comparison to odor/opto responses)
% M. Campbell 11/5/2022: Added unpredicted opto stim before and after odor
%   trials
% M. Campbell 1/3/2023: Changed to just deliver Odors followed by Water
% M. Campbell 7/17/2024: Added unpredicted water trials
% C. Chen 9/4/2025: Changed to OdorWater_VariableDelay

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

COM_Ports = readtable('..\COM_Ports.txt'); % get COM ports from text file (ignored by git)


%% Setup (runs once before the first trial)

mouse = BpodSystem.Status.CurrentSubjectName;


ChunkSize = 22; % trials; chunk size in which to balance trial types
NumTrials = ChunkSize*5; % 110 trials in total, 10 free reward trials, 100 odor trials



BpodSystem.Data.TaskDescription = 'OdorWater_VariableDelay';

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

% These parameters are shared across animals:
S.Experimenter = 'Carol';
S.Mouse = mouse;
S.TrialStartSignal = 0.25; % seconds - LED as a trial start cue 
S.OdorDelay = 1.25; % seconds - pre odor period after LED before odor presentation
S.OdorDuration = 0.5; % seconds

S.RewardDelay = [0.75 1.5 3 6]; % one per odor
S.FracTrials_Odor = [5/ChunkSize 5/ChunkSize 5/ChunkSize 5/ChunkSize]; % fraction trials per odor
S.FracTrials_Free = 1-sum(S.FracTrials_Odor); % fraction free reward trials = 2/22
assert(S.NumOdors == numel(S.RewardDelay),'RewardDelay must have same number of elements as there are odors'); % assert one reward delay per odor
S.RewardAmount = 4; % in uL; same for all odors

S.ITIMean = 18.5;
S.ITIMin = 17;
S.ITIMax = 20;

% display parameters
fprintf('\nSession parameters:\n')
S
fprintf('NumTrials = %d\n',NumTrials);


%% Define trial types: 0 = free reward; 1 = Odor1, 2 = Odor2, etc; 
% Also define omission trials
NumChunks = ceil(NumTrials/ChunkSize); 
TrialTypes = [];
RewardDelays = [];
for chunkIdx = 1:NumChunks
    N_free = round(ChunkSize*S.FracTrials_Free); % 1
    tt_this = zeros(1,N_free); % trial types
    delay_this = zeros(1,N_free); % not sure about whether unexpected reward should have delays
    for odorIdx = 1:S.NumOdors
        N_odor = round(ChunkSize*S.FracTrials_Odor(odorIdx)); % 5
        tt_this = [tt_this odorIdx*ones(1,N_odor)];
        delay_this = [delay_this S.RewardDelay(odorIdx)*ones(1,N_odor)];
    end
    
    max_consec = Inf;
    while tt_this(1)==0 || max_consec > 3
        shuf_idx = randperm(ChunkSize);
        tt_this = tt_this(shuf_idx);
        delay_this = delay_this(shuf_idx);
        max_consec = max(diff([0 find(diff(tt_this)~=0) ChunkSize]));
    end

    
    TrialTypes = [TrialTypes tt_this];
    RewardDelays = [RewardDelays delay_this];
    
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

OdorWaterTrialVisualizer('init', state_colors);
% PokesPlotLicksSlow('init', state_colors, []);

%% Set odors for each trial type in each mouse
% S.OdorValvesOrder is the order of odors for this mouse, 
% loaded in the line S = BpodSystem.ProtocolSettings;
ValveMessages = {['B' 0]}; % Valve 1 is blank
for i = 1:S.NumOdors
    ValveMessages = [ValveMessages {['B' 2^S.OdorValvesOrder(i)+1]}];
end
LoadSerialMessages('ValveModule1', ValveMessages);  % Set serial messages for valve module. Valve 1 is the default that is normally on

%% Set up WavePlayer (Analog Output Module for controlling LED)
W = BpodWavePlayer(COM_Ports.COM_Port{strcmp(COM_Ports.Module,'BpodWavePlayer')}); % Make sure the COM port is correct
SR = 10000; % Sampling rate for analog output
W.SamplingRate = SR;
W.OutputRange = '0V:5V'; 
LED_waveform = [ones(1, SR*S.TrialStartSignal) * 5, zeros(1, SR*0.01)]; % 5V for TrialStartSignal duration, then 0V briefly
W.addWaveform(3, LED_waveform); % Add waveform to channel 3, index 1
WavePlayerMessages = {['P' 3 1]}; % Serial message to play waveform 1 from channel 3
LoadSerialMessages('WavePlayer3', WavePlayerMessages);

%% Odor trials
tic
AccumulatedReward = 0;
for currentTrial = 1:NumTrials
    TrialType = TrialTypes(currentTrial);
    RewardValveTime = GetValveTimes(S.RewardAmount, 1);
    AccumulatedReward = AccumulatedReward+S.RewardAmount;

    if TrialType==0
        CS_state = 'Free Reward';
    else
        CS_state = sprintf('CS%d',TrialType);
    end


    % Serial message to open/close odor valves
    ValveMessage = TrialType+1;
    
    % Calculate ITI for this trial
    ITIDuration = (S.ITIMax - S.ITIMin) * rand() + S.ITIMin;
    
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
            'OutputActions', {'ValveState',1, ... % opens the blank valve, closes the odor valve
            'BNC1', 1});
    else % (LED -> odor delay -> odor -> reward delay -> reward -> ITI)
        sma = AddState(sma, 'Name', 'TrialStartSignal',...
            'Timer', S.TrialStartSignal,...
            'StateChangeConditions', {'Tup', 'OdorDelay'},...
            'OutputActions', {'WavePlayer3', 1, 'BNC1', 1}); % turn on LED, channel 3, first waveform
        sma = AddState(sma, 'Name', 'OdorDelay',...
            'Timer', S.OdorDelay,...
            'StateChangeConditions', {'Tup', CS_state},...
                'OutputActions', {'BNC1', 0}); 
        for tt = 1:S.NumOdors % plotting purpose
            sma = AddState(sma, 'Name', sprintf('CS%d',tt),...
                'Timer', S.OdorDuration,...
                'StateChangeConditions', {'Tup', 'RewardDelay'},...
                'OutputActions', {'ValveModule1', ValveMessage,... % closes the blank valve, opens the odor valve
                    'BNC1', 0}); 
        end 
        sma = AddState(sma, 'Name', 'RewardDelay',...
            'Timer', S.RewardDelay(TrialType),...
            'StateChangeConditions', {'Tup', 'Reward'},...
            'OutputActions', {'ValveModule1', 1,... % opens the blank valve, closes the odor valve
                'BNC1', 0}); 
        sma = AddState(sma, 'Name', 'Reward',...
            'Timer', RewardValveTime,...
            'StateChangeConditions', {'Tup', 'ITI'},...
            'OutputActions', {'ValveState', 1, 'BNC1', 0});
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
        BpodSystem.Data.TrialSettings(currentTrial) = S;
        BpodSystem.Data.TrialTypes(currentTrial) = TrialType;
        BpodSystem.Data.AccumulatedReward(currentTrial) = AccumulatedReward; 
        if TrialType==0
            BpodSystem.Data.OdorID(currentTrial) = nan;
        else
            BpodSystem.Data.OdorID(currentTrial) = S.OdorValvesOrder(TrialType);
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

fprintf('\nProtocol finished\n');

end