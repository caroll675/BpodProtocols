function OdorWater_VariableProbability_FreeRewards

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


ChunkSize = 29; % trials; chunk size in which to balance trial types
NumTrials = ChunkSize*3; % 84 odor trials and 3 free reward trials



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
S.Experimenter = 'Malcolm';
S.Mouse = mouse;
S.ForeperiodDuration = 0.5; % seconds - green computer screen as a trial start cue 
S.PreOdorDelayDuration = 1.25; % seconds - delay after green screen until odor presentation
S.OdorDuration = 0.5; % seconds

S.RewardDelay = [0.4 1.5 3 6]; % one per odor
S.FracTrials_Odor = [7/ChunkSize 7/ChunkSize 7/ChunkSize 7/ChunkSize]; % fraction trials per odor
S.FracTrials_Free = 1-sum(S.FracTrials_Odor); % fraction free reward trials = 1/29
assert(S.NumOdors == numel(S.RewardDelay),'RewardDelay must have same number of elements as there are odors'); % assert one reward delay per odor
S.RewardAmount = 4; % in uL; same for all odors

S.ITIMean = 12;
S.ITIMin = 8;
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
        N_odor = round(ChunkSize*S.FracTrials_Odor(odorIdx)); % 7 
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
    'Foreperiod',[.9,.9,.9],...
    'Reward',[0 1 0],...
    'CS1', [0 1 1],...
    'CS2', [0 0 1],...
    'CS3', [1 0 1],...
    'CS4', [1 1 0],...
    'RewardDelay', [.6 .6 .6],...
    'ITI', [.9,.9,.9]);
PokesPlotLicksSlow('init', state_colors, []);


%% Set odors for each trial type in each mouse
% S.OdorValvesOrder is the order of odors for this mouse, 
% loaded in the line S = BpodSystem.ProtocolSettings;
ValveMessages = {['B' 0]}; % Valve 1 is blank
for i = 1:S.NumOdors
    ValveMessages = [ValveMessages {['B' 2^S.OdorValvesOrder(i)+1]}];
end
LoadSerialMessages('ValveModule1', ValveMessages);  % Set serial messages for valve module. Valve 1 is the default that is normally on


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
    ITIDuration = exprnd(S.ITIMean-S.ITIMin) + S.ITIMin;
    if ITIDuration > S.ITIMax
        ITIDuration = S.ITIMax;
    end
    
    % Display trial type
    if TrialType==0
        fprintf('\tTrial %d:\tTrialType %d\tFree\tTotalReward=%d ITI=%0.1fs\n',...
            currentTrial,TrialType,AccumulatedReward,ITIDuration);
    else
        fprintf('\tTrial %d:\tTrialType %d\tOdor%d\tTotalReward=%d ITI=%0.1fs\n',...
            currentTrial,TrialType,S.OdorValvesOrder(TrialType),AccumulatedReward,ITIDuration);
    end
    
    % Create state matrix
    sma = NewStateMatrix();
    sma = AddState(sma, 'Name', 'Foreperiod',...
        'Timer', S.ForeperiodDuration,...
        'StateChangeConditions', {'Tup', PreOdorDelay},...
        'OutputActions', {'BNC1', 1, 'BNC2', 1}); % how to connect to visual cue?
    if TrialType == 0
        sma = AddState(sma, 'Name', 'PreOdorDelay',...
        'Timer', S.PreOdorDelayDuration,...
        'StateChangeConditions', {'Tup', 'Reward'},...
            'OutputActions', {'BNC1', 0, 'BNC2', 0}); 
    else
        sma = AddState(sma, 'Name', 'PreOdorDelay',...
        'Timer', S.PreOdorDelayDuration,...
        'StateChangeConditions', {'Tup', CS_state},...
            'OutputActions', {'BNC1', 0, 'BNC2', 0}); 
    end
    sma = AddState(sma, 'Name', CS_state,...
        'Timer', S.OdorDuration,...
        'StateChangeConditions', {'Tup', 'RewardDelay'},...
        'OutputActions', {'ValveModule1', ValveMessage,... % closes the blank valve, opens the odor valve
            'BNC1', 0, 'BNC2', 0}); 
        
    sma = AddState(sma, 'Name', 'RewardDelay',...
        'Timer', S.RewardDelay(TrialType),...
        'StateChangeConditions', {'Tup', 'Reward'},...
        'OutputActions', {'ValveModule1', 1,... % opens the blank valve, closes the odor valve
            'BNC1', 0, 'BNC2', 0}); 
    sma = AddState(sma, 'Name', 'Reward',...
        'Timer', RewardValveTime,...
        'StateChangeConditions', {'Tup', 'ITI'},...
        'OutputActions', {'ValveState',1,'BNC1', 0, 'BNC2', 0});
    sma = AddState(sma, 'Name', 'ITI',...
        'Timer', ITIDuration,...
        'StateChangeConditions', {'Tup', 'exit'},...
        'OutputActions', {'BNC1', 0, 'BNC2', 0});

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
        PokesPlotLicksSlow('update');
    end

    % Handle pauses and exit if the user ended the session
    HandlePauseCondition;
    if BpodSystem.Status.BeingUsed == 0
        ModuleWrite('ValveModule1', ['B' 0]); % make sure the odor valves are closed
        return
    end
    
end
toc;

fprintf('\nProtocol finished\n')

end