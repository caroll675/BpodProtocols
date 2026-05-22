function CombinedStimOdorTask_rewardHistoryITI

    % C. Chen 9/14/2025: Combined StimPatterns_FreeWater_7Patterns_Optotag and OdorWater_VariableDelay  

    % C. Chen 12/11/2025: Changed delay duration for odor trials and number of free reward trials
    
    % C. Chen 2/8/2026: use red led to optotag so bluestim = red optotag

    % C. Chen 5/22/2026: stablize reward rate, shorten average ITI and increase
    % chunksize from 33 to 49 (more trials per odor)

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
    
    BpodSystem.Data.TaskDescription = 'CombinedStimOdorTask_rewardHistoryITI';
    
    % Task parameters
    S = BpodSystem.ProtocolSettings; % contains valve order for this mouse in field OdorValvesOdor
    
    % These parameters are specific to each mouse and loaded from file:
    if isempty(fieldnames(S))
        fprintf(['\n******\nWARNING: No saved task parameters found for this mouse.' ...
            '\nGenerating new default parameters.\n******\n']);
        S.NumOdors = input('Number of odors: '); % 4 
        S.OdorValvesOrder = input('Odor valve order (use brackets): ');
        assert(S.NumOdors == numel(S.OdorValvesOrder),'S.NumOdors must match numel(S.OdorValvesOrder)');
        SaveProtocolSettings(S);
    end
    
    %% These parameters are shared across animals:
    S.Experimenter = 'Carol';
    S.Mouse = mouse;
    S.ProtocolName = 'CombinedStimOdorTask_rewardHistoryITI';
    
    %% Odor parameters
    S.ForeperiodDuration = 0.5; % seconds
    S.TrialStartSignal = 0.25; % seconds - LED as a trial start cue 
    S.OdorDelay = 1; % seconds - pre odor period after LED before odor presentation
    S.OdorDuration = 0.5; % seconds

    OdorChunkSize = 49; % trials; chunk size in which to balance trial types
    S.NumOdorTrials = OdorChunkSize*3; % 147 trials in total, 3 free reward trials, 144 odor trials
    
    S.RewardDelay = [0 1 2.5 5.5]; % one per odor
    S.FracTrials_Odor = [12/OdorChunkSize 12/OdorChunkSize 12/OdorChunkSize 12/OdorChunkSize];
    S.FracTrials_Free = 1-sum(S.FracTrials_Odor); % fraction free reward trials = 1/49

    assert(S.NumOdors == numel(S.RewardDelay),'RewardDelay must have same number of elements as there are odors'); % assert one reward delay per odor
    S.RewardAmount = 4; % in uL; same for all odors
    

    S.TargetRewardInterval_odor = 14.7; % seconds/reward; 147 trials is about 36 min
    S.ITIMean_odor = 10.2;              % baseline ITI before reward-rate correction
    S.ITIJitter_odor = 2;               % random +/- jitter
    S.ITIMin_odor = 5;
    S.ITIMax_odor = 13;
    S.ITIRateGain_odor = 0.2;           % how strongly recent reward rate adjusts ITI
    S.ITIMaxCorrection_odor = 2;        % max +/- seconds added by reward-rate correction

    
    %% Parameters from StimPatterns_FreeWater_7Pattern.m
    S.NumPatterns = 3;
    S.StimDurations = [2 2 3]; % seconds; must match loaded stim waveforms
    
    % Num trials
%     S.NumOptotagTrials1 = 60;
%     S.NumStimTrials1 = 3*15;
%     S.NumStimTrials2 = 3*15;
%     S.NumOptotagTrials2 = 60;
    
    S.NumOptotagTrials1 = 0;
    S.NumStimTrials1 = 0;
    S.NumStimTrials2 = 0;
    S.NumOptotagTrials2 = 0;
    
    % ITI duration - note different values for optotag trials
    S.ITIMean_stim = 12;
    S.ITIMin_stim = 8;
    S.ITIMax_stim = 20;
   
    
    S.StimPower_mW = 1; % input('Stim LED power (mW): ');