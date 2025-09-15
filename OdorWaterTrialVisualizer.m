function OdorWaterTrialVisualizer(varargin)

% Customized trial visualizer for OdorWater_VariableDelay protocol
% Handles variable state structures across different trial types
% Allows time-based x-axis specification
% Edited by C. Chen on 09/2025 based on PokesPlotLicksSlow

global BpodSystem
    
action = varargin{1};
state_colors = varargin{2};
switch action

    %% init     
    case 'init' 

        % Initialize handles 
        BpodSystem.GUIHandles.OdorWaterPlot.LicksHandle = [];
        
        % Create figure
        BpodSystem.ProtocolFigures.OdorWaterPlot = figure('Position', [10 40 400 700],'name','OdorWater Trial Visualizer','numbertitle','off', 'MenuBar', 'none', 'Resize', 'on');

        BpodSystem.GUIHandles.OdorWaterPlot.StateColors = state_colors;
        
        % Alignment control
        BpodSystem.GUIHandles.OdorWaterPlot.AlignOnLabel = uicontrol('Style', 'text','String','align on:', 'Position', [30 70 60 20], 'FontWeight', 'normal', 'FontSize', 10,'FontName', 'Arial');
        BpodSystem.GUIHandles.OdorWaterPlot.AlignOnMenu = uicontrol('Style', 'popupmenu','Value',1, 'String', fields(state_colors), 'Position', [95 70 150 20], 'FontWeight', 'normal', 'FontSize', 10, 'BackgroundColor','white', 'FontName', 'Arial','Callback', {@OdorWaterTrialVisualizer, 'alignon'});
        
        % Time window controls
        BpodSystem.GUIHandles.OdorWaterPlot.LeftEdgeTopLabel = uicontrol('Style', 'text','String','start (s):', 'Position', [30 35 60 20], 'FontWeight', 'normal', 'FontSize', 10,'FontName', 'Arial');
        BpodSystem.GUIHandles.OdorWaterPlot.LeftEdgeTop = uicontrol('Style', 'edit','String',-2, 'Position', [95 35 40 20], 'FontWeight', 'normal', 'FontSize', 10, 'BackgroundColor','white', 'FontName', 'Arial','Callback', {@OdorWaterTrialVisualizer, 'time_axis'});
        
        BpodSystem.GUIHandles.OdorWaterPlot.LeftEdgeBottomLabel = uicontrol('Style', 'text','String','end (s):', 'Position', [30 10 60 20], 'FontWeight', 'normal', 'FontSize', 10, 'FontName', 'Arial');
        BpodSystem.GUIHandles.OdorWaterPlot.LeftEdgeBottom = uicontrol('Style', 'edit','String',25, 'Position', [95 10 40 20], 'FontWeight', 'normal', 'FontSize', 10, 'BackgroundColor','white', 'FontName', 'Arial','Callback', {@OdorWaterTrialVisualizer, 'time_axis'});
         
        % Number of trials to display
        BpodSystem.GUIHandles.OdorWaterPlot.LastnLabel = uicontrol('Style', 'text','String','N trials:', 'Position', [150 33 50 20], 'FontWeight', 'normal', 'FontSize', 10, 'FontName', 'Arial');
        BpodSystem.GUIHandles.OdorWaterPlot.Lastn = uicontrol('Style', 'edit','String', 15, 'Position', [205 35 40 20], 'FontWeight', 'normal', 'FontSize', 10, 'BackgroundColor','white', 'FontName', 'Arial','Callback', {@OdorWaterTrialVisualizer, 'time_axis'});
        
        % Trial type filter
        BpodSystem.GUIHandles.OdorWaterPlot.TrialTypeLabel = uicontrol('Style', 'text','String','Show:', 'Position', [260 33 40 20], 'FontWeight', 'normal', 'FontSize', 10, 'FontName', 'Arial');
        BpodSystem.GUIHandles.OdorWaterPlot.TrialTypeMenu = uicontrol('Style', 'popupmenu','Value',1, 'String', {'All', 'Free Reward', 'Odor Trials'}, 'Position', [305 35 80 20], 'FontWeight', 'normal', 'FontSize', 10, 'BackgroundColor','white', 'FontName', 'Arial','Callback', {@OdorWaterTrialVisualizer, 'trial_filter'});
        
        % Axis for plot
        BpodSystem.GUIHandles.OdorWaterPlot.PokesPlotAxis = axes('Position', [0.1 0.38 0.8 0.54],'Color', 0.3*[1 1 1]);
        xlabel('Time (s)');
        ylabel('Trial Number');

        % Initialize state handles
        fnames = fieldnames(state_colors);
        for j = 1:str2double(get(BpodSystem.GUIHandles.OdorWaterPlot.Lastn, 'String'))  % trials
            for i = 1:length(fnames) % states
                BpodSystem.GUIHandles.OdorWaterPlot.StateHandle(j).(fnames{i}) = fill([(i-1) (i-1)+2 (i-1) (i-1)],[(j-1) (j-1) (j-1)+1 (j-1)+1],state_colors.(fnames{i}),'EdgeColor','none');
                set(BpodSystem.GUIHandles.OdorWaterPlot.StateHandle(j).(fnames{i}),'Visible','off');
                hold on;
            end
        end
        
        % Set initial axis limits for the plot
        set(BpodSystem.GUIHandles.OdorWaterPlot.PokesPlotAxis, 'XLim', ...
            [str2double(get(BpodSystem.GUIHandles.OdorWaterPlot.LeftEdgeTop,'String')), ...
            str2double(get(BpodSystem.GUIHandles.OdorWaterPlot.LeftEdgeBottom,'String'))]);
        set(BpodSystem.GUIHandles.OdorWaterPlot.PokesPlotAxis,'YLim', [0 str2double(get(BpodSystem.GUIHandles.OdorWaterPlot.Lastn, 'String'))]);
        
        % Color legend axis
        BpodSystem.GUIHandles.OdorWaterPlot.ColorAxis = axes('Position', [0.15 0.29 0.7 0.03]);
         
        % Plot reference colors
        for i = 1:length(fnames)
            fill([i-0.9 i-0.9 i-0.1 i-0.1], [0 1 1 0], state_colors.(fnames{i}),'EdgeColor','none');
            if length(fnames{i}) < 12
                legend = fnames{i};
            else
                legend = fnames{i}(1:12);
            end
            hold on;
            t = text(i-0.5, -0.5, legend);
            set(t, 'Interpreter', 'none', 'HorizontalAlignment', 'right', 'VerticalAlignment', 'middle', 'Rotation', 90);
            set(gca, 'Visible', 'off');
        end
        ylim([0 1]); xlim([0 length(fnames)]);

    %% update    
    case 'update'

        % Check if figure exists, if not try to re-initialize
        if ~isfield(BpodSystem.ProtocolFigures, 'OdorWaterPlot') || ...
           ~isvalid(BpodSystem.ProtocolFigures.OdorWaterPlot)
            fprintf('OdorWaterTrialVisualizer: Figure not found. Attempting to re-initialize...\n');
            % Try to re-initialize with default colors
            OdorWaterTrialVisualizer('init', state_colors);
            return;
        end

        figure(BpodSystem.ProtocolFigures.OdorWaterPlot);
        axes(BpodSystem.GUIHandles.OdorWaterPlot.PokesPlotAxis);
    

        % Check if GUI handles exist
        if ~isfield(BpodSystem.GUIHandles, 'OdorWaterPlot') || ...
           ~isfield(BpodSystem.GUIHandles.OdorWaterPlot, 'PokesPlotAxis')
            fprintf('OdorWaterTrialVisualizer: GUI handles not found. Call init first.\n');
            return;
        end
        
        
        if ~isfield(BpodSystem.Data, 'nTrials') || BpodSystem.Data.nTrials == 0
            return;
        end
        
        current_trial = BpodSystem.Data.nTrials;
        last_n = str2double(get(BpodSystem.GUIHandles.OdorWaterPlot.Lastn,'String'));

        % Get trial type filter
        trial_filter = get(BpodSystem.GUIHandles.OdorWaterPlot.TrialTypeMenu, 'Value');
        
        % Clear previous plot elements
        if isfield(BpodSystem.GUIHandles.OdorWaterPlot, 'LicksHandle')
            delete(BpodSystem.GUIHandles.OdorWaterPlot.LicksHandle)
        end
        
        % Clear all state handles
        fnames = fieldnames(BpodSystem.GUIHandles.OdorWaterPlot.StateColors);
        for j = 1:last_n
            for i = 1:length(fnames)
                if isfield(BpodSystem.GUIHandles.OdorWaterPlot.StateHandle(j), fnames{i})
                    set(BpodSystem.GUIHandles.OdorWaterPlot.StateHandle(j).(fnames{i}), 'Visible', 'off');
                end
            end
        end
        
        licks = cell(last_n, 1); 
        
        for j = 1:last_n % trials       
            trial_toplot = current_trial - j + 1;
            
            if trial_toplot > 0
                % Get trial type
                if isfield(BpodSystem.Data, 'TrialTypes')
                    trial_type = BpodSystem.Data.TrialTypes(trial_toplot);
                else
                    trial_type = 1; % Default to odor trial if not specified
                end
                
                % Apply trial type filter
                if trial_filter == 2 && trial_type ~= 0 % Show only free reward
                    continue;
                elseif trial_filter == 3 && trial_type == 0 % Show only odor trials
                    continue;
                end
                
                % Get available states for this trial
                if isfield(BpodSystem.Data.RawEvents.Trial{trial_toplot}, 'States')
                    available_states = fieldnames(BpodSystem.Data.RawEvents.Trial{trial_toplot}.States);
                else
                    continue;
                end
                
                % Get alignment state
                thisTrialStateNames = get(BpodSystem.GUIHandles.OdorWaterPlot.AlignOnMenu,'String');
                thisStateName = thisTrialStateNames{get(BpodSystem.GUIHandles.OdorWaterPlot.AlignOnMenu, 'Value')};
                
                % Check if alignment state exists in this trial
                if ismember(thisStateName, available_states)
                    % use the end point of foreperiod to align or use the start point of all other states to align
                    if strcmp(thisStateName,'Foreperiod') 
                        aligning_time = BpodSystem.Data.RawEvents.Trial{trial_toplot}.States.(thisStateName)(2);
                    else
                        aligning_time = BpodSystem.Data.RawEvents.Trial{trial_toplot}.States.(thisStateName)(1);
                    end
                else
                    % If alignment state doesnt exist, align on first available state
                    if ~isempty(available_states)
                        aligning_time = BpodSystem.Data.RawEvents.Trial{trial_toplot}.States.(available_states{1})(1);
                    else
                        continue;
                    end
                end
                
                % Plot states that exist in this trial
                for i = 1:length(available_states)
                    state_name = available_states{i};
                    
                    % Check if this state has a defined color
                    if isfield(BpodSystem.GUIHandles.OdorWaterPlot.StateColors, state_name)
                        t = BpodSystem.Data.RawEvents.Trial{trial_toplot}.States.(state_name) - aligning_time;
                        x_vertices = [t(1) t(2) t(2) t(1)]';
                        y_vertices = [repmat(last_n-j,1,2)+0.1 repmat(last_n-j+1,1,2)-0.1]';
                        
                        % Ensure state handle exists and is properly initialized
                        if size(BpodSystem.GUIHandles.OdorWaterPlot.StateHandle,2) < last_n
                            BpodSystem.GUIHandles.OdorWaterPlot.StateHandle(last_n).(state_name) = fill([0 0 0 0],[0 0 0 0],BpodSystem.GUIHandles.OdorWaterPlot.StateColors.(state_name),'EdgeColor','none');
                        end
                        
                        if ~isfield(BpodSystem.GUIHandles.OdorWaterPlot.StateHandle(last_n-j+1), state_name)
                            BpodSystem.GUIHandles.OdorWaterPlot.StateHandle(last_n-j+1).(state_name) = fill([0 0 0 0],[0 0 0 0],BpodSystem.GUIHandles.OdorWaterPlot.StateColors.(state_name),'EdgeColor','none');
                        end
                        
                        if isempty(BpodSystem.GUIHandles.OdorWaterPlot.StateHandle(last_n-j+1).(state_name))
                            BpodSystem.GUIHandles.OdorWaterPlot.StateHandle(last_n-j+1).(state_name) = fill([0 0 0 0],[0 0 0 0],BpodSystem.GUIHandles.OdorWaterPlot.StateColors.(state_name),'EdgeColor','none');
                        end
                        
                        % Update state handle
                        set(BpodSystem.GUIHandles.OdorWaterPlot.StateHandle(last_n-j+1).(state_name),...
                            'Vertices', [x_vertices y_vertices],...
                            'Visible', 'on');
                    end
                end
                
                % Plot licks if they exist
                if isfield(BpodSystem.Data.RawEvents.Trial{trial_toplot}.Events, 'Port1In')
                    licks{last_n-j+1} = BpodSystem.Data.RawEvents.Trial{trial_toplot}.Events.Port1In - aligning_time;
                else
                    licks{last_n-j+1} = [];
                end
            end
        end

        % Plot lick raster
        [~, ~, BpodSystem.GUIHandles.OdorWaterPlot.LicksHandle] = ...
            plotSpikeRaster(licks, 'PlotType', 'vertline', ... 
            'XLimForCell', [str2double(get(BpodSystem.GUIHandles.OdorWaterPlot.LeftEdgeTop,'String')), ...
            str2double(get(BpodSystem.GUIHandles.OdorWaterPlot.LeftEdgeBottom,'String'))], ...
            'VertSpikePosition', -.4, 'VertSpikeHeight', .6, 'TimePerBin', .01, 'SpikeDuration', .01);
        
        % Set axis limits
        set(BpodSystem.GUIHandles.OdorWaterPlot.PokesPlotAxis, 'XLim', ...
            [str2double(get(BpodSystem.GUIHandles.OdorWaterPlot.LeftEdgeTop,'String')), ...
            str2double(get(BpodSystem.GUIHandles.OdorWaterPlot.LeftEdgeBottom,'String'))]);
        set(BpodSystem.GUIHandles.OdorWaterPlot.PokesPlotAxis,'YLim', [0 last_n]);


    %% alignon callback
    case 'alignon'
        OdorWaterTrialVisualizer('update');
        
    %% time_axis callback  
    case 'time_axis'
        OdorWaterTrialVisualizer('update');
        
    %% trial_filter callback
    case 'trial_filter'
        OdorWaterTrialVisualizer('update');
        
end

end

