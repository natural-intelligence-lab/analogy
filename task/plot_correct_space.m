function [retval] = plot_correct_space(data, values)

% MAKE SURE click events/check variables 

% fig.1: overall performance: % complete (given fixation) & % correct
% fig.2: spatial map of % correct

% TBD: p(saccade to cue), moving window

name={'CorrectTrialsTarget',... 
    'IncorrectTrialsTarget',... 
    'localTrials',... 
    'CorrectTrials',...
    'IncorrectTrials',...
    }; 

%% getting codec and events
nParam=length(name); 
if nargin == 1
    values = cell(nParam,1);
    for i=1:nParam
        values{i}=[];
    end
end

codec = data.event_codec; % structure with fields(tagname - event 'code's, default)
all_events = data.events; % [code,timing,value]?

%% extracting data
event_code=[];
% finding event code for a specific tagname
for iP=1:nParam
    for i = 1:length(codec)
        if (strcmp(codec(i).tagname, name{iP}))
            event_code = [event_code; codec(i).code];
            break
        end
    end
end

% finding values
for iP=1:nParam
    indices = ([all_events(:).event_code] == event_code(iP));
    events = all_events(indices);
    
    if iP==1 || iP==2 % cell
        tmpDataCell=events(:).data;
        tmpValues=cast([tmpDataCell{end}], 'double');
    else
        tmpValues=cast([events(:).data], 'double');        
    end
    tmpValues=tmpValues(:);
    values{iP} =[values{iP}(:); tmpValues];
end

%% other parameters

nTheta=4;
nTarget=4;

target_radius= 3;
cue_radius= 8;

% get XY locations
XY=[];
for i=1:nTheta
    for j=1:nTarget
        cue_angle=360/nTheta*(i-1);
        cueX=cue_radius*cosd(cue_angle);
        cueY=cue_radius*sind(cue_angle);
        
        target_angle=360/nTarget*(j-1);
        XY=[XY; cueX+target_radius*cosd(target_angle) ...
            cueY+target_radius*sind(target_angle)];
    end
end

% plot
nCmap=100;
cmap=flipud(jet(nCmap));

%% MAIN
if ~isempty(values{1}) & ~isempty(values{2}) & ~isempty(values{3}) 
    CorrectTrialsTarget=values{1}(end-(nTheta*nTarget-1):end);
    IncorrectTrialsTarget=values{2}(end-(nTheta*nTarget-1):end);
    pCorrrectTarget=CorrectTrialsTarget./(IncorrectTrialsTarget+CorrectTrialsTarget);
    
    localTrials=values{3}(end);
    CorrectTrials=values{4}(end);
    IncorrectTrials=values{5}(end);
    
    nComplete=CorrectTrials+IncorrectTrials;
    pComplete=nComplete/localTrials*100;
    pCorrect=CorrectTrials/nComplete*100;
    
    % fig.1: overall performance: % complete (given fixation) & % correct
    figure(1); set(gcf,'position',[0 615 420 420],'color','w','resize','off'); hold on;
    plot(localTrials,pComplete,'o','markerfacecolor','r','color','r','linewidth',1,'markersize',3); drawnow; hold on;
    plot(localTrials,pCorrect,'o','markerfacecolor','b','color','b','linewidth',1,'markersize',3); drawnow; hold on;
    xlabel('# trials with fixation'); ylabel('% complete(R)/correct(B)');
    
    % fig.2: spatial map of % correct
    figure(2); set(gcf,'position',[0 0 420 420],'color','w','resize','off'); hold on;
    for i=1:size(XY,1)
       tmpCmap=cmap(ceil(pCorrrectTarget(i)),:);
       plot(XY(i,1),XY(i,2),'s','markerfacecolor',tmpCmap,'color',tmpCmap,'linewidth',1,'markersize',8); drawnow; hold on;
    end
    xlabel('X'); ylabel('Y');
end
%% output
retval = values;

%%

function [m sd n]=meanSDwoNeg(x)
n=nnz(x>0);
m=mean(x(x>0));
sd=std(x(x>0),1);

function plotIdentity(hAx)
% plot identity line with black dotted
% input: hAx for gca of current plot
%
% 2013/10/19
% hansem sohn

hold on;
x=get(hAx,'xlim'); y=get(hAx,'ylim');
if strcmp(get(gca,'xscale'),'log') & strcmp(get(gca,'yscale'),'log')
    loglog([max([x(1);y(1)]);min([x(2);y(2)])],[max([x(1);y(1)]);min([x(2);y(2)])],'k:');
else
    plot([max([x(1);y(1)]);min([x(2);y(2)])],[max([x(1);y(1)]);min([x(2);y(2)])],'k:');
end
hold on;

function [dataWOout,id,pOut]=removeOutlier(data,nSD)
% removing outliers
% input: data [n x 1], nSD for criteria of SD
% output: data without outlier, id to indicate not outlier in the original
% data, pOut for percetage of outliers

idNN=(~isnan(data)); % removing NaN first
idNO=abs(data(idNN)-mean(data(idNN)))<nSD*std(data(idNN));
id=zeros(length(data),1); id(idNN)=idNO; id=logical(id);
pOut=(length(data)-nnz(id))/length(data)*100;
dataWOout=data(id);

% % debug
% a=[rand(10,1); NaN];
% [d,id,p]=removeOutlier(a,3)

function h=plotHorizon(hAx,varargin)
% plot vertical lines into the plot
% input: hAx for gca of current plot
% varargin: first cell element for position (if empty, zero)
%               , 2nd cell for color (if empty, black)
% 2014/10/2
% hansem sohn

hold on;

if isempty(varargin)
    cmap=[0 0 0]; y=0;
else
    if isempty(varargin{1})
        y=0;
    else
        y=varargin{1};
    end
    if isempty(varargin{2})
        cmap=[0 0 0];
    else
        cmap=varargin{2};
    end
end

x=get(hAx,'xlim'); h=[];
for i=1:length(y)
    hTmp=plot([min(x); max(x)],[y(i); y(i)],':','color',cmap); hold on;
    h=[h; hTmp];
end

%% debug
% plot(rand(10,1));
% plotHorizon(gca);
% clf; plotHorizon(gca,0,[1 0 0]);
% plotHorizon(gca,[0 1],[1 0 1]);
% plotHorizon(gca,[],[1 0 1]);
% plotHorizon(gca,[0 1],[]);

function plotWeberLine(hAx,weberF)
% plot linear line with black dotted
% input: hAx for gca of current plot, weber fraction
%
% 2015/8/18
% hansem sohn

hold on;
x=get(hAx,'xlim'); y=get(hAx,'ylim');
if strcmp(get(gca,'xscale'),'log') & strcmp(get(gca,'yscale'),'log')
    loglog([max([x(1);y(1)]);min([x(2);y(2)])],[max([x(1);y(1)]);min([x(2);y(2)])],'k:');
else
    plot([max([x(1);y(1)]);min([x(2);y(2)])],(1+weberF)*[max([x(1);y(1)]);min([x(2);y(2)])],':','color',[0.5 0.5 0.5]);hold on;
    plot([max([x(1);y(1)]);min([x(2);y(2)])],(1-weberF)*[max([x(1);y(1)]);min([x(2);y(2)])],':','color',[0.5 0.5 0.5]);hold on;
end