function [retval] = plot_alpha_staircase(data, values)

% MAKE SURE click events/check variables (Interval/productionInterval)

% fig.1: fix_alpha_min

name={'fix_alpha_min',... % 1 from meta_state
    'localTrials',... % 11
    'CorrectTrials',... % 17
    }; % 16

%% getting codec and events
nParam=length(name); % ts, tp for now
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
        
        tmpValues=cast([events(:).data], 'double'); tmpValues=tmpValues(:);
        values{iP} =[values{iP}(:); tmpValues];

end

%% other parameters

% plot
cmap=[    1.0000         0         0;... % 1 'r' 
    0    0.6000         0;... %    2 'g' 
    0         0    1.0000;... %3    'g'    'b'
    1.0000    0.5000         0;... %4    'o'
    1.0000         0    1.0000;... %5    'm'
    0    1.0000    1.0000;...%6    'c'
    0.5000         0    1.0000;...%7    'p'
    1.0000    1.0000         0;...
    0.5000    0.5000    0.5000;...
    0         0         0;...
    107.0000   66.0000   41.0000]; %'r'    'g'    'b'    'o'    'm'    'c'    'p'    'y'    'gray'    'k'    'brown'


%% MAIN
if ~isempty(values{1}) & ~isempty(values{2}) & ~isempty(values{3}) 

     % fig.4: staircase
     localTrials=values{2}(end);
     fix_alpha_min=values{1}(end);
     figure(4); set(gcf,'position',[0 615 420 420],'color','w','resize','off'); hold on;
     plot(localTrials,fix_alpha_min,'o','markerfacecolor','r','color','r','linewidth',1,'markersize',3); drawnow; hold on;
    xlabel('trials'); ylabel('fix_alpha_min');

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