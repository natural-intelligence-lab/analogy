function [retval] = plot_tstp(data, values)
% click events/check variables (Interval/productionInterval)
name={'ts','tp','prey_opacity','num_turns','RT_offline','end_x_prey','end_x_agent','end_x_distract'}; % ballAlphaProd

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

%%
winF=0.2;

% debug
% if ~isempty(values{1}) && ~isempty(values{2}) && ~isempty(values{3}) && ~isempty(values{4}) && ~isempty(values{5}) &&...
%         ~isempty(values{6}) && ~isempty(values{7}) && ~isempty(values{8})
%     %     fprintf(1, 'ts vs tp: %d vs %d\n', [values{1}(end); round(values{2}(end))]);
%     %    fprintf(1, ', alpha: %d\n', values{3}(end));
% end

%% plot
cmap=[    1.0000         0         0;...
    0    0.6000         0;...
    0         0    1.0000;...
    1.0000    0.5000         0;...
    1.0000         0    1.0000;...
    0    1.0000    1.0000;...
    0.5000         0    1.0000;...
    1.0000    1.0000         0;...
    0.5000    0.5000    0.5000;...
    0         0         0;...
    107.0000   66.0000   41.0000]; %'r'    'g'    'b'    'o'    'm'    'c'    'p'    'y'    'gray'    'k'    'brown'


% tcrit=850; % Tcriteria for long/short prior
if length(values{1})~=length(values{2}) & ~isempty(values{1}) & ~isempty(values{2})
    minN=min([length(values{1}); length(values{2})]);
    values{1}=values{1}(end-minN+1:end);
    values{2}=values{2}(end-minN+1:end);
end

%% MAIN
if ~isempty(values{1}) & ~isempty(values{2}) & ~isempty(values{3}) & ~isempty(values{4}) & ~isempty(values{5}) &...
        ~isempty(values{6}) & ~isempty(values{7}) & ~isempty(values{8})
    %     if nargin==1
    %% plot T vs t
    Ttmp=values{1}(end); ttmp=values{2}(end); RT_offline=values{5}(end);
    % disp((values{3}));
    if length(values{3})>=2
        opacity=values{3}(end-1);
    else
        opacity=values{3}(end);
    end
    if length(values{4})>=2
        nTurn=values{4}(end); % -1);
    else
        nTurn=values{4}(end);
    end
    figure(1); set(gcf,'position',[0 615 560 420],'color','w','resize','off'); hold on;
    if nTurn==0
        cmap='r';
    elseif nTurn==2
        cmap='g';
    elseif nTurn==1
        cmap='m';
    elseif nTurn==3
        cmap='c';
    elseif nTurn==4
        cmap='b';
    end
    
    % plot
    fprintf(1, 'ts vs tp: %d vs %d , alpha: %d\n', [Ttmp; ttmp; opacity]);
    plot(Ttmp+0.2*(rand(1,1)-0.5),ttmp,'o','markerfacecolor',cmap,'color',cmap,'linewidth',1,'markersize',3); drawnow; hold on;

    % TBD: debug
    %    % plot regression for 0 opacity & 2-turn
    %    ts2turn=3.5;
    %    disp([size(values{3}) size(values{1})]);
    %    tmpId=values{3}(:)==0 & values{1}(:) > ts2turn;
    %    B=regress(values{2}(tmpId),[values{1}(tmpId) ones(nnz(tmpId),1)]);
    %    tmpX=xlim;
    %    tmpY=[tmpX; ones(1,2)]*B;
    %    plot(tmpX,tmpY,'r-','linewidth',2);
    
    plotIdentity(gca); plotWeberLine(gca,winF);
    drawnow; hold on;
    xlabel('t_s (s)'); ylabel('t_p (s)');
    %         figure(2); set(gcf,'position',[0 0 560 420],'color','w','resize','off'); % ballAlpha staircase
    
    %% offline error
    end_x_prey=values{6}(end); end_x_agent=values{7}(end); end_x_distract=values{8}(end);
    figure(2); set(gcf,'position',[560 615 560 420],'color','w','resize','off'); hold on;
    plot(end_x_prey-end_x_agent,end_x_distract-end_x_agent,'o','markerfacecolor',cmap,'color',cmap,'linewidth',1,'markersize',3); drawnow; hold on;
    xlabel('prey_x - agent_x'); ylabel('distract_x - agent_x');
    
    %% RT offline
    %        figure(2); set(gcf,'position',[560 615 560 420],'color','w','resize','off'); hold on;
    %plot(Ttmp+0.2*(rand(1,1)-0.5),RT_offline,'o','markerfacecolor',cmap,'color',cmap,'linewidth',1,'markersize',3); drawnow; hold on;
    %xlabel('t_s (s)'); ylabel('RT (offline) (s)');
    
    
    %     % simple linear regression as a model-free check of prior effect (slope<1 & intercept>0)
    %     if length(mut(idShort))>1 && length(sdt(idShort))>1
    %         stats=regstats(mut(idShort),Tmat(idShort),'linear',{'tstat'}); % beta yhat r mse rsquare tstat
    %         fprintf(1, 'intercept: %d (p=%d), slope: %d (p=%d)\n',[stats.tsat.beta(1) stats.tsat.pval(1) stats.tsat.beta(2) stats.tsat.pval(2)]);
    %     end
    %     if length(mut(~idShort))>1 && length(sdt(~idShort))>1
    %         stats=regstats(mut(~idShort),Tmat(~idShort),'linear',{'tstat'}); % beta yhat r mse rsquare tstat
    %         fprintf(1, 'intercept: %d (p=%d), slope: %d (p=%d)\n',[stats.tsat.beta(1) stats.tsat.pval(1) stats.tsat.beta(2) stats.tsat.pval(2)]);
    %     end
    
else
    fprintf(1, 'number of physical intervals does not equal number of production intervals! or empty')
end

% t: productionInterval ((now()-set2_cue_onset_time_us)/1000)
% abort: -2*T
% after ready: (now() - ready_cue_onset_time_us -interval*2000) / 1000.0 (<0)
% after Set1: (now() - set1_cue_onset_time_us -interval*1000) / 1000.0 (<0)
% No Saccade: (now() - set2_cue_onset_time_us) / 1000.0 (>3500)

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