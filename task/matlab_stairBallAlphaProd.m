function [retval] = matlab_stairBallAlphaProd(data, values)
% click events/check variables (Interval/productionInterval)
name={'interval','productionInterval','ballAlpha','idFlashAlpha','Nback','MovingBall_trials','radius','fixY','win_fraction'}; % ballAlphaProd

%% getting codec and events
nParam=length(name); % interval, productionInterval for now
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
%         disp(size(values{iP}));
%     disp(size(tmpValues));
    values{iP} =[values{iP}(:); tmpValues];
end

%%
locFixY=values{8};
idShort=locFixY>0;

if ~isempty(values{9})
    winF=values{9};
else
    winF=0.2;
end

% disp([length(locFixY) length(values{1})]);
if ~isempty(values{1}) && ~isempty(values{2})
    fprintf(1, 'T vs t: %d vs %d\n', [values{1}(end); round(values{2}(end))]);
end

%% plot
if ~isempty(values{5})
    nback=values{5}(end); % disp(nback);
else
    nback=0;
end
% tcrit=850; % Tcriteria for long/short prior
if length(values{1})~=length(values{2}) & ~isempty(values{1}) & ~isempty(values{2})
    minN=min([length(values{1}); length(values{2})]);
    values{1}=values{1}(end-minN+1:end);
    values{2}=values{2}(end-minN+1:end);
end
if ~isempty(values{1}) & ~isempty(values{2})
    if nargin==1
        figure(1); set(gcf,'position',[0 615 560 420],'color','w','resize','off'); % T vs t
         
        % figure(2); set(gcf,'position',[0 0 560 420],'color','w','resize','off'); % ballAlpha staircase
        
        figure(3); set(gcf,'position',[560 0 560 420],'color','w','resize','off'); % normalized bias time course   
        
    end
    nPrior=2;
    Tmat=cell(nPrior,1);
    mut=cell(nPrior,1);
    sdt=cell(nPrior,1);
    nt=cell(nPrior,1);
    
    for iPrior=1:nPrior
        Tmat{iPrior}=unique(values{1}(idShort~=(iPrior-1)));
        nT=length(Tmat{iPrior});
        mut{iPrior}=zeros(nT,1);
        sdt{iPrior}=zeros(nT,1);
        nt{iPrior}=zeros(nT,1);
        numT=length(values{1});
        for i=1:nT
            if nback==0 | numT<=nback % plot all previous
                id=values{1}==Tmat{iPrior}(i) & idShort~=(iPrior-1);
            else
                id=values{1}==Tmat{iPrior}(i) & idShort~=(iPrior-1) & logical([zeros(numT-nback,1); ones(nback,1)]);
            end
            t=values{2}(id);
            tClean=removeOutlier(t(t>0),3);
            [mut{iPrior}(i),sdt{iPrior}(i),nt{iPrior}(i)]=meanSDwoNeg(tClean);
        end
%         idShort=Tmat<tcrit;
    end
    
    % plot T vs t
    Ttmp=values{1}(end); ttmp=values{2}(end);
    figure(1); set(gcf,'position',[0 615 560 420],'color','w','resize','off');
    if locFixY(end)>0, cmap=[1 0 0]; cmap2='m'; % if Ttmp < tcrit, 
    else cmap=[0 0 1];cmap2='c';  end;
    errorbar(Tmat{1},mut{1},sdt{1},'r-'); drawnow; hold on;
    errorbar(Tmat{2},mut{2},sdt{2},'b-'); drawnow; hold on;
%     plot(Ttmp,ttmp,'kd','markerfacecolor','k'); drawnow; hold off;
    plot(Ttmp,ttmp,'o','color',cmap,'markerfacecolor','w','markersize',8); drawnow; hold on; % on
    set(gca,'ylim',[400 1600]); plotIdentity(gca); plotWeberLine(gca,winF);
    drawnow; hold off;
   xlabel('t_s (ms)'); ylabel('t_p (ms)');
    
    figure(3);set(gcf,'position',[560 0 560 420],'color','w','resize','off'); % normalized bias time course
    plotHorizon(gca);
    plot(numT,(ttmp-Ttmp)/Ttmp,'.','color',cmap,'markersize',11); hold all;   
    if nback~=0 & numT>nback % plot all previous
        set(gca,'xlim',[max([1 numT-nback+1]) numT]);
    else
        axis tight;
    end        
    xlabel('trials'); ylabel('(t_p-t_s)/t_s');
    
    % ballAlpha staircase (3 for b/t ready and set; 7 for production)
  %  figure(2);set(gcf,'position',[0 0 560 420],'color','w','resize','off');
  %  if nback~=0
   %     set(gca,'xlim',[max([1 length(values{7})-nback+1]) length(values{7})]);
  %  else
   %     axis tight;
   % end 
   % if ~isempty(values{7})
  %      plot(length(values{7}),values{7}(end),'.','color',cmap,'markersize',11); hold all;
  %  end
  %  plot(length(values{3}),values{3}(end),'.','color',cmap2,'markersize',11); hold all;
  %  legend('flash radius','ball alpha (measure)'); % ball alpha (produce)
  %  xlabel('trials'); ylabel('ball alpha');
    
    %%
%     % check scalar property
%     if length(mut(idShort))>1 && length(sdt(idShort))>1
%         [r,p]=corr(mut(idShort),sdt(idShort));
%         fprintf(1, 'meanWF: %d (p=%d)\n',[mean(sdt(idShort)./mut(idShort)); p]);
%     end
%     if length(mut(~idShort))>1 && length(sdt(~idShort))>1
%         [r,p]=corr(mut(~idShort),sdt(~idShort));
%         fprintf(1, 'meanWF: %d (p=%d)\n',[mean(sdt(~idShort)./mut(~idShort)); p]);
%     end
    
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