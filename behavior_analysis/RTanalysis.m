function RTanalysis(varargin)

% analyze RT of maze-set-go task
% for both offline & online phase
% independent variables: image_size, num_turns, path_length

% plot the following
% 1) offline RT distribution =f(maze size)
% 2) offline RT = f(# turns)
% 3) offline RT = f(total path length)
% 4) offline RT = f(# turns, total path length)
% 
% 5) tp=f(ts|maze size): plotTpTs
% 6) tp-ts=f(# turns): plotmultiple
% 7) variance analysis

% dependence
%: v2struct applytofig4keynote setFigPos plotReg plotCmap pplot.mat

% log
% 2021/5/19: negating prey_distance_at_response for tp

load pplot.mat;

%% input processing
if isempty(varargin)
    sess='2021_05_13_12_29_07';
else
    sess=varargin{1};
end
dataPath='/Users/hansem/Documents/MazeSetGo/behavior_analysis/data/';

load(fullfile(dataPath,[sess '.mat']));
v2struct(trial_df); % num_turns path_length prey_distance_at_response RT_offline image_size
num_turns=double(num_turns);
path_length=double(path_length);
image_size=double(image_size);

% other task variables (TBD: import directly from python)
%   get speed
%         self._prey_speed = 1000. / (60. * ms_per_unit) 
%             meta_state['prey_distance_remaining'] -= self._prey_speed
refreshRate=60;
speed=1000/(refreshRate*2000); % frameWidth / refresh
mazeSize=12;
borderWidth= 0.18;

% translate distance into time
pathLengthScreenUnit=(1 - 2 * borderWidth) * path_length / mazeSize;
ts= double(pathLengthScreenUnit/speed/refreshRate*1000); % [ms]
err=-prey_distance_at_response/speed/refreshRate*1000; % [ms]
tp=ts+err;
rt=RT_offline;

%         # Prey distance remaining is how far prey has to go to exit maze
%         # It will be continually updated in the meta_state as the prey moves
%         prey_distance_remaining = (self._prey_lead_in +
%             (1 - 2 * self._border_width) * len(prey_path) / maze_size)             [frame]
%         self._prey_lead_in = 0.15
%         self._border_width = 0.18  # boundary space around the maze on all sides

%% preprocessing
% remove outlier trials based on timing error
[dTmp,id,p]=removeOutlierMAD(tp-ts,5);

% remove outlier trials based on RT offline
id(rt>10)=0;

num_turns = num_turns(id);
path_length =path_length(id); 
image_size = image_size(id);

tp=tp(id);
ts=ts(id);
err=err(id);
rt=rt(id);

% plot simple histograms
nBin=30;
figure; setFigPos(1,1);
histogram(num_turns,nBin,'facecolor','w');
xlabel('# turns'); ylabel('trials'); applytofig4keynote;

figure; setFigPos(1,2);
histogram(path_length,nBin,'facecolor','w');
xlabel('path length [grid cell]'); ylabel('trials'); applytofig4keynote;

figure; setFigPos(1,3);
histogram(image_size,nBin,'facecolor','w');
xlabel('image size [degree]'); ylabel('trials'); applytofig4keynote;

close all;

%% main plots
% 1) offline RT distribution =f(maze size)
imageSizeList=unique(image_size);
figure; setFigPos(2,1); hold all;
violinPlot(rt,'groups',image_size,'histOpt',1,'showMM',4,'xNames',imageSizeList,'yLabel','RT offline [ms]');
xlabel('image size [deg]');
applytofig4keynote;

hFig=figure; setFigPos(2,1); hold all;
for i=1:length(imageSizeList)
    histStairs(rt(imageSizeList(i)==image_size),nBin,1,hFig,pplot.cmap{i});
end
xlabel('RT offline [ms]'); ylabel('p(trials)'); 
axis tight;
applytofig4keynote;

% 2) offline RT = f(# turns)
hFig=figure; setFigPos(2,2); hold all;
plot(num_turns,rt,'ko','markerfacecolor','w');
axis tight;
plotReg(num_turns,rt,hFig,'k');
xlabel('# turns'); ylabel('RT offline [ms]');
applytofig4keynote;

% 3) offline RT = f(total path length)
hFig=figure; setFigPos(2,3); hold all;
plot(path_length,rt,'ko','markerfacecolor','w');
axis tight;
plotReg(path_length,rt,hFig,'k');
xlabel('path length [grid]'); ylabel('RT offline [ms]');
applytofig4keynote;

% 4) offline RT = f(# turns, total path length)
hFig=figure; setFigPos(2,4); hold all;
plot(path_length,num_turns,'ko','markerfacecolor','w');
axis tight;
plotReg(path_length,num_turns,hFig,'k');
xlabel('path length [grid]'); ylabel('# turns');
applytofig4keynote;

[rho,pval]=partialcorr(num_turns(:),rt(:),path_length(:));
disp([rho pval]);

hFig=figure; setFigPos(2,4); hold all;
tmpCmap=parula(length(unique(num_turns)));
plot3multiple(path_length(:),num_turns(:),rt(:),tmpCmap(num_turns(:)+1,:),'o');
set(gca,'view',[0 0]);
axis tight; grid on;
xlabel('path length [grid]'); ylabel('# turns'); zlabel('RT offline [ms]');
applytofig4keynote;

% % export 3D moive
% OptionZ.FrameRate=15;OptionZ.Duration=5.5;OptionZ.Periodic=true;
% CaptureFigVid([-20,10;-110,10;-190,80;-290,10;-380,10], 'offlineRT_nTurns_pathLength',OptionZ)

% 5) tp=f(ts|maze size): plotTpTs
hF=figure; setFigPos(1,1); % summary
[mut,sdt,nt,tmp]=plotTpTs(ts,tp,1,'k',hF,'-',10);
ylim([0 max(ylim)]);applytofig4keynote;

hF=figure; setFigPos(1,1); % summary
for i=1:length(imageSizeList)
plotTpTs(ts(imageSizeList(i)==image_size),tp(imageSizeList(i)==image_size),...
    1,pplot.cmap{i},hF,'-',10);
end
ylim([0 max(ylim)]);applytofig4keynote;

% 6) tp-ts=f(# turns): plotmultiple
hFig=figure; setFigPos(1,2); hold all;
plot(num_turns,err,'ko','markerfacecolor','w');
axis tight;
plotReg(num_turns,err,hFig,'k');
xlabel('# turns'); ylabel('t_p - t_s (ms)');
applytofig4keynote;

% tp-ts=f(# turns)
[rho,pval]=partialcorr(num_turns(:),err(:),path_length(:));
disp([rho pval]);

% tp-ts=f(path length)
[rho,pval]=partialcorr(num_turns(:),err(:),path_length(:));
disp([rho pval]);

% 7) variance analysis: sd vs ts

% find 0-turn & measure wf
sd=std(tp(num_turns==0));
m=mean(tp(num_turns==0));

% get non-zero-turn trials
[mut,sdt,nt,tmp]=plotTpTs(ts,tp,0,'k',hF,'-',10);

figure; hold all; setFigPos(1,5);
plot(mut,sdt,'ko-','markerfacecolor','w');
xlim([0 4000]);
plot([0 max(xlim)],[0 sd/m*max(xlim)],'k:');
plot(m,sd,'ko','markerfacecolor','k');
axis tight;xlim([0 4000]);
plotVertical(gca,unique(ts(num_turns==0)),[]);

xlabel('mean t_p');
ylabel('SD t_p');
applytofig4keynote;

