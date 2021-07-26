function generateMazeImage(varargin)

% generate maze image from serialized files
% input: folder location for files
%
% for now, no distractor & only prey path (for RNN)
% not include different image size (all square 12)

%% init

if nargin==0
    folder='/Users/hansem/Documents/MazeSetGo/task/stimuli/random/Random12Square';
else
    folder=varargin{1};
end

% delimiter of files
delimit_prey={'[[',']]'};
delimit_nTurns={'num_turns": ', ', "path'};
delimit_path={', "path_length": ','}]'};

% image
lineColor=.85;
bgColor=.5;
imSize = 12; % pixel

% info about prey path in maze class
%         Indexing scheme: Width is component 0 of cell indices, height is
%         component 1. Width is indexed left-to-right, height is indexed
%         bottom-to-top.
% 
%         Args:
%             width: Int. Width of the grid of maze cells.
%             height: Int. Height of the grid of maze cells.
%             prey_path: Iterable of int 2-tuples, which are indices of cells in
%                 the prey path.

%% main
cd(folder);
fs=dir;

cd('..');
mkdir('image');
cd(folder);

nTurn=[];
path=[];
pathLength=[];

for i=1:length(fs)
    if ~fs(i).isdir % not . or ..
        d=importdata(fs(i).name);
        id=str2num(fs(i).name);
        % prey
        i0=regexp(d{1},delimit_prey{1});
        i1=regexp(d{1},delimit_prey{2});        
        strPrey=d{1}((i0+1):i1);
        mat1d=str2num(strPrey);        
        mat2d=reshape(mat1d,[2 length(mat1d)/2]); % 2 x path
        mat2d=mat2d+1; % indexing in python start from 0
        
        % n turns
        i0=regexp(d{1},delimit_nTurns{1});
        i1=regexp(d{1},delimit_nTurns{2});        
        strNTurns=d{1}((i0+12):(i1-1));
        
                
        % path length
        i0=regexp(d{1},delimit_path{1});
        i1=regexp(d{1},delimit_path{2});        
        strPathL=d{1}((i0+16):(i1-1));
        
        
        % not include 0-turn trials with non-12 path length
        if ~(str2num(strNTurns)==0 & (str2num(strPathL)+1)~=imSize)
            path{id}=mat2d;
            nTurn=[nTurn; str2num(strNTurns)];
            pathLength=[pathLength; str2num(strPathL)];
            
            %% image generation
            im=ones(imSize,imSize)*bgColor;
            for j=1:size(mat2d,2)
                x=mat2d(1,j);
                y=mat2d(2,j);
                im(x,y)=lineColor;
            end
            imwrite(im,['../image/' fs(i).name '.png'],'png','Alpha',double(im~=bgColor));
%             pause(0.5);
        end
                
    end
end % for i=1:length(fs)

cd('../image');
path=path(~cellfun('isempty',path)); % remove 0-turn trials with non-12 path length
save('mazeParams.mat','nTurn','path','pathLength');
