%% This code is for SHST with the eigenvectors of Ly are normalized in the form of random walk.
%clc;
clear; close all;

addpath(genpath('makeBipartiteGraph'));
addpath(genpath('ParamTuning'));
addpath(genpath( 'makeSuperpixels'));
addpath(genpath('utilities'));
addpath (genpath('evals'));
addpath (genpath('others'));


%% parameter setting
para.nb = 1; % neighbours for adjacency matrix
para.alpha = 0.001;
para.beta = 20;% scale for exponential in similarity measuring
para.mode =  'ColorCovMat';%'Neighbours';%'ColorTexton'; % 'SAS'; %
para.votingmode = 'HierarchicalTree';% 'VotingTransfer'
Nsegs = 5;
para.rangeOfK = 2:20;
%para.knn = 1;
para.minSizeThreshold = 0.1:0.1:0.9;%50; a 0.1 increamental in each step
%para.lifetimelevel = 1;%the lifetime level decide the cluster number,1 represents the highest lifetime; and,2 is the second highest...
para.distType = 'LogEuclidean'; % LogEuclidean or Foerstner

bsdsRoot = '.\BSDS';
fid = fopen(fullfile(bsdsRoot,'Nsegs500.txt'),'r');
Nimgs = 500;
[BSDS_INFO] = fscanf(fid,'%d %d\n',[2,Nimgs]);
fclose(fid);
maxlifetimelevel = 10;
NumOfminSize = length(para.minSizeThreshold);
%numOflifetimelevel=length(maxlifetimelevel);
% PRI_all_HST = zeros(Nimgs,maxlifetimelevel,NumOfminSize);
% VoI_all_HST = zeros(Nimgs,maxlifetimelevel,NumOfminSize);
% GCE_all_HST = zeros(Nimgs,maxlifetimelevel,NumOfminSize);
% BDE_all_HST = zeros(Nimgs,maxlifetimelevel,NumOfminSize);
 
PRI_all_HST = zeros(Nimgs,maxlifetimelevel*NumOfminSize);
VoI_all_HST = zeros(Nimgs,maxlifetimelevel*NumOfminSize);
GCE_all_HST = zeros(Nimgs,maxlifetimelevel*NumOfminSize);
BDE_all_HST = zeros(Nimgs,maxlifetimelevel*NumOfminSize);

for idxI=1:Nimgs
    img_name = int2str(BSDS_INFO(1,idxI));
    img_loc = fullfile(bsdsRoot,'images','test',[img_name,'.jpg']);    
    if ~exist(img_loc,'file')
        img_loc = fullfile(bsdsRoot,'images','train',[img_name,'.jpg']);
        if ~exist(img_loc,'file')
            img_loc = fullfile(bsdsRoot,'images','val',[img_name,'.jpg']); 
        end
    end
    img = im2double(imread(img_loc)); [X,Y,~] = size(img);img_size=[X,Y];
    
   
   SpPath = fullfile(bsdsRoot,'Superpixels',img_name,[img_name,'.mat']);
   CovMatPath = fullfile(bsdsRoot,'CovDistance',img_name); % because the file name is different to the distance type.
   TextonsPath = fullfile(bsdsRoot,'Textons',img_name,[img_name,'texton64.mat']);
   TextonDistPath = fullfile(bsdsRoot,'TextonDist',img_name);
   featPath = make_featPath(SpPath,CovMatPath,TextonsPath,TextonDistPath);
   [seg,labels_img,seg_vals,seg_lab_vals,seg_edges,seg_img]=loadSpSegmentations(SpPath); 

    % Build bipartite graph
     B = build_bipartite_graph(img,para,seg,seg_lab_vals,seg_edges,featPath);
%     switch para.mode
%         case 'SAS'
%             B = build_bipartite_graph_SAS(img,para,seg,seg_lab_vals,seg_edges); 
%         case 'ColorCovMat'
%             B = build_bipartite_graph_ColorCovMat();
%     end
    % Evaluation preparing: get the ground-turth
    gt_loc = fullfile(bsdsRoot,'groundTruth','test',[img_name,'.mat']);
    if ~exist(gt_loc,'file')
        gt_loc = fullfile(bsdsRoot,'groundTruth','train',[img_name,'.mat']);
        if ~exist(gt_loc,'file')
            gt_loc = fullfile(bsdsRoot,'groundTruth','val',[img_name,'.mat']); 
        end
    end
    gt = load(gt_loc);
    gt_imgs = cell(1,length(gt.groundTruth));
    for t=1:length(gt_imgs)
        gt_imgs{t} = double(gt.groundTruth{t}.Segmentation);
    end
    
    % Applying Voting method
    switch para.votingmode 
        case 'Tcut'
        case 'Vcut'
            label_img_Vcut = VcutTuning(B,Nsegs,[X,Y]);%the label_img_Vcut is a cell
            % save the segmentation result
            out_path = fullfile('results',['BSDS500_',para.mode],img_name,'Vcut');
            if ~exist(out_path,'dir')
                mkdir(out_path);
            end
            %view_segmentationSegTree(img,label_img(:),out_path,img_name,Nsegs,0);
           for k = 1:numOfK
                out_vals = eval_segmentation(label_img_Vcut{k},gt_imgs); 
               % fprintf('%6s: %2d %9.6f, %9.6f, %9.6f, %9.6f \n', img_name, Nsegs, out_vals.PRI, out_vals.VoI, out_vals.GCE, out_vals.BDE);
%                 PRI_all(idxI,k) = out_vals.PRI;
%                 VoI_all(idxI,k) = out_vals.VoI;
%                 GCE_all(idxI,k) = out_vals.GCE;
%                 BDE_all(idxI,k) = out_vals.BDE;
           end

        case 'HierarchicalTree'
            % the tree saving path
            saveResultsPath = fullfile(bsdsRoot,'SegTree_RamdomWalkNormalized',para.mode,img_name);
            %IsDone = [saveResultsPath,'\treeData.mat']; %tempelary used for check if the image has been done already 
%             if ~exist(IsDone,'file')   % check if the tree has been done already,tempelary used here
            if ~exist(saveResultsPath,'dir')
                mkdir(saveResultsPath)
            end
           % get the incident matrix
            H = B(1:X*Y,:);
            H(H>0)=1;
            % build the tree
            [Tree,Parent,Kept,ic,RegionSize] = build_hierarchical_tree_minSizeTuning2(H,B,para); % the Tree,Parent,Kept are in cell format 

            % save the tree
            saveTree_minSizeTuning(Tree,Parent,Kept,ic,RegionSize,[saveResultsPath,'\treeData.mat']);
%             else
%                 loadTree(IsDone);
%             end
            % set segment result saving path
            out_path = fullfile('ResultsOfGivenNsegs',['BSDS500_',para.mode],img_name,'SegTree'); 
            if ~exist(out_path,'dir')
                mkdir(out_path);
            end
            % varialbes for show the result on screen
            outputPRI_best = 0;
%             outputVoI = 0;
%             outputGCE = 0;
%             outputBDE = 0;
%             LFTlevel =0;
%             minSizeThreshold = 0;
            PRI_HST = zeros(maxlifetimelevel,NumOfminSize);
            VoI_HST = zeros(maxlifetimelevel,NumOfminSize);
            GCE_HST = zeros(maxlifetimelevel,NumOfminSize);
            BDE_HST = zeros(maxlifetimelevel,NumOfminSize);             
            for t = 1:NumOfminSize                
                for lifetimelevel = 1:maxlifetimelevel
                    [label_img, NumOfSeg] = getLabel_with_specified_cluster_number(Tree{t},ic,Parent{t},Kept{t},img_size,lifetimelevel);
                    % save the segment result  
                    %view_segmentationSegTree(img,label_img(:),out_path,img_name,lifetimelevel,0);
                    % Evaluation
                    out_vals = eval_segmentation(label_img,gt_imgs);                    
                    PRI_HST(lifetimelevel,t) = out_vals.PRI;
                    VoI_HST(lifetimelevel,t) = out_vals.VoI;
                    GCE_HST(lifetimelevel,t) = out_vals.GCE;
                    BDE_HST(lifetimelevel,t) = out_vals.BDE;
                    if outputPRI_best<out_vals.PRI;
                        outputPRI_best = out_vals.PRI;
                        outputVoI = out_vals.VoI;
                        outputGCE = out_vals.GCE;
                        outputBDE = out_vals.BDE;
                        LFTlevel = lifetimelevel;
                        minSizeThreshold = para.minSizeThreshold(t);
                        templabel_img = label_img;
                    end
                end             
            end
            PRI_all_HST(idxI,:) = PRI_HST(:);
            VoI_all_HST(idxI,:) = VoI_HST(:);
            GCE_all_HST(idxI,:) = GCE_HST(:);
            BDE_all_HST(idxI,:) = BDE_HST(:);
            % save the best segment result  
            view_segmentationSegTree(img,templabel_img(:),out_path,img_name,LFTlevel,0);
    end 
    fprintf('%6s: %1.2f %2d %9.6f, %9.6f, %9.6f, %9.6f \n', img_name, minSizeThreshold, LFTlevel, outputPRI_best, outputVoI, outputGCE, outputBDE);
    %fprintf('%6s: is done.\n', img_name);
end
EvaluationResult_path = fullfile('ResultsOfGivenNsegs',para.votingmode,'BSDS500');
if ~exist(EvaluationResult_path,'dir')
    mkdir(EvaluationResult_path);
end
% reshape into a 3d array.
PRI_all_HST = reshape(PRI_all_HST,Nimgs,maxlifetimelevel,NumOfminSize);
VoI_all_HST = reshape(VoI_all_HST,Nimgs,maxlifetimelevel,NumOfminSize);
GCE_all_HST = reshape(GCE_all_HST,Nimgs,maxlifetimelevel,NumOfminSize);
BDE_all_HST = reshape(BDE_all_HST,Nimgs,maxlifetimelevel,NumOfminSize);
save([EvaluationResult_path,'\',para.mode,'ResultAll.mat'],'PRI_all_HST','VoI_all_HST','GCE_all_HST','BDE_all_HST');
[PRI,~]=max(max(PRI_all_HST,[],2),[],3);
I = bsxfun(@eq,PRI_all_HST,PRI);
fprintf('Mean: %14.6f, %9.6f, %9.6f, %9.6f \n', mean(PRI), mean(VoI_all_HST(I)), mean(GCE_all_HST(I)), mean(BDE_all_HST(I)));
