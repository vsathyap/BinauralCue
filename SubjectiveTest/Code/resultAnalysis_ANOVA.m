% Returns statistics by analysing excel files in ResultsFull folder in the
% parent folder.
% Two way ANOVa test - 5 columns(for 5 methods), 4 rows(for 4 sources),
% with 18 subjects (The repetitions are averaged for each subject)
% 
%%
clc
clear all
close all
folder = cd;
folder = erase(folder,'\Code');
folder_results = 'ResultsFull';


filePattern = fullfile([folder '\' folder_results], '*.xlsx');

xlsxfiles = dir(filePattern); 
numfiles = length(xlsxfiles);
%Ranges from excel sheet
SceneA_Rep1_range = 'C3:K7';
SceneA_Rep2_range = 'C9:K13';
SceneB_Rep1_range = 'C16:K20';
SceneB_Rep2_range = 'C22:K26';
rep =2;
sheet_num = 1;
num_algorithms = 6; %including unprocessed(for each Vasudha and Metin individually)

% num_sources = 4; %including target
% % Initialization for target included
% sc_a_vasudha = zeros(4,6,numfiles,rep);
% sc_b_vasudha = zeros(4,6,numfiles,rep);
% sc_a_metin = zeros(4,6,numfiles,rep);
% sc_b_metin = zeros(4,6,numfiles,rep);
% 
% mean_unprocessed_sceneA = zeros(4,1);
% mean_unprocessed_sceneB = zeros(4,1);

num_sources = 3; %excluding target
% Initialization for target not included
sc_a_vasudha = zeros(num_sources,6,numfiles,rep);
sc_b_vasudha = zeros(num_sources,6,numfiles,rep);
sc_a_metin = zeros(num_sources,6,numfiles,rep);
sc_b_metin = zeros(num_sources,6,numfiles,rep);

mean_unprocessed_sceneA = zeros(num_sources,1);
mean_unprocessed_sceneB = zeros(num_sources,1);


for i = 1:numfiles
    file_iter = fullfile(xlsxfiles(i).folder,xlsxfiles(i).name);
    sceneA_Rep1 = xlsread(file_iter,sheet_num,SceneA_Rep1_range);
    sceneA_Rep2 = xlsread(file_iter,sheet_num,SceneA_Rep2_range);
    sceneB_Rep1 = xlsread(file_iter,sheet_num,SceneB_Rep1_range);
    sceneB_Rep2 = xlsread(file_iter,sheet_num,SceneB_Rep2_range);
    
    sc_a_vasudha(:,1:6,i,1) = sceneA_Rep1(2:4,1:6);
    sc_a_vasudha(:,1:6,i,2) = sceneA_Rep2(2:4,1:6);
    mean_unprocessed_sceneA = 0.5*(sc_a_vasudha(:,1,i,1) + sc_a_vasudha(:,1,i,2));
    sc_b_vasudha(:,1:6,i,1) = sceneB_Rep1(2:4,1:6);
    sc_b_vasudha(:,1:6,i,2) = sceneB_Rep2(2:4,1:6);
    mean_unprocessed_sceneB = 0.5*(sc_b_vasudha(:,1,i,1) + sc_b_vasudha(:,1,i,2));
    % Removing the reference direction from each algortihm, to reduce error
    sc_a_vasudha(:,1:6,i,1) = abs(sc_a_vasudha(:,1:6,i,1) - mean_unprocessed_sceneA);
    sc_a_vasudha(:,1:6,i,2) = abs(sc_a_vasudha(:,1:6,i,2) - mean_unprocessed_sceneA);
    sc_b_vasudha(:,1:6,i,1) = abs(sc_b_vasudha(:,1:6,i,1) - mean_unprocessed_sceneB);
    sc_b_vasudha(:,1:6,i,2) = abs(sc_b_vasudha(:,1:6,i,2) - mean_unprocessed_sceneB);
    
%     %Moving scaledILD's to column 7:9 for uniformity
%     sceneA_Rep1(5,7:9) = sceneA_Rep1(5,4:6);
%     sceneA_Rep2(5,7:9) = sceneA_Rep2(5,4:6);
%     sceneB_Rep1(5,7:9) = sceneB_Rep1(5,4:6);
%     sceneB_Rep2(5,7:9) = sceneB_Rep2(5,4:6);
%     
%     sc_a_metin(:,1:6,i,1) = [sceneA_Rep1(1:3,1:3),sceneA_Rep1(1:3,7:9);...
%         sceneA_Rep1(5,1:3),sceneA_Rep1(5,7:9)];
%     sc_a_metin(:,1:6,i,2) = [sceneA_Rep2(1:3,1:3),sceneA_Rep2(1:3,7:9);...
%         sceneA_Rep2(5,1:3),sceneA_Rep2(5,7:9)];
%     mean_unprocessed_sceneA = 0.5*(sc_a_metin(:,1,i,1) + sc_a_metin(:,1,i,2));
%     sc_b_metin(:,1:6,i,1) = [sceneB_Rep1(1:3,1:3),sceneB_Rep1(1:3,7:9);...
%         sceneB_Rep1(5,1:3),sceneB_Rep1(5,7:9)];
%     sc_b_metin(:,1:6,i,2) = [sceneB_Rep2(1:3,1:3),sceneB_Rep2(1:3,7:9);...
%         sceneB_Rep2(5,1:3),sceneB_Rep2(5,7:9)];
%     mean_unprocessed_sceneB = 0.5*(sc_b_metin(:,1,i,1) + sc_b_metin(:,1,i,2));
%     % Removing the reference direction from each algortihm, to reduce error
%     sc_a_metin(:,1:6,i,1) = abs(sc_a_metin(:,1:6,i,1) - mean_unprocessed_sceneA);
%     sc_b_metin(:,1:6,i,1) = abs(sc_b_metin(:,1:6,i,1) - mean_unprocessed_sceneB);
%     sc_a_metin(:,1:6,i,2) = abs(sc_a_metin(:,1:6,i,2) - mean_unprocessed_sceneA);
%     sc_b_metin(:,1:6,i,2) = abs(sc_b_metin(:,1:6,i,2) - mean_unprocessed_sceneB);
%         
end

%Average the repetitions and permute the dimensions
sc_a_vasudha = permute(mean(sc_a_vasudha,4),[3 2 1]);
sc_b_vasudha = permute(mean(sc_b_vasudha,4),[3 2 1]);

% sc_a_metin = permute(mean(sc_a_metin, 4),[3 2 1]);
% sc_b_metin = permute(mean(sc_b_metin,4),[3 2 1]);

% Arranging the data for the ANOVA test
numMethods = 4;%Without the unprocessed case and bmvdr
numSources = 3;
numSubjects = numfiles;
for i = 1:numSources
    sc_A_V(numSubjects*(i-1)+1:numSubjects*(i-1)+numSubjects,:) = squeeze(sc_a_vasudha(:,3:6,i));
    sc_B_V(numSubjects*(i-1)+1:numSubjects*(i-1)+numSubjects,:) = squeeze(sc_b_vasudha(:,3:6,i));
%     sc_A_M(numSubjects*(i-1)+1:numSubjects*(i-1)+numSubjects,:) = squeeze(sc_a_metin(:,3:6,i));
%     sc_B_M(numSubjects*(i-1)+1:numSubjects*(i-1)+numSubjects,:) = squeeze(sc_b_metin(:,3:6,i));
end

[~,~,statsA] = anova2(sc_A_V,numSubjects);
[~,~,statsB] = anova2(sc_B_V,numSubjects);

