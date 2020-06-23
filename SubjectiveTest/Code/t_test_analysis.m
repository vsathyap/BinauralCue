% Returns statistics by analysing excel files in ResultsFull folder in the
% parent folder.
%the t-test and anova is done here.
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

% Initialization for target included
sc_a_vasudha = zeros(4,6,numfiles,rep);
sc_b_vasudha = zeros(4,6,numfiles,rep);
sc_a_metin = zeros(4,6,numfiles,rep);
sc_b_metin = zeros(4,6,numfiles,rep);

mean_unprocessed_sceneA = zeros(4,1);
mean_unprocessed_sceneB = zeros(4,1);

%t-test results when comapring the 3 proposed methods against all the 5
%methods for scene A and B
h_a_vasudha = zeros(3,num_algorithms-1);
h_b_vasudha = zeros(3,num_algorithms-1);
p_a_vasudha = zeros(3,num_algorithms-1);
p_b_vasudha = zeros(3,num_algorithms-1);
h_a_metin = zeros(3,num_algorithms-1);
h_b_metin = zeros(3,num_algorithms-1);
p_a_metin = zeros(3,num_algorithms-1);
p_b_metin = zeros(3,num_algorithms-1);

for i = 1:numfiles
    file_iter = fullfile(xlsxfiles(i).folder,xlsxfiles(i).name);
    sceneA_Rep1 = xlsread(file_iter,sheet_num,SceneA_Rep1_range);
    sceneA_Rep2 = xlsread(file_iter,sheet_num,SceneA_Rep2_range);
    sceneB_Rep1 = xlsread(file_iter,sheet_num,SceneB_Rep1_range);
    sceneB_Rep2 = xlsread(file_iter,sheet_num,SceneB_Rep2_range);
    
    sc_a_vasudha(:,1:6,i,1) = sceneA_Rep1(1:4,1:6);
    sc_a_vasudha(:,1:6,i,2) = sceneA_Rep2(1:4,1:6);
    mean_unprocessed_sceneA = 0.5*(sc_a_vasudha(:,1,i,1) + sc_a_vasudha(:,1,i,2));
    sc_b_vasudha(:,1:6,i,1) = sceneB_Rep1(1:4,1:6);
    sc_b_vasudha(:,1:6,i,2) = sceneB_Rep2(1:4,1:6);
    mean_unprocessed_sceneB = 0.5*(sc_b_vasudha(:,1,i,1) + sc_b_vasudha(:,1,i,2));
    % Removing the reference direction from each algortihm, to reduce error
    sc_a_vasudha(:,1:6,i,1) = abs(sc_a_vasudha(:,1:6,i,1) - mean_unprocessed_sceneA);
    sc_a_vasudha(:,1:6,i,2) = abs(sc_a_vasudha(:,1:6,i,2) - mean_unprocessed_sceneA);
    sc_b_vasudha(:,1:6,i,1) = abs(sc_b_vasudha(:,1:6,i,1) - mean_unprocessed_sceneB);
    sc_b_vasudha(:,1:6,i,2) = abs(sc_b_vasudha(:,1:6,i,2) - mean_unprocessed_sceneB);
    
    %Moving scaledILD's to column 7:9 for uniformity
    sceneA_Rep1(5,7:9) = sceneA_Rep1(5,4:6);
    sceneA_Rep2(5,7:9) = sceneA_Rep2(5,4:6);
    sceneB_Rep1(5,7:9) = sceneB_Rep1(5,4:6);
    sceneB_Rep2(5,7:9) = sceneB_Rep2(5,4:6);
    
    sc_a_metin(:,1:6,i,1) = [sceneA_Rep1(1:3,1:3),sceneA_Rep1(1:3,7:9);...
        sceneA_Rep1(5,1:3),sceneA_Rep1(5,7:9)];
    sc_a_metin(:,1:6,i,2) = [sceneA_Rep2(1:3,1:3),sceneA_Rep2(1:3,7:9);...
        sceneA_Rep2(5,1:3),sceneA_Rep2(5,7:9)];
    mean_unprocessed_sceneA = 0.5*(sc_a_metin(:,1,i,1) + sc_a_metin(:,1,i,2));
    sc_b_metin(:,1:6,i,1) = [sceneB_Rep1(1:3,1:3),sceneB_Rep1(1:3,7:9);...
        sceneB_Rep1(5,1:3),sceneB_Rep1(5,7:9)];
    sc_b_metin(:,1:6,i,2) = [sceneB_Rep2(1:3,1:3),sceneB_Rep2(1:3,7:9);...
        sceneB_Rep2(5,1:3),sceneB_Rep2(5,7:9)];
    mean_unprocessed_sceneB = 0.5*(sc_b_metin(:,1,i,1) + sc_b_metin(:,1,i,2));
    % Removing the reference direction from each algortihm, to reduce error
    sc_a_metin(:,1:6,i,1) = abs(sc_a_metin(:,1:6,i,1) - mean_unprocessed_sceneA);
    sc_b_metin(:,1:6,i,1) = abs(sc_b_metin(:,1:6,i,1) - mean_unprocessed_sceneB);
    sc_a_metin(:,1:6,i,2) = abs(sc_a_metin(:,1:6,i,2) - mean_unprocessed_sceneA);
    sc_b_metin(:,1:6,i,2) = abs(sc_b_metin(:,1:6,i,2) - mean_unprocessed_sceneB);
   
       
end
%Averaging across repetitions and and sources
sc_a_vasudha = squeeze(mean(sc_a_vasudha,[4,1]));
sc_b_vasudha = squeeze(mean(sc_b_vasudha,[4,1]));

sc_a_metin = squeeze(mean(sc_a_metin,[4,1]));
sc_b_metin = squeeze(mean(sc_b_metin,[4,1]));

%Doing the two tailed t-test, getting the p-values
%Rows: P-ILd,R_ILD_1,R_ILD_2
%Columns: BMVDR, JBLCMV, P-ILD,R-ILD_1,R_ILD_2
for alg = 2:num_algorithms
    for i = 4:6
    [h_a_vasudha(i-3, alg-1), p_a_vasudha(i-3, alg-1)] = ttest(sc_a_vasudha(alg,:),sc_a_vasudha(i,:));
    [h_b_vasudha(i-3, alg-1), p_b_vasudha(i-3, alg-1)] = ttest(sc_b_vasudha(alg,:),sc_b_vasudha(i,:));
    end   
end

%To remove 'NaN'
h_a_vasudha(isnan(h_a_vasudha)) = 1;
h_a_vasudha(isnan(h_a_vasudha)) = 1;
p_a_vasudha(isnan(p_a_vasudha)) = 1;
p_b_vasudha(isnan(p_b_vasudha)) = 1;

%Doing the two tailed t-test
%Rows: scaledILD_0.2,scaledILD_0.6,scaledILD_1
%Columns: BMVDR, JBLCMV,scaledILD_0.2,scaledILD_0.6,scaledILD_1
for alg = 2:num_algorithms
    for i = 4:6
    [h_a_metin(i-3, alg-1), p_a_metin(i-3, alg-1)] = ttest(sc_a_vasudha(alg,:),sc_a_vasudha(i,:));
    [h_b_metin(i-3, alg-1), p_b_metin(i-3, alg-1)] = ttest(sc_b_vasudha(alg,:),sc_b_vasudha(i,:));
    end   
end

%To remove 'NaN'
h_a_metin(isnan(h_a_metin)) = 1;
h_a_metin(isnan(h_a_metin)) = 1;
p_a_metin(isnan(p_a_metin)) = 1;
p_b_metin(isnan(p_b_metin)) = 1;

% %To display a table
% fA_vasudha = figure;
% subplot(2,1,1);
% col = {'Unprocessed','BMVDR','JBLCMV','P-ILD','R-ILD_1','R-ILD_2'};
% row = {'P-ILD','R-ILD_1','R-ILD_2'};
% tA=uitable(fA_vasudha,'data',p_a_vasudha,'columnname',col,'RowName',row);
% 
% subplot(2,1,2);
% col = {'Unprocessed','BMVDR','JBLCMV','P-ILD','R-ILD_1','R-ILD_2'};
% row = {'P-ILD','R-ILD_1','R-ILD_2'};
% tB=uitable(fA_vasudha,'data',p_b_vasudha,'columnname',col,'RowName',row);
