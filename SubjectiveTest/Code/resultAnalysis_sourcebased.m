% Returns statistics by analysing excel files in ResultsFull folder in the
% parent folder.
% statistics measured per source and algorithm.(mean,median,0,25 quartile,
% 0,75 quartile)
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
num_sources = 4; %including target

% Initialization for target included
sc_a_vasudha = zeros(4,6,numfiles,rep);
sc_b_vasudha = zeros(4,6,numfiles,rep);
sc_a_metin = zeros(4,6,numfiles,rep);
sc_b_metin = zeros(4,6,numfiles,rep);

mean_unprocessed_sceneA = zeros(4,1);
mean_unprocessed_sceneB = zeros(4,1);

%statisctics for each scene, each algorithm(ignoring unprocessed), for each source
statistics_a_vasudha = zeros(num_sources,num_algorithms-1,4);
statistics_b_vasudha = zeros(num_sources,num_algorithms-1,4);
statistics_a_metin = zeros(num_sources,num_algorithms-1,4);
statistics_b_metin = zeros(num_sources,num_algorithms-1,4);

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

%To combine the repetitions and subjects into a vector
sc_a_vasudha = sc_a_vasudha(1:4,1:6,:);
sc_b_vasudha = sc_b_vasudha(1:4,1:6,:);

sc_a_metin = sc_a_metin(1:4,1:6,:);
sc_b_metin = sc_b_metin(1:4,1:6,:);

for alg = 2:num_algorithms
    for src = 1:num_sources
        statistics_a_vasudha(src,alg-1,:) =[mean(sc_a_vasudha(src,alg,:)) median(sc_a_vasudha(src,alg,:)) ...
            quantile(sc_a_vasudha(src,alg,:),0.25) quantile(sc_a_vasudha(src,alg,:),0.75)];
        statistics_b_vasudha(src,alg-1,:) =[mean(sc_b_vasudha(src,alg,:)) median(sc_b_vasudha(src,alg,:)) ...
            quantile(sc_b_vasudha(src,alg,:),0.25) quantile(sc_b_vasudha(src,alg,:),0.75)];
        
        statistics_a_metin(src,alg-1,:) =[mean(sc_a_metin(src,alg,:)) median(sc_a_metin(src,alg,:)) ...
            quantile(sc_a_metin(src,alg,:),0.25) quantile(sc_a_metin(src,alg,:),0.75)];
        statistics_b_metin(src,alg-1,:) =[mean(sc_b_metin(src,alg,:)) median(sc_b_metin(src,alg,:)) ...
            quantile(sc_b_metin(src,alg,:),0.25) quantile(sc_b_metin(src,alg,:),0.75)];
    end
end

%Plots

algoNames_v = {'BMVDR','JBLCMV','ILD','R\_ILD\_1','R\_ILD\_2'};
algoNames_m = {'BMVDR','JBLCMV','ILD_{0.2}','ILD_{0.6}','ILD_1'};

srcNames_v = {'Female Speech','Male Speech','Music','HF Signal'};
srcNames_m = {'Female Speech','Male Speech','Music','LF Signal'};

txt1 = {'Scene 1'};
txt2= {'Scene 2'};

f1 = figure;
sgtitle(txt1,'FontSize',14);
for src=1:num_sources
    subplot(2,2,src)
    plot(1:num_algorithms-1,statistics_a_vasudha(src,:,1),'go','MarkerSize',7,'LineWidth',1)
    hold on
    plot(1:num_algorithms-1,statistics_a_vasudha(src,:,2),'rs','MarkerSize',7,'LineWidth',1)
    hold on
    errorbar(1:num_algorithms-1, statistics_a_vasudha(src,:,2), statistics_a_vasudha(src,:,3),...
        statistics_a_vasudha(src,:,4),'LineStyle','none');
    xlim([0,6])
    set(gca, 'XTick', 1:num_algorithms-1, 'XTickLabel', algoNames_v)
    ylabel('localization error (degrees)');
    ylim([0 200]);
    grid on
    title(srcNames_v{src});
    xtickangle(45);
    legend('mean','median')
end

f2 = figure;
sgtitle(txt2,'FontSize',14);
for src=1:num_sources
    subplot(2,2,src)
    plot(1:num_algorithms-1,statistics_b_vasudha(src,:,1),'go','MarkerSize',7,'LineWidth',1)
    hold on
    plot(1:num_algorithms-1,statistics_b_vasudha(src,:,2),'rs','MarkerSize',7,'LineWidth',1)
    hold on
    errorbar(1:num_algorithms-1, statistics_b_vasudha(src,:,2), statistics_b_vasudha(src,:,3),...
        statistics_b_vasudha(src,:,4),'LineStyle','none');
    xlim([0,6])
    set(gca, 'XTick', 1:num_algorithms-1, 'XTickLabel', algoNames_v)
    ylabel('localization error (degrees)');
    ylim([0 200]);
    grid on
    title(srcNames_v{src});
    xtickangle(45);
    legend('mean','median')
end

%Plots for Metin
f3 = figure;
sgtitle(txt1,'FontSize',14);
for src=1:num_sources
    subplot(2,2,src)
    plot(1:num_algorithms-1,statistics_a_metin(src,:,1),'go','MarkerSize',7,'LineWidth',1)
    hold on
    plot(1:num_algorithms-1,statistics_a_metin(src,:,2),'rs','MarkerSize',7,'LineWidth',1)
    hold on
    errorbar(1:num_algorithms-1, statistics_a_metin(src,:,2), statistics_a_metin(src,:,3),...
        statistics_a_metin(src,:,4),'LineStyle','none');
    xlim([0,6])
    set(gca, 'XTick', 1:num_algorithms-1, 'XTickLabel', algoNames_m)
    ylabel('localization error (degrees)');
    ylim([0 200]);
    grid on
    title(srcNames_m{src});
    xtickangle(45);
    legend('mean','median')
end

f4 = figure;
sgtitle(txt2,'FontSize',14);
for src=1:num_sources
    subplot(2,2,src)
    plot(1:num_algorithms-1,statistics_b_metin(src,:,1),'go','MarkerSize',7,'LineWidth',1)
    hold on
    plot(1:num_algorithms-1,statistics_b_metin(src,:,2),'rs','MarkerSize',7,'LineWidth',1)
    hold on
    errorbar(1:num_algorithms-1, statistics_b_metin(src,:,2), statistics_b_metin(src,:,3),...
        statistics_b_metin(src,:,4),'LineStyle','none');
    xlim([0,6])
    set(gca, 'XTick', 1:num_algorithms-1, 'XTickLabel', algoNames_m)
    ylabel('localization error (degrees)');
    ylim([0 200]);
    grid on
    title(srcNames_m{src});
    xtickangle(45);
    legend('mean','median')
end


