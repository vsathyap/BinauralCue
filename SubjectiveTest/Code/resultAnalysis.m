% Returns statistics by analysing excel files in ResultsFull folder in the
% parent folder.
% statistics is seperate for different scenes and it is
% structured as statistics_sc_a = zeros(numb_algorithm,4), first column is
% mean, second is median, third is 0.25 quantile and fourth is 0.75
% quantile
% 
%%
clc
clear all
close all
folder = cd;
folder = erase(folder,'Code');
folder_results = 'ResultsFull';


filePattern = fullfile([folder folder_results], '*.xlsx');

xlsxfiles = dir(filePattern); 
numfiles = length(xlsxfiles);

signal_err_num = 8;
SceneA_Rep1_range = 'C3:K7';
SceneA_Rep2_range = 'C9:K13';
SceneB_Rep1_range = 'C16:K20';
SceneB_Rep2_range = 'C22:K26';
rep =2;
sheet_num = 1;
% Initialization for target included
% sc_a_bmvdr_jblcmv_err = zeros(5*numfiles*rep,2);
% sc_b_bmvdr_jblcmv_err = zeros(5*numfiles*rep,2);
% sc_a_rest_err = zeros(4*numfiles*rep,6);
% sc_b_rest_err = zeros(4*numfiles*rep,6);
% mean_unprocessed_sceneA=zeros(5,1);
% mean_unprocessed_sceneB=zeros(5,1);
% mean_unprocessed_vasudha_scA=zeros(4,1);
% mean_unprocessed_metin_scA=zeros(4,1);
% mean_unprocessed_vasudha_scB=zeros(4,1);
% mean_unprocessed_metin_scB=zeros(4,1);
% Initialization for target excluded
sc_a_bmvdr_jblcmv_err = zeros(4*numfiles*rep,2);
sc_b_bmvdr_jblcmv_err = zeros(4*numfiles*rep,2);
sc_a_rest_err = zeros(3*numfiles*rep,6);
sc_b_rest_err = zeros(3*numfiles*rep,6);
mean_unprocessed_sceneA=zeros(4,1);
mean_unprocessed_sceneB=zeros(4,1);
mean_unprocessed_vasudha_scA=zeros(3,1);
mean_unprocessed_metin_scA=zeros(3,1);
mean_unprocessed_vasudha_scB=zeros(3,1);
mean_unprocessed_metin_scB=zeros(3,1);


for i = 1:numfiles
    file_iter = fullfile(xlsxfiles(i).folder,xlsxfiles(i).name);
    sceneA_Rep1 = xlsread(file_iter,sheet_num,SceneA_Rep1_range);
    sceneA_Rep2 = xlsread(file_iter,sheet_num,SceneA_Rep2_range);
    sceneB_Rep1 = xlsread(file_iter,sheet_num,SceneB_Rep1_range);
    sceneB_Rep2 = xlsread(file_iter,sheet_num,SceneB_Rep2_range);
    
    sceneA_Rep1(4,7:9) = sceneA_Rep1(5,4:6);
    sceneA_Rep2(4,7:9) = sceneA_Rep2(5,4:6);
    sceneB_Rep1(4,7:9) = sceneB_Rep1(5,4:6);
    sceneB_Rep2(4,7:9) = sceneB_Rep2(5,4:6);
    
    unprocessed_index = 1;
    bmvdr_index = 2;
    jblcmv_index = 3;
    ild_index = 4;
    r_ild_1_index = 5;
    r_ild_2_index = 6;
    scaled_ild_0_2_index = 7;
    scaled_ild_0_6_index = 8;
    scaled_ild_1_index = 9;
    
    
%% With target included in the error calcualation
%     mean_unprocessed_sceneA = (sceneA_Rep1(:,unprocessed_index)+...
%         sceneA_Rep2(:,unprocessed_index))/2;
%     mean_unprocessed_sceneB = (sceneB_Rep1(:,unprocessed_index)+...
%         sceneB_Rep2(:,unprocessed_index))/2;
%     mean_unprocessed_vasudha_scA=(sceneA_Rep1(1:4,unprocessed_index)+...
%         sceneA_Rep2(1:4,unprocessed_index))/2;
%     mean_unprocessed_metin_scA(1:3)=(sceneA_Rep1(1:3,unprocessed_index)+...
%         sceneA_Rep2(1:3,unprocessed_index))/2;
%     mean_unprocessed_metin_scA(4) = (sceneA_Rep1(5,unprocessed_index)+...
%         sceneA_Rep2(5,unprocessed_index))/2;
%     mean_unprocessed_vasudha_scB=(sceneB_Rep1(1:4,unprocessed_index)+...
%         sceneB_Rep2(1:4,unprocessed_index))/2;
%     mean_unprocessed_metin_scB(1:3)=(sceneB_Rep1(1:3,unprocessed_index)+...
%         sceneB_Rep2(1:3,unprocessed_index))/2;
%     mean_unprocessed_metin_scB(4)=(sceneB_Rep1(5,unprocessed_index)+...
%         sceneB_Rep2(5,unprocessed_index))/2;
%     
%     sc_a_bmvdr_jblcmv_err(10*(i-1)+1:10*i,1) = [abs(mean_unprocessed_sceneA-sceneA_Rep1(:,bmvdr_index));...
%         abs(mean_unprocessed_sceneA-sceneA_Rep2(:,bmvdr_index))];
%     sc_a_bmvdr_jblcmv_err(10*(i-1)+1:10*i,2) = [abs(mean_unprocessed_sceneA-sceneA_Rep1(:,jblcmv_index));...
%         abs(mean_unprocessed_sceneA-sceneA_Rep2(:,jblcmv_index))];
%     sc_b_bmvdr_jblcmv_err(10*(i-1)+1:10*i,1)=[abs(mean_unprocessed_sceneB-sceneB_Rep1(:,bmvdr_index));...
%         abs(mean_unprocessed_sceneB-sceneB_Rep2(:,bmvdr_index))];
%     sc_b_bmvdr_jblcmv_err(10*(i-1)+1:10*i,2)=[abs(mean_unprocessed_sceneB-sceneB_Rep1(:,jblcmv_index));...
%         abs(mean_unprocessed_sceneB-sceneB_Rep2(:,jblcmv_index))];
%     sc_a_rest_err(8*(i-1)+1:8*i,1)=[abs(mean_unprocessed_vasudha_scA-sceneA_Rep1(1:4,ild_index));...
%         abs(mean_unprocessed_vasudha_scA-sceneA_Rep2(1:4,ild_index))];
%     sc_b_rest_err(8*(i-1)+1:8*i,1)=[abs(mean_unprocessed_vasudha_scB-sceneB_Rep1(1:4,ild_index));...
%         abs(mean_unprocessed_vasudha_scB-sceneB_Rep2(1:4,ild_index))];
%     sc_a_rest_err(8*(i-1)+1:8*i,2)=[abs(mean_unprocessed_vasudha_scA-sceneA_Rep1(1:4,r_ild_1_index));...
%         abs(mean_unprocessed_vasudha_scA-sceneA_Rep2(1:4,r_ild_1_index))];
%     sc_b_rest_err(8*(i-1)+1:8*i,2)=[abs(mean_unprocessed_vasudha_scB-sceneB_Rep1(1:4,r_ild_1_index));...
%         abs(mean_unprocessed_vasudha_scB-sceneB_Rep2(1:4,r_ild_1_index))];
%     
%     sc_a_rest_err(8*(i-1)+1:8*i,3)=[abs(mean_unprocessed_vasudha_scA-sceneA_Rep1(1:4,r_ild_2_index));...
%         abs(mean_unprocessed_vasudha_scA-sceneA_Rep2(1:4,r_ild_2_index))];
%     sc_b_rest_err(8*(i-1)+1:8*i,3)=[abs(mean_unprocessed_vasudha_scB-sceneB_Rep1(1:4,r_ild_2_index));...
%         abs(mean_unprocessed_vasudha_scB-sceneB_Rep2(1:4,r_ild_2_index))];
%     sc_a_rest_err(8*(i-1)+1:8*i,4)=[abs(mean_unprocessed_metin_scA-sceneA_Rep1(1:4,scaled_ild_0_2_index));...
%         abs(mean_unprocessed_metin_scA-sceneA_Rep2(1:4,scaled_ild_0_2_index))];
%     sc_b_rest_err(8*(i-1)+1:8*i,4)=[abs(mean_unprocessed_metin_scB-sceneB_Rep1(1:4,scaled_ild_0_2_index));...
%         abs(mean_unprocessed_metin_scB-sceneB_Rep2(1:4,scaled_ild_0_2_index))];
%     sc_a_rest_err(8*(i-1)+1:8*i,5)=[abs(mean_unprocessed_metin_scA-sceneA_Rep1(1:4,scaled_ild_0_6_index));...
%         abs(mean_unprocessed_metin_scA-sceneA_Rep2(1:4,scaled_ild_0_6_index))];
%     sc_b_rest_err(8*(i-1)+1:8*i,5)=[abs(mean_unprocessed_metin_scB-sceneB_Rep1(1:4,scaled_ild_0_6_index));...
%         abs(mean_unprocessed_metin_scB-sceneB_Rep2(1:4,scaled_ild_0_6_index))];
%     sc_a_rest_err(8*(i-1)+1:8*i,6)=[abs(mean_unprocessed_metin_scA-sceneA_Rep1(1:4,scaled_ild_1_index));...
%         abs(mean_unprocessed_metin_scA-sceneA_Rep2(1:4,scaled_ild_1_index))];
%     sc_b_rest_err(8*(i-1)+1:8*i,6)=[abs(mean_unprocessed_metin_scB-sceneB_Rep1(1:4,scaled_ild_1_index));...
%         abs(mean_unprocessed_metin_scB-sceneB_Rep2(1:4,scaled_ild_1_index))];

%% Without Target included in the error calculation
mean_unprocessed_sceneA = (sceneA_Rep1(2:end,unprocessed_index)+...
    sceneA_Rep2(2:end,unprocessed_index))/2;
mean_unprocessed_sceneB = (sceneB_Rep1(2:end,unprocessed_index)+...
    sceneB_Rep2(2:end,unprocessed_index))/2;
mean_unprocessed_vasudha_scA=(sceneA_Rep1(2:4,unprocessed_index)+...
    sceneA_Rep2(2:4,unprocessed_index))/2;
mean_unprocessed_metin_scA(1:2)=(sceneA_Rep1(2:3,unprocessed_index)+...
    sceneA_Rep2(2:3,unprocessed_index))/2;
mean_unprocessed_metin_scA(3) = (sceneA_Rep1(5,unprocessed_index)+...
    sceneA_Rep2(5,unprocessed_index))/2;
mean_unprocessed_vasudha_scB=(sceneB_Rep1(2:4,unprocessed_index)+...
    sceneB_Rep2(2:4,unprocessed_index))/2;
mean_unprocessed_metin_scB(1:2)=(sceneB_Rep1(2:3,unprocessed_index)+...
    sceneB_Rep2(2:3,unprocessed_index))/2;
mean_unprocessed_metin_scB(3)=(sceneB_Rep1(5,unprocessed_index)+...
    sceneB_Rep2(5,unprocessed_index))/2;



sc_a_bmvdr_jblcmv_err(8*(i-1)+1:8*i,1) = [abs(mean_unprocessed_sceneA-sceneA_Rep1(2:end,bmvdr_index));...
    abs(mean_unprocessed_sceneA-sceneA_Rep2(2:end,bmvdr_index))];
sc_a_bmvdr_jblcmv_err(8*(i-1)+1:8*i,2) = [abs(mean_unprocessed_sceneA-sceneA_Rep1(2:end,jblcmv_index));...
    abs(mean_unprocessed_sceneA-sceneA_Rep2(2:end,jblcmv_index))];
sc_b_bmvdr_jblcmv_err(8*(i-1)+1:8*i,1)=[abs(mean_unprocessed_sceneB-sceneB_Rep1(2:end,bmvdr_index));...
    abs(mean_unprocessed_sceneB-sceneB_Rep2(2:end,bmvdr_index))];
sc_b_bmvdr_jblcmv_err(8*(i-1)+1:8*i,2)=[abs(mean_unprocessed_sceneB-sceneB_Rep1(2:end,jblcmv_index));...
    abs(mean_unprocessed_sceneB-sceneB_Rep2(2:end,jblcmv_index))];
sc_a_rest_err(6*(i-1)+1:6*i,1)=[abs(mean_unprocessed_vasudha_scA-sceneA_Rep1(2:4,ild_index));...
    abs(mean_unprocessed_vasudha_scA-sceneA_Rep2(2:4,ild_index))];
sc_b_rest_err(6*(i-1)+1:6*i,1)=[abs(mean_unprocessed_vasudha_scB-sceneB_Rep1(2:4,ild_index));...
    abs(mean_unprocessed_vasudha_scB-sceneB_Rep2(2:4,ild_index))];
sc_a_rest_err(6*(i-1)+1:6*i,2)=[abs(mean_unprocessed_vasudha_scA-sceneA_Rep1(2:4,r_ild_1_index));...
    abs(mean_unprocessed_vasudha_scA-sceneA_Rep2(2:4,r_ild_1_index))];
sc_b_rest_err(6*(i-1)+1:6*i,2)=[abs(mean_unprocessed_vasudha_scB-sceneB_Rep1(2:4,r_ild_1_index));...
    abs(mean_unprocessed_vasudha_scB-sceneB_Rep2(2:4,r_ild_1_index))];

sc_a_rest_err(6*(i-1)+1:6*i,3)=[abs(mean_unprocessed_vasudha_scA-sceneA_Rep1(2:4,r_ild_2_index));...
    abs(mean_unprocessed_vasudha_scA-sceneA_Rep2(2:4,r_ild_2_index))];
sc_b_rest_err(6*(i-1)+1:6*i,3)=[abs(mean_unprocessed_vasudha_scB-sceneB_Rep1(2:4,r_ild_2_index));...
    abs(mean_unprocessed_vasudha_scB-sceneB_Rep2(2:4,r_ild_2_index))];
sc_a_rest_err(6*(i-1)+1:6*i,4)=[abs(mean_unprocessed_metin_scA-sceneA_Rep1(2:4,scaled_ild_0_2_index));...
    abs(mean_unprocessed_metin_scA-sceneA_Rep2(2:4,scaled_ild_0_2_index))];
sc_b_rest_err(6*(i-1)+1:6*i,4)=[abs(mean_unprocessed_metin_scB-sceneB_Rep1(2:4,scaled_ild_0_2_index));...
    abs(mean_unprocessed_metin_scB-sceneB_Rep2(2:4,scaled_ild_0_2_index))];
sc_a_rest_err(6*(i-1)+1:6*i,5)=[abs(mean_unprocessed_metin_scA-sceneA_Rep1(2:4,scaled_ild_0_6_index));...
    abs(mean_unprocessed_metin_scA-sceneA_Rep2(2:4,scaled_ild_0_6_index))];
sc_b_rest_err(6*(i-1)+1:6*i,5)=[abs(mean_unprocessed_metin_scB-sceneB_Rep1(2:4,scaled_ild_0_6_index));...
    abs(mean_unprocessed_metin_scB-sceneB_Rep2(2:4,scaled_ild_0_6_index))];
sc_a_rest_err(6*(i-1)+1:6*i,6)=[abs(mean_unprocessed_metin_scA-sceneA_Rep1(2:4,scaled_ild_1_index));...
    abs(mean_unprocessed_metin_scA-sceneA_Rep2(2:4,scaled_ild_1_index))];
sc_b_rest_err(6*(i-1)+1:6*i,6)=[abs(mean_unprocessed_metin_scB-sceneB_Rep1(2:4,scaled_ild_1_index));...
    abs(mean_unprocessed_metin_scB-sceneB_Rep2(2:4,scaled_ild_1_index))];

    
    
end

statistic_sc_a = [mean(sc_a_bmvdr_jblcmv_err(:,1)) median(sc_a_bmvdr_jblcmv_err(:,1)) ...
    quantile(sc_a_bmvdr_jblcmv_err(:,1),0.25) quantile(sc_a_bmvdr_jblcmv_err(:,1),0.75);
    mean(sc_a_bmvdr_jblcmv_err(:,2)) median(sc_a_bmvdr_jblcmv_err(:,2)) ...
    quantile(sc_a_bmvdr_jblcmv_err(:,2),0.25) quantile(sc_a_bmvdr_jblcmv_err(:,2),0.75);
    mean(sc_a_rest_err(:,1)) median(sc_a_rest_err(:,1)) ...
    quantile(sc_a_rest_err(:,1),0.25) quantile(sc_a_rest_err(:,1),0.75);
    mean(sc_a_rest_err(:,2)) median(sc_a_rest_err(:,2)) ...
    quantile(sc_a_rest_err(:,2),0.25) quantile(sc_a_rest_err(:,2),0.75)
    mean(sc_a_rest_err(:,3)) median(sc_a_rest_err(:,3)) ...
    quantile(sc_a_rest_err(:,3),0.25) quantile(sc_a_rest_err(:,3),0.75)
    mean(sc_a_rest_err(:,4)) median(sc_a_rest_err(:,4)) ...
    quantile(sc_a_rest_err(:,4),0.25) quantile(sc_a_rest_err(:,4),0.75)
    mean(sc_a_rest_err(:,5)) median(sc_a_rest_err(:,5)) ...
    quantile(sc_a_rest_err(:,5),0.25) quantile(sc_a_rest_err(:,5),0.75)
    mean(sc_a_rest_err(:,6)) median(sc_a_rest_err(:,6)) ...
    quantile(sc_a_rest_err(:,6),0.25) quantile(sc_a_rest_err(:,6),0.75)
    ];
statistic_sc_b = [mean(sc_b_bmvdr_jblcmv_err(:,1)) median(sc_b_bmvdr_jblcmv_err(:,1)) ...
    quantile(sc_b_bmvdr_jblcmv_err(:,1),0.25) quantile(sc_b_bmvdr_jblcmv_err(:,1),0.75);
    mean(sc_b_bmvdr_jblcmv_err(:,2)) median(sc_b_bmvdr_jblcmv_err(:,2)) ...
    quantile(sc_b_bmvdr_jblcmv_err(:,2),0.25) quantile(sc_b_bmvdr_jblcmv_err(:,2),0.75);
    mean(sc_b_rest_err(:,1)) median(sc_b_rest_err(:,1)) ...
    quantile(sc_b_rest_err(:,1),0.25) quantile(sc_b_rest_err(:,1),0.75);
    mean(sc_b_rest_err(:,2)) median(sc_b_rest_err(:,2)) ...
    quantile(sc_b_rest_err(:,2),0.25) quantile(sc_b_rest_err(:,2),0.75)
    mean(sc_b_rest_err(:,3)) median(sc_b_rest_err(:,3)) ...
    quantile(sc_b_rest_err(:,3),0.25) quantile(sc_b_rest_err(:,3),0.75)
    mean(sc_b_rest_err(:,4)) median(sc_b_rest_err(:,4)) ...
    quantile(sc_b_rest_err(:,4),0.25) quantile(sc_b_rest_err(:,4),0.75)
    mean(sc_b_rest_err(:,5)) median(sc_b_rest_err(:,5)) ...
    quantile(sc_b_rest_err(:,5),0.25) quantile(sc_b_rest_err(:,5),0.75)
    mean(sc_b_rest_err(:,6)) median(sc_b_rest_err(:,6)) ...
    quantile(sc_b_rest_err(:,6),0.25) quantile(sc_b_rest_err(:,6),0.75)
    ];
number_of_algorithms=8;
algoNames = {'BMVDR','JBLCMV','ILD','R\_ILD\_1','R\_ILD\_2',...
    'ILD_{0.2}','ILD_{0.6}','ILD_1'};

txt1 = {'Scene 1'};
txt2= {'Scene 2'};


subplot(2,1,1)
plot(1:number_of_algorithms,statistic_sc_a(:,1),'go','MarkerSize',7,'LineWidth',1)
hold on
plot(1:number_of_algorithms,statistic_sc_a(:,2),'rs','MarkerSize',7,'LineWidth',1)
hold on
errorbar(1:number_of_algorithms, statistic_sc_a(:,2), statistic_sc_a(:,3),...
    statistic_sc_a(:,4),'LineStyle','none');
xlim([0,9])
set(gca, 'XTick', 1:number_of_algorithms, 'XTickLabel', algoNames)
ylabel('localization error (degrees)');
ylim([0 150]);
grid on
legend('mean','median')
text(0,130,txt1,'bold','FontSize',14)
subplot(2,1,2)
plot(1:number_of_algorithms,statistic_sc_b(:,1),'go','MarkerSize',7,'LineWidth',1)
hold on
plot(1:number_of_algorithms,statistic_sc_b(:,2),'rs','MarkerSize',7,'LineWidth',1)
hold on
errorbar(1:number_of_algorithms, statistic_sc_b(:,2), statistic_sc_b(:,3),...
    statistic_sc_b(:,4),'LineStyle','none');
xlim([0,9])
set(gca, 'XTick', 1:number_of_algorithms, 'XTickLabel', algoNames)
ylabel('localization error (degrees)');
ylim([0 150]);
grid on
legend('mean','median')
text(0,130,txt2,'bold','FontSize',14)

