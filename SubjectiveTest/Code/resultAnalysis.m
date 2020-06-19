%%
% ResultsFull folder with all the results available
% should be in the path
% Returns loc_err and statistics where loc_err is structures as
% loc_err = zeros(numfiles*3,signal_err_num)
% first three rows are for the first result file,
% second three rows are the second result file etc.
% first column is bmvdr, second is jblcmv, third is
% ild, fourth is r_ild_1, fifth is r_ild_2, sixth is
% scaled_ild 0_2, seventh is 0.6 and eight is 1.
% statistics is seperate for different scenes and it is
% structured as statistics_sc_a = zeros(numb)algorithm,4), first column is
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
mean_err = zeros(numfiles*10,signal_err_num);%3 because sc1, sc2, total err
sc_a_bmvdr_jblcmv_err = zeros(5*numfiles*rep,2);
sc_b_bmvdr_jblcmv_err = zeros(5*numfiles*rep,2);

sc_a_rest_err = zeros(4*numfiles*rep,6);
sc_b_rest_err = zeros(4*numfiles*rep,6);

mean_unprocessed_sceneA=zeros(5,1);
mean_unprocessed_sceneB=zeros(5,1);
mean_unprocessed_vasudha_scA=zeros(4,1);
mean_unprocessed_metin_scA=zeros(4,1);
mean_unprocessed_vasudha_scB=zeros(4,1);
mean_unprocessed_metin_scB=zeros(4,1);

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
    
    mean_unprocessed_sceneA = (sceneA_Rep1(:,unprocessed_index)+...
        sceneA_Rep2(:,unprocessed_index))/2;
    mean_unprocessed_sceneB = (sceneB_Rep1(:,unprocessed_index)+...
        sceneB_Rep2(:,unprocessed_index))/2;
    mean_unprocessed_vasudha_scA=(sceneA_Rep1(1:4,unprocessed_index)+...
        sceneA_Rep2(1:4,unprocessed_index))/2;
    mean_unprocessed_metin_scA(1:3)=(sceneA_Rep1(1:3,unprocessed_index)+...
        sceneA_Rep2(1:3,unprocessed_index))/2;
    mean_unprocessed_metin_scA(4) = (sceneA_Rep1(5,unprocessed_index)+...
        sceneA_Rep2(5,unprocessed_index))/2;
    mean_unprocessed_vasudha_scB=(sceneB_Rep1(1:4,unprocessed_index)+...
        sceneB_Rep2(1:4,unprocessed_index))/2;
    mean_unprocessed_metin_scB(1:3)=(sceneB_Rep1(1:3,unprocessed_index)+...
        sceneB_Rep2(1:3,unprocessed_index))/2;
    mean_unprocessed_metin_scB(4)=(sceneB_Rep1(5,unprocessed_index)+...
        sceneB_Rep2(5,unprocessed_index))/2;
    
    

    
    sc_a_bmvdr_jblcmv_err(10*(i-1)+1:10*i,1) = [abs(mean_unprocessed_sceneA-sceneA_Rep1(:,bmvdr_index));...
        abs(mean_unprocessed_sceneA-sceneA_Rep2(:,bmvdr_index))];
    sc_a_bmvdr_jblcmv_err(10*(i-1)+1:10*i,2) = [abs(mean_unprocessed_sceneA-sceneA_Rep1(:,jblcmv_index));...
        abs(mean_unprocessed_sceneA-sceneA_Rep2(:,jblcmv_index))];
    sc_b_bmvdr_jblcmv_err(10*(i-1)+1:10*i,1)=[abs(mean_unprocessed_sceneB-sceneB_Rep1(:,bmvdr_index));...
        abs(mean_unprocessed_sceneB-sceneB_Rep2(:,bmvdr_index))];
    sc_b_bmvdr_jblcmv_err(10*(i-1)+1:10*i,2)=[abs(mean_unprocessed_sceneB-sceneB_Rep1(:,jblcmv_index));...
        abs(mean_unprocessed_sceneB-sceneB_Rep2(:,jblcmv_index))];
    sc_a_rest_err(8*(i-1)+1:8*i,1)=[abs(mean_unprocessed_vasudha_scA-sceneA_Rep1(1:4,ild_index));...
        abs(mean_unprocessed_vasudha_scA-sceneA_Rep2(1:4,ild_index))];
    sc_b_rest_err(8*(i-1)+1:8*i,1)=[abs(mean_unprocessed_vasudha_scB-sceneB_Rep1(1:4,ild_index));...
        abs(mean_unprocessed_vasudha_scB-sceneB_Rep2(1:4,ild_index))];
    sc_a_rest_err(8*(i-1)+1:8*i,2)=[abs(mean_unprocessed_vasudha_scA-sceneA_Rep1(1:4,r_ild_1_index));...
        abs(mean_unprocessed_vasudha_scA-sceneA_Rep2(1:4,r_ild_1_index))];
    sc_b_rest_err(8*(i-1)+1:8*i,2)=[abs(mean_unprocessed_vasudha_scB-sceneB_Rep1(1:4,r_ild_1_index));...
        abs(mean_unprocessed_vasudha_scB-sceneB_Rep2(1:4,r_ild_1_index))];
    
    sc_a_rest_err(8*(i-1)+1:8*i,3)=[abs(mean_unprocessed_metin_scA-sceneA_Rep1(1:4,r_ild_2_index));...
        abs(mean_unprocessed_metin_scA-sceneA_Rep2(1:4,r_ild_2_index))];
    sc_b_rest_err(8*(i-1)+1:8*i,3)=[abs(mean_unprocessed_metin_scB-sceneB_Rep1(1:4,r_ild_2_index));...
        abs(mean_unprocessed_metin_scB-sceneB_Rep2(1:4,r_ild_2_index))];
    sc_a_rest_err(8*(i-1)+1:8*i,4)=[abs(mean_unprocessed_metin_scA-sceneA_Rep1(1:4,scaled_ild_0_2_index));...
        abs(mean_unprocessed_metin_scA-sceneA_Rep2(1:4,scaled_ild_0_2_index))];
    sc_b_rest_err(8*(i-1)+1:8*i,4)=[abs(mean_unprocessed_metin_scB-sceneB_Rep1(1:4,scaled_ild_0_2_index));...
        abs(mean_unprocessed_metin_scB-sceneB_Rep2(1:4,scaled_ild_0_2_index))];
    sc_a_rest_err(8*(i-1)+1:8*i,5)=[abs(mean_unprocessed_metin_scA-sceneA_Rep1(1:4,scaled_ild_0_6_index));...
        abs(mean_unprocessed_metin_scA-sceneA_Rep2(1:4,scaled_ild_0_6_index))];
    sc_b_rest_err(8*(i-1)+1:8*i,5)=[abs(mean_unprocessed_metin_scB-sceneB_Rep1(1:4,scaled_ild_0_6_index));...
        abs(mean_unprocessed_metin_scB-sceneB_Rep2(1:4,scaled_ild_0_6_index))];
    sc_a_rest_err(8*(i-1)+1:8*i,6)=[abs(mean_unprocessed_metin_scA-sceneA_Rep1(1:4,scaled_ild_1_index));...
        abs(mean_unprocessed_metin_scA-sceneA_Rep2(1:4,scaled_ild_1_index))];
    sc_b_rest_err(8*(i-1)+1:8*i,6)=[abs(mean_unprocessed_metin_scB-sceneB_Rep1(1:4,scaled_ild_1_index));...
        abs(mean_unprocessed_metin_scB-sceneB_Rep2(1:4,scaled_ild_1_index))];
   
    
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
errorbar(1:number_of_algorithms, statistic_sc_a(:,1), statistic_sc_a(:,3),...
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
errorbar(1:number_of_algorithms, statistic_sc_b(:,1), statistic_sc_b(:,3),...
    statistic_sc_b(:,4),'LineStyle','none');
xlim([0,9])
set(gca, 'XTick', 1:number_of_algorithms, 'XTickLabel', algoNames)
ylabel('localization error (degrees)');
ylim([0 150]);
grid on
legend('mean','median')
text(0,130,txt2,'bold','FontSize',14)

% was used at some point for analysis
%     sc1_err_BMVDR = mean(abs(mean_unprocessed_sceneA-sceneA_Rep1(:,bmvdr_index))+...
%         abs(mean_unprocessed_sceneA-sceneA_Rep2(:,bmvdr_index))/2);
%     sc1_err_JBLCMV = mean(abs(mean_unprocessed_sceneA-sceneA_Rep1(:,jblcmv_index))+...
%         abs(mean_unprocessed_sceneA-sceneA_Rep2(:,jblcmv_index))/2);
%     sc1_err_ILD = mean(abs(mean_unprocessed_vasudha_scA-sceneA_Rep1(1:4,ild_index))+...
%         abs(mean_unprocessed_vasudha_scA-sceneA_Rep2(1:4,ild_index))/2);
%     sc1_err_r_ILD_1 = mean(abs(mean_unprocessed_vasudha_scA-sceneA_Rep1(1:4,r_ild_1_index))+...
%         abs(mean_unprocessed_vasudha_scA-sceneA_Rep2(1:4,r_ild_1_index))/2);
%     sc1_err_r_ILD_2 = mean(abs(mean_unprocessed_vasudha_scA-sceneA_Rep1(1:4,r_ild_2_index))+...
%         abs(mean_unprocessed_vasudha_scA-sceneA_Rep2(1:4,r_ild_2_index))/2);
%     sc1_err_scaledILD_0_2 = mean(abs(mean_unprocessed_metin_scA-sceneA_Rep1(1:4,scaled_ild_0_2_index))+...
%         abs(mean_unprocessed_metin_scA-sceneA_Rep2(1:4,scaled_ild_0_2_index))/2);
%     sc1_err_scaledILD_0_6 = mean(abs(mean_unprocessed_metin_scA-sceneA_Rep1(1:4,scaled_ild_0_6_index))+...
%         abs(mean_unprocessed_metin_scA-sceneA_Rep2(1:4,scaled_ild_0_6_index))/2);
%     sc1_err_scaledILD_1 = mean(abs(mean_unprocessed_metin_scA-sceneA_Rep1(1:4,scaled_ild_1_index))+...
%         abs(mean_unprocessed_metin_scA-sceneA_Rep2(1:4,scaled_ild_1_index))/2);
%     
%     sc2_err_BMVDR = mean(abs(mean_unprocessed_sceneB-sceneB_Rep1(:,bmvdr_index))+...
%         abs(mean_unprocessed_sceneB-sceneB_Rep2(:,bmvdr_index))/2);
%     sc2_err_JBLCMV = mean(abs(mean_unprocessed_sceneB-sceneB_Rep1(:,jblcmv_index))+...
%         abs(mean_unprocessed_sceneB-sceneB_Rep2(:,jblcmv_index))/2);
%     sc2_err_ILD = mean(abs(mean_unprocessed_vasudha_scB-sceneB_Rep1(1:4,ild_index))+...
%         abs(mean_unprocessed_vasudha_scB-sceneB_Rep2(1:4,ild_index))/2);
%     sc2_err_r_ILD_1 = mean(abs(mean_unprocessed_vasudha_scB-sceneB_Rep1(1:4,r_ild_1_index))+...
%         abs(mean_unprocessed_vasudha_scB-sceneB_Rep2(1:4,r_ild_1_index))/2);
%     sc2_err_r_ILD_2 = mean(abs(mean_unprocessed_vasudha_scB-sceneB_Rep1(1:4,r_ild_2_index))+...
%         abs(mean_unprocessed_vasudha_scB-sceneB_Rep2(1:4,r_ild_2_index))/2);
%     sc2_err_scaledILD_0_2 = mean(abs(mean_unprocessed_metin_scB-sceneB_Rep1(1:4,scaled_ild_0_2_index))+...
%         abs(mean_unprocessed_metin_scB-sceneB_Rep2(1:4,scaled_ild_0_2_index))/2);
%     sc2_err_scaledILD_0_6 = mean(abs(mean_unprocessed_metin_scB-sceneB_Rep1(1:4,scaled_ild_0_6_index))+...
%         abs(mean_unprocessed_metin_scB-sceneB_Rep2(1:4,scaled_ild_0_6_index))/2);
%     sc2_err_scaledILD_1 = mean(abs(mean_unprocessed_metin_scB-sceneB_Rep1(1:4,scaled_ild_1_index))+...
%         abs(mean_unprocessed_metin_scB-sceneB_Rep2(1:4,scaled_ild_1_index))/2);
%     
%     tot_err_bmvdr = (sc1_err_BMVDR+sc2_err_BMVDR)/2;
%     tot_err_jblcmv = (sc1_err_JBLCMV+sc2_err_JBLCMV)/2;
%     tot_err_ild = (sc1_err_ILD+sc2_err_ILD)/2;  
%     tot_err_r_ild_1 = (sc1_err_r_ILD_1+sc2_err_r_ILD_1)/2;
%     tot_err_r_ild_2 = (sc1_err_r_ILD_2+sc2_err_r_ILD_2)/2;
%     tot_err_scaledILD_0_2 = (sc1_err_scaledILD_0_2+sc2_err_scaledILD_0_2)/2;
%     tot_err_scaledILD_0_6 = (sc1_err_scaledILD_0_6+sc2_err_scaledILD_0_6)/2;
%     tot_err_scaledILD_1 = (sc1_err_scaledILD_1+sc2_err_scaledILD_1)/2;
% 
%     mean_err(3*(i-1)+1,:)=[sc1_err_BMVDR,sc1_err_JBLCMV,sc1_err_ILD,...
%         sc1_err_r_ILD_1,sc1_err_r_ILD_2,sc1_err_scaledILD_0_2,...
%         sc1_err_scaledILD_0_6,sc1_err_scaledILD_1];
%     mean_err(3*(i-1)+2,:)=[sc2_err_BMVDR,sc2_err_JBLCMV,sc2_err_ILD,...
%         sc2_err_r_ILD_1,sc2_err_r_ILD_2,sc2_err_scaledILD_0_2,...
%         sc2_err_scaledILD_0_6,sc2_err_scaledILD_1];
%     mean_err(3*(i-1)+3,:)=[tot_err_bmvdr,tot_err_jblcmv,tot_err_ild,...
%         tot_err_r_ild_1,tot_err_r_ild_2,tot_err_scaledILD_0_2,...
%         tot_err_scaledILD_0_6,tot_err_scaledILD_1];