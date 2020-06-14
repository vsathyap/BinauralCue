clear all;
close all;
clc;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                                                         %
%                   Setup Acoustic Scene & Microphone array               %
%                                                                         %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
numbInter = 3;                       % number of interferers.

theta_s = 0;                         % azimuth of the target w.r.t. array.
disp('1.Scene 1');
disp('2.Scene 2');
disp('2.Scene 3');
Scenario = input('Enter the Test Scenario:');
switch Scenario
    case 1
        theta_U = [-90 60 -45];  % azimuths of interferers w.r.t array.
        %90-Right ear, -90-Left ear
        theta_DVF = [90 300 45];  % azimuths of interferers w.r.t array.

        Scene = 1;
    case 2
        theta_U = [90 -60 30];  % azimuths of interferers w.r.t array.
        theta_DVF = [270 60 330];  % azimuths of interferers w.r.t array.

        %90-Right ear, -90-Left ear
        Scene = 1;
    case 3
        theta_U = [-75 -15 -60];  % azimuths of interferers w.r.t array.
        %90-Right ear, -90-Left ear
        theta_DVF = [75 15 60];  % azimuths of interferers w.r.t array.

        Scene = 2;
    otherwise
        theta_U = [-90 60 -45];  % azimuths of interferers w.r.t array.
        %90-Right ear, -90-Left ear
        theta_DVF = [90 330 45];  % azimuths of interferers w.r.t array.

        Scene = 1;
end
 
phi_s = 0;                           % elevation of the target w.r.t. array.
phi_U = zeros(size(theta_U));        % elevations of interferers w.r.t array.

s_name = 'InputAudioFiles/Sound_1.wav';    % target .wav file.

U_names_all = {'InputAudioFiles/Sound_2.wav',...  % interferers' .wav files.
    'InputAudioFiles/Sound_3.wav',...
    'InputAudioFiles/Sound_4.wav'};

duration = 4;                        % duration of mic. signals in sec.
SNRs = zeros(size(theta_U));     % interferers' input SNRs (dB) at left reference mic.
SNR_mic = 50;                        % mic. self noise input SNR (dB).


M = 4;                               % number of mics (even number).
ref_mics = [1 M];                    % reference microphones' indices.

if (Scene == 1)
    N = 5e-02*16000;                     % time-frame length in samples.
elseif (Scene == 2)
    N = 300*1e-03*16000;                     % time-frame length in samples.
else
    N = 5e-02*16000;                     % time-frame length in samples.
end
NFFT = 2^nextpow2(N);                % FFT length.

numbMethods = 1;
%Methods:
% 1.BMVDR
% 2.JBLCMV
% 3.ILD
% 4.ILD_relaxed
% 9.Low Enhanced
method = 9; % To run all the 4 methods, to run individually-- change the method to the corresponding number.
if(method == 8)
    numbMethods = 4;
end

U_names = {U_names_all{1,1:numbInter}};
theta_U = theta_U(1:numbInter);
phi_U = phi_U(1:numbInter);
SNRs = SNRs(1:numbInter);
noiseSPL = 65; % noise SPL at 65 dB
SNR_overall = -5; % -5dB SNR
% Generate microphone signals
[Y, X, V, A, B, Fs] =...
    GenerateMicrophoneArraySignals(Scene, s_name,theta_s,phi_s,U_names,theta_U,phi_U,...
    duration,SNRs,noiseSPL,SNR_overall,SNR_mic,NFFT,...
    N,M,ref_mics);

near_field_distances = [0.2,0.4,0.6,0.8,1];
ear_location = 100;
ILD_scale = DVF_ILD(near_field_distances,Fs,NFFT,ear_location,theta_DVF);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                                                         %
%                       Processing signals                                %
%                                                                         %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

version = 'fixed';        % The spatial filter is computed once.
CPSDM_sel = 'noise';      % The spatial filter uses the noise CPSDM.


c = 0.1;%[0.1:0.2:0.9];                  % relaxation parameter for RJBLCMV_SCO method.
ILD_max = 4; %in dB              % max error in ILD allowed.

iter = 5;
ild_low = near_field_distances(iter);

if(strcmp(CPSDM_sel,'noisy'))
    [x_hat_L,x_hat_R,Ws_L,Ws_R] =...
        BinauralProcessing(method,numbMethods,Y,X,A,B,N,NFFT,ref_mics,c,ILD_max,version,Fs);
elseif(strcmp(CPSDM_sel,'noise'))
    % ideal VAD: used to compute the noise CPSDM.
    vad_thres = ( mean(abs(X(ref_mics(1),:))) )/15;
    vd_L = idealVAD(X(ref_mics(1),:),vad_thres,200);
    [x_hat_L,x_hat_R,Ws_L,Ws_R] = BinauralProcessing(method,numbMethods,Y,X,A,B,N,NFFT,ref_mics,c,ILD_max,version,Fs,vd_L,ILD_scale(:,:,iter));
end

%Normalising the output to be used in the subjective Tests

X_Pre_rms = [X(ref_mics(1),:)' X(ref_mics(2),:)']./(sum([X(ref_mics(1),:) X(ref_mics(2),:)].^2).^(0.5));
audiowrite(['/Users/localadmin/Documents/GitHub/BinauralCue/SubjectiveTest/OutputAudioFiles/Sound_Sc' num2str(Scenario) '_Unprocessed_Target.wav'],X_Pre_rms,Fs);
for r = 1:numbInter
    Y_Pre_rms(:,:,r) = [V(ref_mics(1),:,r)' V(ref_mics(2),:,r)']./(sum([V(ref_mics(1),:,r) V(ref_mics(2),:,r)].^2).^(0.5));
    audiowrite(['/Users/localadmin/Documents/GitHub/BinauralCue/SubjectiveTest/OutputAudioFiles/Sound_Sc' num2str(Scenario) '_Unprocessed_r' num2str(r) '.wav'],Y_Pre_rms(:,:,r),Fs);
end


for met = 1:numbMethods
    [Y_N_L(:,:,met), Y_N_R(:,:,met)] = processedNoise(V,N,NFFT,squeeze(Ws_L(:,:,:,numbInter,met)),squeeze(Ws_R(:,:,:,numbInter,met)));
    [X_L(:,met),X_R(:,met)] = processedNoise(X,N,NFFT,squeeze(Ws_L(:,:,:,numbInter,met)),squeeze(Ws_R(:,:,:,numbInter,met)));
    X_rms(:,:,met) = [X_L(:,met) X_R(:,met)]./(sum([X_L(:,met)' X_R(:,met)'].^2).^(0.5));
    audiowrite(['/Users/localadmin/Documents/GitHub/BinauralCue/SubjectiveTest/OutputAudioFiles/Sound_Sc' num2str(Scenario) '_m' num2str(5) '_ILD' num2str(ild_low) 'Target.wav'],X_rms(:,:,met),Fs);

    for r = 1:numbInter
        
        Y_rms(:,:,r,met) = [Y_N_L(:,r,met) Y_N_R(:,r,met)]./(sum([Y_N_L(:,r,met)' Y_N_R(:,r,met)'].^2).^(0.5));
        audiowrite(['/Users/localadmin/Documents/GitHub/BinauralCue/SubjectiveTest/OutputAudioFiles/Sound_Sc' num2str(Scenario) '_m' num2str(5) '_r' num2str(r) '_ILD' num2str(ild_low) '.wav'],Y_rms(:,:,r,met),Fs,'BitsPerSample',32);
    end
end


