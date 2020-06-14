function [Y, X, V, A, B, fs] =...
         GenerateMicrophoneArraySignals(scene, s_name,theta_s,phi_s,U_names,...
                                        thetas_U,phis_U,duration,SNRs,SPL,SNR_overall,SNR_mic,...
                                        NFFT,Lframe,M,ref_mics)
% Purpose: Computes the microphone signals and parameters of an anechoic 
%          acoustic scene.
%
% Args:  
%       1) s_name (string):  the file name of the target signal.
%
%       2) theta_s (1 x 1):  the azimuth of the target signal w.r.t the 
%                            microphone array.
%
%       3) U_names (string): the file names of r interferers.
%
%       4) thetas_U (r x 1): The azimuths of the r interferers w.r.t the 
%                            microphone array.
%
%       5) duration (1 x 1): Duration of computed microphone signals. All
%                            .wav files should have at least this duration.
%
%       6) SNRs (r x 1):     SNRs(i) is the SNR of the target signal and 
%                            the i-th interfering signal at the left
%                            reference microphone.
%
%       9) SNR_mic (1 x 1):  SNR of target signal and mic. self noise at
%                            the left reference microphone.
%
%       10) NFFT (1 x 1):    length of FFT.
%
%       11) Lframe (1 x 1):  This is the time-frame length.
%
%       12) M (1 x 1):       is the number of mics.
%
%       13) ref_mics (2 x1): indices of reference mics.
%
%
%
% Return:
%       1) Y (M x Lsig):     Y(i,:) is the acquired noisy sig from the
%                            i-th mic.
%                         
%       2) X (M x Lsig):     The target signal at the M mics.
%
%       3) A (M x NFFT/2+1): ATFs of the target signal at the M mics.
%
%       4) B (M x NFFT/2+1 x r): ATFs of the r interferers at the M mics.
%
%       5) Fs ( 1x1) : Sampling frequency of the target signal
%
% Notes:
%       1) Lsig is the length of the signal.
%       2) r is the number of interferers.
%
%
% Author: Vasudha Sathyapriyan
%

%addpath(genpath('/Users/localadmin/Documents/GitHub/BinauralCue/SubjectiveTest/InputAudioFiles'))
%%%%%%%%%%%%%%%%%%%%%%%%%%% Initialization %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[s,fs] = audioread(s_name);         % target source at original location.
s = s(1:fs*duration);
s = s(:);
Lsig = length(s);

r = length(thetas_U);               % number of interferers.
U = zeros(Lsig,r);                  % interferers at original locations.
for inter_i=1:r
    temp = getNoise(char(U_names(inter_i)),fs);
    temp = temp(:);
    U(:,inter_i) = temp(1:Lsig);
end


%%%%%%%%%%%%%%%%%%%%%% Acquisition of Target %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[h] = signalPropagation(scene,M,theta_s,phi_s,Lframe,fs);

% i_offset = max(round(fs*(distance_from_head/c)-30),1); % skip first zeros from impulse responses.

A = computeTrueATFs(NFFT,Lframe,h);

X = ComputeMicrophoneSignals2(s,Lsig,Lframe,h);

N = zeros(M,Lsig);                  % microphones-self noise.
for mic_i=1:M
    N(mic_i,:) = randn(Lsig,1);
end 
% scaling the mic noise w.r.t target
N = fixSNR(X,N,SNR_mic,ref_mics,Lframe);
Y = zeros(size(X));

%%%%%%%%%%%%%%%%%%%%%% Acquisition of Interferers %%%%%%%%%%%%%%%%%%%%%%%%%
E_N = zeros(M,Lsig,r) ;%+repmat(N,1,1,r);
V = zeros(M,Lsig,r) ;
B = zeros(M,NFFT/2+1,r);            % interferers's ATFs w.r.t mic. array.
for numbInter=1:r    
    [h] = signalPropagation(scene,M,thetas_U(numbInter),phis_U(numbInter),Lframe,fs);
            	
   for i=1:M
        temp = conv(U(:,numbInter),h(i,:));
        temp = temp(:);
        V(i,:,numbInter) = temp(1:Lsig);
         
        tempfft = fft(h(i,:),NFFT);
        B(i,:,numbInter) = tempfft(1:NFFT/2+1);
    end
   
    E_N(:,:,numbInter) = E_N(:,:,max(1,numbInter-1)) + V(:,:,numbInter);
    
  %fixing the SNR by scaling X w.r.t noise, then adding the interfering signals and microphone noise 
    Y(:,:,numbInter) = fixSNR_overall(X,(squeeze(E_N(:,:,numbInter))+N),SNR_overall,ref_mics,Lframe) + E_N(:,:,numbInter) + N; 
end

end


function noise = getNoise(noise_name,fs)
% Resample the noise with the sampling frequency of the signal.
%
% Author: Andreas Koutrouvelis

[noise, FsN] = audioread(noise_name);

if(~isempty(fs))
    if(FsN ~= fs)
        noise = resampling(noise,fs,FsN);
    end
end

end


function [h] = signalPropagation(scene,M,theta,phi,N,Fs)
% It simulates the propagation of the signal 's'
% from the target position to the microphones positions .
if(scene == 1)
    shortIR = 50*1e-03*Fs; %T_60 is 50ms
    temp = loadHRIR('Anechoic', 80, phi, theta,'all');
    IR_Fs = temp.fs;
    IR = resampling(temp.data,Fs,IR_Fs);
    h = [IR(1:shortIR,3)';IR(1:shortIR,5)';IR(1:shortIR,6)';IR(1:shortIR,4)'];
elseif(scene == 2)
    shortIR = 10*1e-03*Fs; %Early rtf is 10ms
    temp = loadHRIR('Office_I',theta,'bte');
    IR_Fs = temp.fs;
    IR = resampling(temp.data,Fs,IR_Fs);
    h = [IR(1:shortIR,1)';IR(1:shortIR,3)';IR(1:shortIR,4)';IR(1:shortIR,2)'];
else
    shortIR = 5e-02*Fs; %T_60 is 50ms
    temp = loadHRIR('Anechoic', 80, phi, theta,'all');
    IR_Fs = temp.fs;
    IR = resampling(temp.data,Fs,IR_Fs);
    h = [IR(1:shortIR,3)';IR(1:shortIR,5)';IR(1:shortIR,6)';IR(1:shortIR,4)'];
end

end


function mic_sigs = ComputeMicrophoneSignals2(sig,Lsig,N_ATF,h)
% It computes the microphone signals.
%
% Author: Andreas Koutrouvelis
[M,L_h] = size(h);

mic_sigs = zeros(M,Lsig);
    
for mic_i=1:M
    temp_sig = conv(sig,h(mic_i,:));
    temp_sig = temp_sig(1:Lsig);
    mic_sigs(mic_i,:) = temp_sig(:);
end

end


function speech_scaled = fixSNR_overall(target,noise,SNR,ref_mics,Lframe)
% This function scales the noise shuch that the SNR at
% the left reference microphone is equal to the given SNR value.
%
% Author: Andreas Koutrouvelis
[M,N1] = size(target);
[M,N2] = size(noise);
N = min([N1 N2]);

target_new = target(:,1:N);
noise_new = noise(:,1:N);

vad_thres = ( mean(abs(target_new(ref_mics(1),:))) )/15;
vd_L_target = idealVAD(target_new(ref_mics(1),:),vad_thres,Lframe);
target_tempL = target_new(ref_mics(1),vd_L_target == 1);
target_tempL = target_tempL(:);

vad_thres = ( mean(abs(noise_new(ref_mics(1),:))) )/15;
vd_L_noise = idealVAD(noise_new(ref_mics(1),:),vad_thres,Lframe);
noise_tempL = noise_new(ref_mics(1),vd_L_noise == 1);
noise_tempL = noise_tempL(:);


varNoise = var(noise_tempL);

var_target = varNoise * (10^(SNR/10)) ;

scaling_factor = sqrt(var_target)/sqrt(var(target_tempL));

speech_scaled = scaling_factor.*target_new;

end

function noise_scaled = fixSNR(target,noise,SNR,ref_mics,Lframe)
% This function scales the noise shuch that the SNR at
% the left reference microphone is equal to the given SNR value.
%
% Author: Andreas Koutrouvelis
[M,N1] = size(target);
[M,N2] = size(noise);
N = min([N1 N2]);

target_new = target(:,1:N);
noise_new = noise(:,1:N);

vad_thres = ( mean(abs(target_new(ref_mics(1),:))) )/15;
vd_L_target = idealVAD(target_new(ref_mics(1),:),vad_thres,Lframe);
target_tempL = target_new(ref_mics(1),vd_L_target == 1);
target_tempL = target_tempL(:);

vad_thres = ( mean(abs(noise_new(ref_mics(1),:))) )/15;
vd_L_noise = idealVAD(noise_new(ref_mics(1),:),vad_thres,Lframe);
noise_tempL = noise_new(ref_mics(1),vd_L_noise == 1);
noise_tempL = noise_tempL(:);

varTarget = var(target_tempL);

var_noise = varTarget / (10^(SNR/10)) ;

scaling_factor = sqrt(var_noise)/sqrt(var(noise_tempL));

noise_scaled = scaling_factor.*noise_new;

end

function noise_scaled = fixNoiseSPL(noise,SPL,ref_mics,Lframe)
% This function scales the noise shuch that the noise is at the given SPL.

vad_thres = ( mean(abs(noise(ref_mics(1),:))) )/15;
vd_L_noise = idealVAD(noise(ref_mics(1),:),vad_thres,Lframe);
noise_tempL = noise(ref_mics(1),vd_L_noise == 1);
noise_tempL = noise_tempL(:);

var_noise = 1 / (10^(SPL/10)) ;

scaling_factor = sqrt(var_noise)/sqrt(var(noise_tempL));

noise_scaled = scaling_factor.*noise;

end
function ATFs = computeTrueATFs(NFFT,N_ATF,h,i_offset)
% Purpose: It computes the ATFs of the given impulse responses.
%
% Author: Andreas Koutrouvelis
if (nargin < 3)
    error('You must provide at least the front and middle impulse responses!\n');
end

if (nargin < 4)
    i_offset = 1;
    % does not skip initial zeros.
end

h = h(:,i_offset:end);

[M,L_h] = size(h);

ATFs = zeros(M,NFFT/2+1);

for mic_i=1:M
    temp = fft(h(mic_i,:),NFFT);
    ATFs(mic_i,:) = temp(1:NFFT/2+1);
end

end


function resampled = resampling(sig,Fnext,Fprev)
% It resamples from Fprev to Fnext frequency. 
%
% Author: Andreas Koutrouvelis

order = 50;
[~,c] = size(sig);
resampled = [];
for i=1:c
    resampled = [resampled resample(sig(:,i),Fnext,Fprev,order)];
end

resampled = resampled.*(Fprev/Fnext); 
end