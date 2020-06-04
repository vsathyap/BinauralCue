function [x_hat_L,x_hat_R,Ws_L,Ws_R] = BinauralProcessing(method,numbMethods,Y_signals,X,A,B,N,NFFT,ref_mics,c,k_max,version,Fs,VAD)

[M, L_Y,numbInter] = size(Y_signals);

Shift = N/2;                       % 50% OLA step.
Nfr = floor(L_Y/Shift);            % number of frames.

frequency = 0:Fs/NFFT:Fs/2;

x_hat_L = squeeze(zeros(Shift*Nfr,numbInter,numbMethods));
x_hat_R = squeeze(zeros(Shift*Nfr,numbInter,numbMethods));
Ws_L = squeeze(zeros(M,NFFT/2+1,Nfr,numbInter,numbMethods));
Ws_R = squeeze(zeros(M,NFFT/2+1,Nfr,numbInter,numbMethods));


for r = numbInter
    Y = Y_signals(:,:,r);
    if(strcmp(version,'fixed') && isempty(VAD))
        CPSDM = computeAverageCPSD(Y,N,Shift,NFFT);
    elseif(strcmp(version,'fixed') && ~isempty(VAD))
        CPSDM = computeAverageCPSD(Y(:,VAD == 0),N,Shift,NFFT);
        CPSDM_x = computeAverageCPSD(Y(:,VAD == 1),N,Shift,NFFT);
        CPSDM_y = computeAverageCPSD(Y,N,Shift,NFFT);
    else
        CPSDM = [];
    end
    
    switch method
        case {1,2,3,4,5,6,7}           
            [x_hat_L(:,r),x_hat_R(:,r),Ws_L(:,:,:,r),Ws_R(:,:,:,r),ITF_s_in, ITF_int_in, ITF_s_out,...
                ITF_int_out,SNR_in,SNR_out] = computeMethod(A,B(:,:,1:r),c,r,k_max,ref_mics,Fs,N,Nfr,Y,CPSDM,CPSDM_x,CPSDM_y,M,NFFT,version,method);

        case 4
            for c_val = 1:length(c)
                [x_hat_L(:,r),x_hat_R(:,r),Ws_L(:,:,:,r),Ws_R(:,:,:,r),ITF_s_in, ITF_int_in, ITF_s_out,...
                    ITF_int_out,SNR_in,SNR_out] = computeMethod(A,B(:,:,1:r),c(c_val),r,k_max,ref_mics,Fs,N,Nfr,Y,CPSDM,CPSDM_x,CPSDM_y,M,NFFT,version,method);
            end
        case 8
            for met = 1: numbMethods
                [x_hat_L(:,r,met),x_hat_R(:,r,met),Ws_L(:,:,:,r,met),Ws_R(:,:,:,r,met),ITF_s_in, ITF_int_in, ITF_s_out,...
                    ITF_int_out,SNR_in, SNR_out]= computeMethod(A,B(:,:,1:r),c,r,k_max,ref_mics,Fs,N,Nfr,Y,CPSDM,CPSDM_x,CPSDM_y,M,NFFT,version,met);         
            end
    end
end
end

function [x_hat_L,x_hat_R,Ws_L,Ws_R,ITF_s_in, ITF_int_in, ITF_s_out,...
    ITF_int_out,SNR_in, SNR_out] = computeMethod(A,B,c,r,k_max,ref_mics,Fs,N,Nfr,Y,CPSDM,CPSDM_x,CPSDM_y,M,NFFT,version,method)

slice = 1:N;
Win = sqrt(hanning(N));
Shift = N/2;
frequency = 0:Fs/NFFT:Fs/2;
Bin_1500 = frequency >= 1500;

Buffer_d_hat_L = zeros(Shift,1);
Buffer_d_hat_R = zeros(Shift,1);
tosave = 1:Shift;
x_hat_L = zeros(Shift*Nfr,1);
x_hat_R = zeros(Shift*Nfr,1);
Ws_L = zeros(M,NFFT/2+1,Nfr);
Ws_R = zeros(M,NFFT/2+1,Nfr);

ITF_s_in = zeros(NFFT/2+1, Nfr-1);
ITF_int_in = zeros(NFFT/2+1,r, Nfr-1);
ITF_s_out = zeros(NFFT/2+1, Nfr-1);
ITF_int_out = zeros(NFFT/2+1,r, Nfr-1);
SNR_in = zeros(Nfr-1,1);
SNR_out = zeros(Nfr-1,1);

w_L =[];
w_R =[];

CPSDM_counter = 1;
%%%%%%%%%%%%%%%%%%% enahancement frame by frame %%%%%%%%%%%%%%%%%%%%%%%
for l=1:Nfr-1
    Y_frame = Y(:,slice);
    for mic_i = 1:M
        Y_frame(mic_i,:) = (Win').*Y_frame(mic_i,:);
    end
    
    Y_frame_fft = obtainFFTs(Y_frame,NFFT);
    Y_frame_fft = Y_frame_fft(:,1:NFFT/2+1);
    
    %     if(strcmp(version,'adaptive'))
    %         if(~isempty(VAD))
    %             % noise CPSDM update
    %             if( (sum(VAD(slice)) == 0))
    %                 CPSDM = updateCPSD(CPSDM,Y_frame_fft,CPSDM_counter); % updated average.
    %                 CPSDM_counter = CPSDM_counter + 1;
    %             end
    %         else
    %             % noisy CPSDM update
    %             CPSDM = updateCPSD(CPSDM,Y_frame_fft,CPSDM_counter); % updated average.
    %             CPSDM_counter = CPSDM_counter + 1;
    %         end
    %     end
    
    if(CPSDM_counter-1 >= M || strcmp(version,'fixed')) % if CPSDM_counter-1 < M then CPSDM is low rank if we have adaptive implementation.
        switch method
            case 1
                if(strcmp(version,'fixed'))
                    [x_hat_frame_L,x_hat_frame_R,w_L,w_R,ITF_s_in(:,l), ITF_int_in(:,:,l), ITF_s_out(:,l),...
                        ITF_int_out(:,:,l),SNR_in(l), SNR_out(l)] = BMVDR(Y_frame_fft,A,B,N,CPSDM,CPSDM_x,ref_mics,w_L,w_R,Fs);
                elseif(strcmp(version,'adaptive'))
                    [x_hat_frame_L,x_hat_frame_R,w_L,w_R] = BMVDR(Y_frame_fft,A,B,N,CPSDM,CPSDM_x,ref_mics,[],[]);
                end  
            case 2
                if(strcmp(version,'fixed'))
                    [x_hat_frame_L,x_hat_frame_R,w_L,w_R,ITF_s_in(:,l), ITF_int_in(:,:,l), ITF_s_out(:,l),...
                        ITF_int_out(:,:,l),SNR_in(l), SNR_out(l)]= JBLCMV(Y_frame_fft,A,B,N,CPSDM,CPSDM_x,ref_mics,w_L,w_R);
                elseif(strcmp(version,'adaptive'))
                    [x_hat_frame_L,x_hat_frame_R,w_L,w_R] = JBLCMV(Y_frame_fft,A,B,N,CPSDM,CPSDM_x,ref_mics,[],[]);
                end
            case 3
                if(strcmp(version,'fixed'))
                    [x_hat_frame_L,x_hat_frame_R,w_L,w_R,ITF_s_in(:,l), ITF_int_in(:,:,l), ITF_s_out(:,l),...
                        ITF_int_out(:,:,l),SNR_in(l), SNR_out(l)] = BBILD_SDCR(Y_frame_fft,A,B,N,CPSDM,CPSDM_x,Bin_1500,ref_mics,r,w_L,w_R);
                elseif(strcmp(version,'adaptive'))
                    [x_hat_frame_L,x_hat_frame_R,w_L,w_R] = BBILD_SDCR(Y_frame_fft,A,B,N,CPSDM,CPSDM_x,Bin_1500,ref_mics,r,[],[]);
                end
            case 4
                if(strcmp(version,'fixed'))
                    [x_hat_frame_L,x_hat_frame_R,w_L,w_R,ITF_s_in(:,l), ITF_int_in(:,:,l), ITF_s_out(:,l),...
                        ITF_int_out(:,:,l),SNR_in(l), SNR_out(l)] = RBBILD_SDCR(Y_frame_fft,A,B,N,CPSDM,CPSDM_x,Bin_1500,ref_mics,c,w_L,w_R);
                elseif(strcmp(version,'adaptive'))
                    [x_hat_frame_L,x_hat_frame_R,w_L,w_R] = RBBILD_SDCR(Y_frame_fft,A,B,N,CPSDM,CPSDM_x,Bin_1500,ref_mics,c,[],[]);
                end
            case 5
                if(strcmp(version,'fixed'))
                    [x_hat_frame_L,x_hat_frame_R,w_L,w_R,ITF_s_in(:,l), ITF_int_in(:,:,l), ITF_s_out(:,l),...
                        ITF_int_out(:,:,l),SNR_in(l), SNR_out(l)] = RBBILD_SDCR_4dB(Y_frame_fft,A,B,N,CPSDM,CPSDM_x,Bin_1500,ref_mics,k_max,w_L,w_R);
                elseif(strcmp(version,'adaptive'))
                    [x_hat_frame_L,x_hat_frame_R,w_L,w_R] = RBBILD_SDCR_4dB(Y_frame_fft,A,B,N,CPSDM,CPSDM_x,Bin_1500,ref_mics,c,[],[]);
                end
            case 6
                if(strcmp(version,'fixed'))
                    [x_hat_frame_L,x_hat_frame_R,w_L,w_R,ITF_s_in(:,l), ITF_int_in(:,:,l), ITF_s_out(:,l),...
                        ITF_int_out(:,:,l),SNR_in(l), SNR_out(l)] = RJBLCMV_SCO(Y_frame_fft,A,B,N,CPSDM,CPSDM_x,ref_mics,c,k_max,w_L,w_R);
                elseif(strcmp(version,'adaptive'))
                    [x_hat_frame_L,x_hat_frame_R,w_L,w_R] = RJBLCMV_SCO(Y_frame_fft,A,B,N,CPSDM,CPSDM_x,ref_mics,c,k_max,[],[]);
                end
            case 7
                if(strcmp(version,'fixed'))
                    [x_hat_frame_L,x_hat_frame_R,w_L,w_R,ITF_s_in(:,l), ITF_int_in(:,:,l), ITF_s_out(:,l),...
                        ITF_int_out(:,:,l),SNR_in(l), SNR_out(l)] = RJBLCMV_SDCR(Y_frame_fft,A,B,N,CPSDM,CPSDM_x,ref_mics,c,w_L,w_R);
                elseif(strcmp(version,'adaptive'))
                    [x_hat_frame_L,x_hat_frame_R,w_L,w_R] = RJBLCMV_SDCR(Y_frame_fft,A,B,N,CPSDM,CPSDM_x,ref_mics,c,[],[]);
                end
            otherwise
                x_hat_frame_L = zeros(N,1);
                x_hat_frame_R = zeros(N,1);
        end
        
        Ws_L(:,:,l) = w_L(:,:);
        Ws_R(:,:,l) = w_R(:,:);
    else
        x_hat_frame_L = zeros(N,1);
        x_hat_frame_R = zeros(N,1);
    end
    
    
    x_hat_frame_win_L = Win.*x_hat_frame_L;
    x_hat_frame_win_R = Win.*x_hat_frame_R;
    
    
    % OLA
    x_hat_frame_win_L(1:Shift) = x_hat_frame_win_L(1:Shift) + Buffer_d_hat_L;
    x_hat_frame_win_R(1:Shift) = x_hat_frame_win_R(1:Shift) + Buffer_d_hat_R;
    x_hat_L(tosave) = x_hat_frame_win_L(1:Shift);
    x_hat_R(tosave) = x_hat_frame_win_R(1:Shift);
    Buffer_d_hat_L = x_hat_frame_win_L(Shift+1:N);
    Buffer_d_hat_R = x_hat_frame_win_R(Shift+1:N);
    
    
    % move to the next frame
    slice = slice+Shift;
    tosave = tosave+Shift;
end

end

function Y_frame_fft = obtainFFTs(Y_frame,NFFT)
% Purpose: Computes the FFTs of the signals in Y_frame.
%
% Author: Andreas Koutrouvelis
Y_frame_fft = fft(Y_frame.',NFFT).';

end



function CPSDM = computeAverageCPSD(Y,N,Shift,NFFT)
% Purpose: This function computes the average CPSDM.
%
% Author: Andreas Koutrouvelis
CPSDM = [];
slice = 1:N;

[~,L_Y] = size(Y);

Nfr = floor(L_Y/Shift);

for l=1:Nfr-1
    sig_frame = Y(:,slice);
    
    X_frame_fft = fft(sig_frame.',NFFT).';
    CPSDM = updateCPSD(CPSDM,X_frame_fft(:,1:NFFT/2+1),l);
    
    slice = slice + Shift;
end


end


function P = updateCPSD(P_prev,X_frame_fft,l)
% Purpose: This function updates the CPSDM at each time-frame.
%
% Author: Andreas Koutrouvelis
[M,NFFT_small] = size(X_frame_fft);
NFFT = NFFT_small*2 - 2;

P_current = zeros(M,M,NFFT/2+1);
P = zeros(M,M,NFFT/2+1);


for freq_bin=1:NFFT/2+1
    P_current(:,:,freq_bin) = X_frame_fft(:,freq_bin)*X_frame_fft(:,freq_bin)';
end


if(isempty(P_prev))
    P(:,:,:) = P_current(:,:,:);
else
    P = ((l-1)/l).*P_prev + (1/l).*P_current;
end

end




function [x_L,x_R,w_L,w_R, ITF_s_in, ITF_int_in, ITF_s_out, ITF_int_out, SNR_in, SNR_out] = RJBLCMV_SCO(y,a,b,N,P,Px,ref_mics,c,k_max,w_L_prev,w_R_prev)
% Purpose: This function implements the RJBLCMV beamformer using CVX.
%
% Author: Andreas Koutrouvelis

[M,NFFT_small] = size(y);
NFFT = 2*NFFT_small-2;
w_L = zeros(M,NFFT/2+1);
w_R = zeros(M,NFFT/2+1);
x_L_fft = zeros(NFFT,1);
x_R_fft = zeros(NFFT,1);
[~,~,r] = size(b);

if(c == 0 && r > 2*M-3)
    m = 2*M-3;
else
    m = r;
end

ITF_s_in = zeros(NFFT/2+1,1);
ITF_int_in = zeros(NFFT/2+1,m);
ITF_s_out = zeros(NFFT/2+1,1);
ITF_int_out = zeros(NFFT/2+1,m);
SNR_in_num = zeros(NFFT/2+1,1);
SNR_in_den = zeros(NFFT/2+1,1);
SNR_out_num = zeros(NFFT/2+1,1);
SNR_out_den = zeros(NFFT/2+1,1);
e_L = zeros(M,1);
e_L(ref_mics(1)) = 1;
e_R = zeros(M,1);
e_R(ref_mics(2)) = 1;

for freq_bin=1:NFFT/2+1
    P_n = P(:,:,freq_bin);
    P_x = Px(:,:,freq_bin);
    if(isempty(w_L_prev) || isempty(w_R_prev))
        A = a(:,freq_bin);
        
        
        A_L = A(ref_mics(1));
        A_R = A(ref_mics(2));
        
        Lambda_1 = [A zeros(size(A)); zeros(size(A)) A];
        f_1 = [A_L A_R]';
        
        Lambda_2 = zeros(2*M,m);
        Lambda_BB = zeros(M,m);
        for numbInter=1:m
            B = b(:,freq_bin,numbInter);
            B_L = B(ref_mics(1));
            B_R = B(ref_mics(2));
            Lambda_2(:,numbInter) = [B.*B_R; -B.*B_L];
            
            Lambda_BB(:,numbInter) = B.*( (B_R*A_L-B_L*A_R)/A_R );
        end
        
        Lambda = [Lambda_1 Lambda_2];
        
        tilda_P_n = [P_n zeros(size(P_n)); zeros(size(P_n)) P_n];
        tilda_P_n_sqrt = sqrtm(tilda_P_n)';
        
        % next four lines compute BMVDR binaural filter.
        temp = tilda_P_n \ Lambda_1;
        w_BMVDR = temp * ( ((Lambda_1')*temp) \ f_1 );
        epsilon = c*abs(w_BMVDR(length(w_BMVDR)/2+1:end)'*Lambda_BB);
        epsilon = epsilon(:);
        
        if(c < 1 && c > 0 && m~=0)
            tau = c;
            min_errorr = 10^10;
            min_errorr_w = zeros(2*M,1);
            for iter=1:k_max+1
                cvx_begin
                cvx_solver sedumi
                cvx_precision best;
                cvx_quiet(true);
                variable w_hat(2*M) complex;
                minimize( norm(tilda_P_n_sqrt*w_hat,2) )
                subject to
                abs(Lambda_2'*w_hat) <= epsilon;
                Lambda_1'*w_hat == f_1;
                cvx_end
                
                [stopp,errorr] = stoping_criterion(w_hat,w_BMVDR,b(:,freq_bin,:),c);
                
                if(stopp)
                    min_errorr_w = w_hat;
                    break;
                else
                    tau = tau - (c/k_max);
                    if(tau<0)
                        tau = 0;
                    end
                    epsilon = tau*abs(w_hat(length(w_hat)/2+1:end)'*Lambda_BB);
                    epsilon = epsilon(:);
                    
                    if(min_errorr > errorr)
                        min_errorr = errorr;
                        min_errorr_w = w_hat;
                    end
                end
            end
            w_hat = min_errorr_w;
            
            
            w_L(:,freq_bin) = w_hat(1:length(w_hat)/2);
            w_R(:,freq_bin) = w_hat(length(w_hat)/2+1:end);
        elseif(c>=1 || m==0)
            % BMVDR
            w_L(:,freq_bin) = w_BMVDR(1:length(w_BMVDR)/2);
            w_R(:,freq_bin) = w_BMVDR(length(w_BMVDR)/2+1:end);
        elseif(c==0 && m~=0)
            % JBLCMV
            f = [f_1; zeros(m,1)];
            temp = tilda_P_n \ Lambda;
            w_hat = temp * ( ((Lambda')*temp) \ f );
            w_L(:,freq_bin) = w_hat(1:length(w_hat)/2);
            w_R(:,freq_bin) = w_hat(length(w_hat)/2+1:end);
        end
        
    else
        w_L = w_L_prev;
        w_R = w_R_prev;
    end
    
    x_L_fft(freq_bin) = w_L(:,freq_bin)'*y(:,freq_bin);
    x_R_fft(freq_bin) = w_R(:,freq_bin)'*y(:,freq_bin);
    
    %ITF at the input
    ITF_s_in(freq_bin) = a(ref_mics(1),freq_bin)/a(ref_mics(2),freq_bin);
    for numbInter=1:m
        ITF_int_in(freq_bin,numbInter) = b(ref_mics(1),freq_bin,numbInter)/b(ref_mics(2),freq_bin,numbInter);
    end
    
    %ITF at the output
    ITF_s_out(freq_bin) = (w_L(:,freq_bin)'*a(:,freq_bin))/(w_R(:,freq_bin)'*a(:,freq_bin));
    for numbInter=1:m
        ITF_int_out(freq_bin,numbInter) = (w_L(:,freq_bin)'*b(:,freq_bin,numbInter))/(w_R(:,freq_bin)'*b(:,freq_bin,numbInter));
    end
    
    SNR_in_num(freq_bin) = (e_L'*P_x*e_L + e_R'*P_x*e_R);
    SNR_in_den(freq_bin) = (e_L'*P_n*e_L + e_R'*P_n*e_R);
    SNR_out_num(freq_bin) = (w_L(:,freq_bin)'*P_x*w_L(:,freq_bin) + w_R(:,freq_bin)'*P_x*w_R(:,freq_bin));
    SNR_out_den(freq_bin) = (w_L(:,freq_bin)'*P_n*w_L(:,freq_bin) + w_R(:,freq_bin)'*P_n*w_R(:,freq_bin));
end

SNR_in = 10*log10(abs(sum(SNR_in_num)/sum(SNR_in_den)));
SNR_out = 10*log10(abs(sum(SNR_out_num)/sum(SNR_out_den)));


x_L_fft(NFFT/2+2:end) = flipud(conj(x_L_fft(2:end/2)).').';
x_R_fft(NFFT/2+2:end) = flipud(conj(x_R_fft(2:end/2)).').';

x_L = real(ifft(x_L_fft,NFFT,'symmetric'));
x_L = x_L(1:N);  % remove the zero padding
x_L = x_L(:);

x_R = real(ifft(x_R_fft,NFFT,'symmetric'));
x_R = x_R(1:N);  % remove the zero padding
x_R = x_R(:);

end


function [stopp,sum_errorr] = stoping_criterion(w_hat,w_BMVDR,a_inter,c)
% Purpose: Stoping criterion for RJBLCMV beamformer.
%
% Author: Andreas Koutrouvelis

[M,f,m] = size(a_inter);
stopp = 1;

sum_errorr = 0;
for numbInter=1:m
    B = zeros(M,1);
    B = a_inter(:,1,numbInter);
    
    left = abs( (w_hat(1:M)'*B)/(w_hat(M+1:end)'*B) - B(1)/B(M) );
    right = c*abs( (w_BMVDR(1:M)'*B)/(w_BMVDR(M+1:end)'*B) - B(1)/B(M) );
    errorr = left - right;
    if(errorr > 0)
        sum_errorr = sum_errorr + abs(errorr);
        stopp = 0;
    end
end

end
function [x_L,x_R,w_L,w_R, ITF_s_in, ITF_int_in, ITF_s_out, ITF_int_out, SNR_in, SNR_out] = RJBLCMV_SDCR(y,a,b,N,P,Px,ref_mics,c,w_L_prev,w_R_prev)
% Purpose: This function implements the RJBLCMV_SDCR beamformer.
%
% Author: Andreas Koutrouvelis
[M,NFFT_small] = size(y);
NFFT = 2*NFFT_small-2;
w_L = zeros(M,NFFT/2+1);
w_R = zeros(M,NFFT/2+1);
x_L_fft = zeros(NFFT,1);
x_R_fft = zeros(NFFT,1);
[~,~,r] = size(b);

if(c == 0 && r > 2*M-3)
    m = 2*M-3;
else
    m = r;
end

ITF_s_in = zeros(NFFT/2+1,1);
ITF_int_in = zeros(NFFT/2+1,m);
ITF_s_out = zeros(NFFT/2+1,1);
ITF_int_out = zeros(NFFT/2+1,m);
SNR_in_num = zeros(NFFT/2+1,1);
SNR_in_den = zeros(NFFT/2+1,1);
SNR_out_num = zeros(NFFT/2+1,1);
SNR_out_den = zeros(NFFT/2+1,1);
e_L = zeros(M,1);
e_L(ref_mics(1)) = 1;
e_R = zeros(M,1);
e_R(ref_mics(2)) = 1;

for freq_bin=1:NFFT/2+1
    P_n = P(:,:,freq_bin);
    P_n = 0.5*(P_n + P_n');  % makes sure that the CPSDM is Hermitian
    P_x = Px(:,:,freq_bin);
    P_x = 0.5*(P_x + P_x');
    if(isempty(w_L_prev) || isempty(w_R_prev))
        A = a(:,freq_bin);
        
        A_L = A(ref_mics(1));
        A_R = A(ref_mics(2));
        
        Lambda_1 = [A zeros(size(A)); zeros(size(A)) A];
        f_1 = [A_L A_R]';
        
        BigMatrix = zeros(2*M,2*M,m);
        
        BB = zeros(M,M,m);
        LL = zeros(m,1);
        LR = zeros(m,1);
        RR = zeros(m,1);
        RL = zeros(m,1);
        for numbInter=1:m
            B = b(:,freq_bin,numbInter);
            B_L = B(ref_mics(1));
            B_R = B(ref_mics(2));
            
            e = c*abs(A_L/A_R - B_L/B_R);
            
            LL(numbInter) = abs(B_R)^2;
            LR(numbInter) = - (B_L')*B_R;
            RL(numbInter) = - (B_R')*B_L;
            RR(numbInter) = (abs(B_L)^2 - (abs(B_R)^2)*(e^2));
            
            BB(:,:,numbInter) = b(:,freq_bin,numbInter)*b(:,freq_bin,numbInter)';
        end
        
        tilda_P_n = [P_n zeros(size(P_n)); zeros(size(P_n)) P_n];
        
        if(c < 1 && c > 0 && m~=0)
            for numbInter=1:m
                BigMatrix(:,:,numbInter) = [LL(numbInter).*BB(:,:,numbInter) LR(numbInter).*BB(:,:,numbInter); RL(numbInter).*BB(:,:,numbInter) RR(numbInter).*BB(:,:,numbInter)];
            end
            
            
            cvx_begin sdp;
            cvx_solver sedumi
            cvx_precision best;
            cvx_quiet(true);
            variable w_hat(2*M,1) complex;
            variable X(2*M,2*M) complex semidefinite;
            minimize( real(trace(X*tilda_P_n)) )
            subject to
            for number_interferers=1:m
                real(trace(X*BigMatrix(:,:,number_interferers))) <= 0;
            end
            Lambda_1'*w_hat == f_1;
            0.5*([X w_hat; w_hat' 1] + [X w_hat; w_hat' 1]') >= 0;
            cvx_end
            
            w_L(:,freq_bin) = w_hat(1:length(w_hat)/2);
            w_R(:,freq_bin) = w_hat(length(w_hat)/2+1:end);
            
        elseif(c>=1 || m==0)
            % BMVDR
            temp = tilda_P_n \ Lambda_1;
            w_BMVDR = temp * ( ((Lambda_1')*temp) \ f_1 );
            w_L(:,freq_bin) = w_BMVDR(1:length(w_BMVDR)/2);
            w_R(:,freq_bin) = w_BMVDR(length(w_BMVDR)/2+1:end);
        elseif(c==0 && m~=0)
            % JBLCMV
            f = [f_1; zeros(m,1)];
            
            Lambda_2 = zeros(2*M,m);
            for numbInter=1:m
                B = b(:,freq_bin,numbInter);
                B_L = B(ref_mics(1));
                B_R = B(ref_mics(2));
                Lambda_2(:,numbInter) = [B.*B_R; -B.*B_L];
            end
            
            Lambda = [Lambda_1 Lambda_2];
            
            temp = tilda_P_n \ Lambda;
            w_hat = temp * ( ((Lambda')*temp) \ f );
            w_L(:,freq_bin) = w_hat(1:length(w_hat)/2);
            w_R(:,freq_bin) = w_hat(length(w_hat)/2+1:end);
        end
        
    else
        w_L = w_L_prev;
        w_R = w_R_prev;
    end
    
    x_L_fft(freq_bin) = w_L(:,freq_bin)'*y(:,freq_bin);
    x_R_fft(freq_bin) = w_R(:,freq_bin)'*y(:,freq_bin);
    
    %ITF at the input
    ITF_s_in(freq_bin) = a(ref_mics(1),freq_bin)/a(ref_mics(2),freq_bin);
    for numbInter=1:m
        ITF_int_in(freq_bin,numbInter) = b(ref_mics(1),freq_bin,numbInter)/b(ref_mics(2),freq_bin,numbInter);
    end
    
    %ITF at the output
    ITF_s_out(freq_bin) = (w_L(:,freq_bin)'*a(:,freq_bin))/(w_R(:,freq_bin)'*a(:,freq_bin));
    for numbInter=1:m
        ITF_int_out(freq_bin,numbInter) = (w_L(:,freq_bin)'*b(:,freq_bin,numbInter))/(w_R(:,freq_bin)'*b(:,freq_bin,numbInter));
    end
    
    SNR_in_num(freq_bin) = (e_L'*P_x*e_L + e_R'*P_x*e_R);
    SNR_in_den(freq_bin) = (e_L'*P_n*e_L + e_R'*P_n*e_R);
    SNR_out_num(freq_bin) = (w_L(:,freq_bin)'*P_x*w_L(:,freq_bin) + w_R(:,freq_bin)'*P_x*w_R(:,freq_bin));
    SNR_out_den(freq_bin) = (w_L(:,freq_bin)'*P_n*w_L(:,freq_bin) + w_R(:,freq_bin)'*P_n*w_R(:,freq_bin));
end

SNR_in = 10*log10(abs(sum(SNR_in_num)/sum(SNR_in_den)));
SNR_out = 10*log10(abs(sum(SNR_out_num)/sum(SNR_out_den)));

x_L_fft(NFFT/2+2:end) = flipud(conj(x_L_fft(2:end/2)).').';
x_R_fft(NFFT/2+2:end) = flipud(conj(x_R_fft(2:end/2)).').';

x_L = real(ifft(x_L_fft,NFFT,'symmetric'));
x_L = x_L(1:N);  % remove the zero padding
x_L = x_L(:);

x_R = real(ifft(x_R_fft,NFFT,'symmetric'));
x_R = x_R(1:N);  % remove the zero padding
x_R = x_R(:);

end
function [x_L,x_R,w_L,w_R, ITF_s_in, ITF_int_in, ITF_s_out, ITF_int_out, SNR_in, SNR_out] = BMVDR(y,a,b,N,P,Px,ref_mics,w_L_prev,w_R_prev,Fs)
% Purpose: This function implements the joint BLCMV (JBLCMV) beamformer.
%
% Author: Andreas Koutrouvelis

[M,NFFT_small] = size(y);
NFFT = 2*NFFT_small-2;
w_L = zeros(M,NFFT/2+1);
w_R = zeros(M,NFFT/2+1);
x_L_fft = zeros(NFFT,1);
x_R_fft = zeros(NFFT,1);
[notused1,notused2,r] = size(b);


ITF_s_in = zeros(NFFT/2+1,1);
ITF_int_in = zeros(NFFT/2+1,r);
ITF_s_out = zeros(NFFT/2+1,1);
ITF_int_out = zeros(NFFT/2+1,r);
SNR_in_num = zeros(NFFT/2+1,1);
SNR_in_den = zeros(NFFT/2+1,1);
SNR_out_num = zeros(NFFT/2+1,1);
SNR_out_den = zeros(NFFT/2+1,1);
e_L = zeros(M,1);
e_L(ref_mics(1)) = 1;
e_R = zeros(M,1);
e_R(ref_mics(2)) = 1;


for freq_bin=1:NFFT/2+1
    if(isempty(w_L_prev) || isempty(w_R_prev))
        A = a(:,freq_bin);
        P_n = P(:,:,freq_bin);
        P_x = Px(:,:,freq_bin);
        
        A_L = A(ref_mics(1));
        A_R = A(ref_mics(2));
        
        Lambda = [A zeros(size(A)); zeros(size(A)) A];
        f = [A_L A_R]';
        
        tilda_P_n = [P_n zeros(size(P_n)); zeros(size(P_n)) P_n];
        
        temp = tilda_P_n \ Lambda;
        w_hat = temp * ( ((Lambda')*temp) \ f );
        
        w_L(:,freq_bin) = w_hat(1:length(w_hat)/2);
        w_R(:,freq_bin) = w_hat(length(w_hat)/2+1:end);
    else
        w_L = w_L_prev;
        w_R = w_R_prev;
        P_n = P(:,:,freq_bin);
        P_x = Px(:,:,freq_bin);
    end
    
    x_L_fft(freq_bin) = w_L(:,freq_bin)'*y(:,freq_bin);
    x_R_fft(freq_bin) = w_R(:,freq_bin)'*y(:,freq_bin);
    
    %ITF at the input
    ITF_s_in(freq_bin) = a(ref_mics(1),freq_bin)/a(ref_mics(2),freq_bin);
    for numbInter=1:r
        ITF_int_in(freq_bin,numbInter) = b(ref_mics(1),freq_bin,numbInter)/b(ref_mics(2),freq_bin,numbInter);
    end
    
    %ITF at the output
    ITF_s_out(freq_bin) = (w_L(:,freq_bin)'*a(:,freq_bin))/(w_R(:,freq_bin)'*a(:,freq_bin));
    for numbInter=1:r
        ITF_int_out(freq_bin,numbInter) = (w_L(:,freq_bin)'*b(:,freq_bin,numbInter))/(w_R(:,freq_bin)'*b(:,freq_bin,numbInter));
    end
    
    SNR_in_num(freq_bin) = (e_L'*P_x*e_L + e_R'*P_x*e_R);
    SNR_in_den(freq_bin) = (e_L'*P_n*e_L + e_R'*P_n*e_R);
    SNR_out_num(freq_bin) = (w_L(:,freq_bin)'*P_x*w_L(:,freq_bin) + w_R(:,freq_bin)'*P_x*w_R(:,freq_bin));
    SNR_out_den(freq_bin) = (w_L(:,freq_bin)'*P_n*w_L(:,freq_bin) + w_R(:,freq_bin)'*P_n*w_R(:,freq_bin));
    
end

SNR_in = 10*log10(abs(sum(SNR_in_num)/sum(SNR_in_den)));
SNR_out = 10*log10(abs(sum(SNR_out_num)/sum(SNR_out_den)));

x_L_fft(NFFT/2+2:end) = flipud(conj(x_L_fft(2:end/2)).').';
x_R_fft(NFFT/2+2:end) = flipud(conj(x_R_fft(2:end/2)).').';


x_L = real(ifft(x_L_fft,NFFT,'symmetric'));
x_L = x_L(1:N);  % remove the zero padding
x_L = x_L(:);

x_R = real(ifft(x_R_fft,NFFT,'symmetric'));
x_R = x_R(1:N);  % remove the zero padding
x_R = x_R(:);



end
function [x_L,x_R,w_L,w_R, ITF_s_in, ITF_int_in, ITF_s_out, ITF_int_out, SNR_in, SNR_out] = JBLCMV(y,a,b,N,P,Px,ref_mics,w_L_prev,w_R_prev)
% Purpose: This function implements the joint BLCMV (JBLCMV) beamformer.
%
% Author: Andreas Koutrouvelis

[M,NFFT_small] = size(y);
NFFT = 2*NFFT_small-2;
w_L = zeros(M,NFFT/2+1);
w_R = zeros(M,NFFT/2+1);
x_L_fft = zeros(NFFT,1);
x_R_fft = zeros(NFFT,1);
[notused1,notused2,r] = size(b);

m = min(r,2*M-3);  % This is used when we have many sources.

ITF_s_in = zeros(NFFT/2+1,1);
ITF_int_in = zeros(NFFT/2+1,r);
ITF_s_out = zeros(NFFT/2+1,1);
ITF_int_out = zeros(NFFT/2+1,r);
SNR_in_num = zeros(NFFT/2+1,1);
SNR_in_den = zeros(NFFT/2+1,1);
SNR_out_num = zeros(NFFT/2+1,1);
SNR_out_den = zeros(NFFT/2+1,1);
e_L = zeros(M,1);
e_L(ref_mics(1)) = 1;
e_R = zeros(M,1);
e_R(ref_mics(2)) = 1;

for freq_bin=1:NFFT/2+1
    if(isempty(w_L_prev) || isempty(w_R_prev))
        A = a(:,freq_bin);
        P_n = P(:,:,freq_bin);
        P_x = Px(:,:,freq_bin);
        
        A_L = A(ref_mics(1));
        A_R = A(ref_mics(2));
        
        Lambda_1 = [A zeros(size(A)); zeros(size(A)) A];
        f_1 = [A_L A_R]';
        
        Lambda_2 = zeros(2*M,m);
        for numbInter=1:m
            B = b(:,freq_bin,numbInter);
            B_L = B(ref_mics(1));
            B_R = B(ref_mics(2));
            Lambda_2(:,numbInter) = [B.*B_R; -B.*B_L];
        end
        f_2 = zeros(m,1);
        
        Lambda = [Lambda_1 Lambda_2];
        f = [f_1; f_2];
        
        tilda_P_n = [P_n zeros(size(P_n)); zeros(size(P_n)) P_n];
        
        temp = tilda_P_n \ Lambda;
        w_hat = temp * ( ((Lambda')*temp) \ f );
        
        w_L(:,freq_bin) = w_hat(1:length(w_hat)/2);
        w_R(:,freq_bin) = w_hat(length(w_hat)/2+1:end);
        
        
    else
        w_L = w_L_prev;
        w_R = w_R_prev;
        P_n = P(:,:,freq_bin);
        P_x = Px(:,:,freq_bin);
    end
    
    x_L_fft(freq_bin) = w_L(:,freq_bin)'*y(:,freq_bin);
    x_R_fft(freq_bin) = w_R(:,freq_bin)'*y(:,freq_bin);
    
    %ITF at the input
    ITF_s_in(freq_bin) = a(ref_mics(1),freq_bin)/a(ref_mics(2),freq_bin);
    for numbInter=1:r
        ITF_int_in(freq_bin,numbInter) = b(ref_mics(1),freq_bin,numbInter)/b(ref_mics(2),freq_bin,numbInter);
    end
    
    %ITF at the output
    ITF_s_out(freq_bin) = (w_L(:,freq_bin)'*a(:,freq_bin))/(w_R(:,freq_bin)'*a(:,freq_bin));
    for numbInter=1:r
        ITF_int_out(freq_bin,numbInter) = (w_L(:,freq_bin)'*b(:,freq_bin,numbInter))/(w_R(:,freq_bin)'*b(:,freq_bin,numbInter));
    end
    
    SNR_in_num(freq_bin) = (e_L'*P_x*e_L + e_R'*P_x*e_R);
    SNR_in_den(freq_bin) = (e_L'*P_n*e_L + e_R'*P_n*e_R);
    SNR_out_num(freq_bin) = (w_L(:,freq_bin)'*P_x*w_L(:,freq_bin) + w_R(:,freq_bin)'*P_x*w_R(:,freq_bin));
    SNR_out_den(freq_bin) = (w_L(:,freq_bin)'*P_n*w_L(:,freq_bin) + w_R(:,freq_bin)'*P_n*w_R(:,freq_bin));
end

SNR_in = 10*log10(abs(sum(SNR_in_num)/sum(SNR_in_den)));
SNR_out = 10*log10(abs(sum(SNR_out_num)/sum(SNR_out_den)));

x_L_fft(NFFT/2+2:end) = flipud(conj(x_L_fft(2:end/2)).').';
x_R_fft(NFFT/2+2:end) = flipud(conj(x_R_fft(2:end/2)).').';


x_L = real(ifft(x_L_fft,NFFT,'symmetric'));
x_L = x_L(1:N);  % remove the zero padding
x_L = x_L(:);

x_R = real(ifft(x_R_fft,NFFT,'symmetric'));
x_R = x_R(1:N);  % remove the zero padding
x_R = x_R(:);


end
function [x_L,x_R,w_L,w_R, ITF_s_in, ITF_int_in, ITF_s_out, ITF_int_out, SNR_in, SNR_out] = BBILD_SDCR(y,a,b,N,P,Px,Bin_1500,ref_mics,r,w_L_prev,w_R_prev)
% Purpose: This function implements the RJBLCMV_SDCR beamformer.
%
% Author: Andreas Koutrouvelis
[M,NFFT_small] = size(y);
NFFT = 2*NFFT_small-2;
w_L = zeros(M,NFFT/2+1);
w_R = zeros(M,NFFT/2+1);
x_L_fft = zeros(NFFT,1);
x_R_fft = zeros(NFFT,1);


m = min(r,2*M-3);

ITF_s_in = zeros(NFFT/2+1,1);
ITF_int_in = zeros(NFFT/2+1,r);
ITF_s_out = zeros(NFFT/2+1,1);
ITF_int_out = zeros(NFFT/2+1,r);
SNR_in_num = zeros(NFFT/2+1,1);
SNR_in_den = zeros(NFFT/2+1,1);
SNR_out_num = zeros(NFFT/2+1,1);
SNR_out_den = zeros(NFFT/2+1,1);
e_L = zeros(M,1);
e_L(ref_mics(1)) = 1;
e_R = zeros(M,1);
e_R(ref_mics(2)) = 1;

for freq_bin=1:NFFT/2+1
    if(isempty(w_L_prev) || isempty(w_R_prev))
        A = a(:,freq_bin);
        P_n = P(:,:,freq_bin);
        P_n = 0.5*(P_n + P_n');  % makes sure that the CPSDM is Hermitian
        P_x = Px(:,:,freq_bin);
        P_x = 0.5*(P_x + P_x');
        
        A_L = A(ref_mics(1));
        A_R = A(ref_mics(2));
        
        Lambda_1 = [A zeros(size(A)); zeros(size(A)) A];
        f_1 = [A_L A_R]';
        
        BigMatrix = zeros(2*M,2*M,m);
        
        BB = zeros(M,M,m);
        LL = zeros(m,1);
        RR = zeros(m,1);
        
        Lambda_2 = zeros(2*M,m);
        for numbInter=1:m
            B = b(:,freq_bin,numbInter);
            B_L = B(ref_mics(1));
            B_R = B(ref_mics(2));
            
            Lambda_2(:,numbInter) = [B.*B_R; -B.*B_L];
            
            LL(numbInter) = abs(B_R)^2;
            RR(numbInter) = -abs(B_L)^2;
            
            BB(:,:,numbInter) = b(:,freq_bin,numbInter)*b(:,freq_bin,numbInter)';
        end
        f_2 = zeros(m,1);
        Lambda = [Lambda_1 Lambda_2];
        f = [f_1; f_2];
        
        tilda_P_n = [P_n zeros(size(P_n)); zeros(size(P_n)) P_n];
        if(Bin_1500(freq_bin))
            
            for numbInter=1:m
                BigMatrix(:,:,numbInter) = [BB(:,:,numbInter).*LL(numbInter) ,zeros(M);zeros(M), BB(:,:,numbInter).*RR(numbInter)];
            end
         
            cvx_begin sdp;
            cvx_solver sedumi
            cvx_precision best;
            cvx_quiet(true);
            
            variable w_hat(2*M,1) complex;
            variable X(2*M,2*M) complex semidefinite;
            minimize( real(trace(X*tilda_P_n)) )
            subject to
            for number_interferers=1:m
                real(trace(X*BigMatrix(:,:,number_interferers))) == 0;
            end
            Lambda_1'*w_hat == f_1;
            trace(X*(Lambda_1*Lambda_1')) - w_hat'*Lambda_1*f_1 - f_1'*Lambda_1'*w_hat + f_1'*f_1 == 0;
            0.5*([X w_hat; w_hat' 1] + [X w_hat; w_hat' 1]') >= 0;
            cvx_end
        else
            temp = tilda_P_n \ Lambda;
            w_hat = temp * ( ((Lambda')*temp) \ f );
        end
        w_L(:,freq_bin) = w_hat(1:length(w_hat)/2);
        w_R(:,freq_bin) = w_hat(length(w_hat)/2+1:end);
        
    else
        w_L = w_L_prev;
        w_R = w_R_prev;
        P_n = P(:,:,freq_bin);
        P_n = 0.5*(P_n + P_n');  % makes sure that the CPSDM is Hermitian
        P_x = Px(:,:,freq_bin);
        P_x = 0.5*(P_x + P_x');
    end
    
    x_L_fft(freq_bin) = w_L(:,freq_bin)'*y(:,freq_bin);
    x_R_fft(freq_bin) = w_R(:,freq_bin)'*y(:,freq_bin);
    
    %ITF at the input
    ITF_s_in(freq_bin) = a(ref_mics(1),freq_bin)/a(ref_mics(2),freq_bin);
    for numbInter=1:r
        ITF_int_in(freq_bin,numbInter) = b(ref_mics(1),freq_bin,numbInter)/b(ref_mics(2),freq_bin,numbInter);
    end
    
    %ITF at the output
    ITF_s_out(freq_bin) = (w_L(:,freq_bin)'*a(:,freq_bin))/(w_R(:,freq_bin)'*a(:,freq_bin));
    for numbInter=1:r
        ITF_int_out(freq_bin,numbInter) = (w_L(:,freq_bin)'*b(:,freq_bin,numbInter))/(w_R(:,freq_bin)'*b(:,freq_bin,numbInter));
    end
    
    SNR_in_num(freq_bin) = (e_L'*P_x*e_L + e_R'*P_x*e_R);
    SNR_in_den(freq_bin) = (e_L'*P_n*e_L + e_R'*P_n*e_R);
    SNR_out_num(freq_bin) = (w_L(:,freq_bin)'*P_x*w_L(:,freq_bin) + w_R(:,freq_bin)'*P_x*w_R(:,freq_bin));
    SNR_out_den(freq_bin) = (w_L(:,freq_bin)'*P_n*w_L(:,freq_bin) + w_R(:,freq_bin)'*P_n*w_R(:,freq_bin));
end
SNR_in = 10*log10(abs(sum(SNR_in_num)/sum(SNR_in_den)));
SNR_out = 10*log10(abs(sum(SNR_out_num)/sum(SNR_out_den)));

x_L_fft(NFFT/2+2:end) = flipud(conj(x_L_fft(2:end/2)).').';
x_R_fft(NFFT/2+2:end) = flipud(conj(x_R_fft(2:end/2)).').';

x_L = real(ifft(x_L_fft,NFFT,'symmetric'));
x_L = x_L(1:N);  % remove the zero padding
x_L = x_L(:);

x_R = real(ifft(x_R_fft,NFFT,'symmetric'));
x_R = x_R(1:N);  % remove the zero padding
x_R = x_R(:);

end
function [x_L,x_R,w_L,w_R, ITF_s_in, ITF_int_in, ITF_s_out, ITF_int_out, SNR_in, SNR_out] = RBBILD_SDCR(y,a,b,N,P,Px,Bin_1500,ref_mics,c,w_L_prev,w_R_prev)
% Purpose: This function implements the RJBLCMV_SDCR beamformer.
%
% Author: Andreas Koutrouvelis
[M,NFFT_small] = size(y);
NFFT = 2*NFFT_small-2;
w_L = zeros(M,NFFT/2+1);
w_R = zeros(M,NFFT/2+1);
x_L_fft = zeros(NFFT,1);
x_R_fft = zeros(NFFT,1);
[~,~,r] = size(b);

if(c == 0 && r > 2*M-3)
    m = 2*M-3;
else
    m = r;
end

ITF_s_in = zeros(NFFT/2+1,1);
ITF_int_in = zeros(NFFT/2+1,m);
ITF_s_out = zeros(NFFT/2+1,1);
ITF_int_out = zeros(NFFT/2+1,m);
SNR_in_num = zeros(NFFT/2+1,1);
SNR_in_den = zeros(NFFT/2+1,1);
SNR_out_num = zeros(NFFT/2+1,1);
SNR_out_den = zeros(NFFT/2+1,1);
e_L = zeros(M,1);
e_L(ref_mics(1)) = 1;
e_R = zeros(M,1);
e_R(ref_mics(2)) = 1;

for freq_bin=1:NFFT/2+1
    P_n = P(:,:,freq_bin);
    P_n = 0.5*(P_n + P_n');  % makes sure that the CPSDM is Hermitian
    P_x = Px(:,:,freq_bin);
    P_x = 0.5*(P_x + P_x');
    if(isempty(w_L_prev) || isempty(w_R_prev))
        A = a(:,freq_bin);
        
        A_L = A(ref_mics(1));
        A_R = A(ref_mics(2));
        
        Lambda_1 = [A zeros(size(A)); zeros(size(A)) A];
        f_1 = [A_L A_R]';
        
        BigMatrix1 = zeros(2*M,2*M,m);
        BigMatrix2 = zeros(2*M,2*M,m);
        
        BB = zeros(M,M,m);
        LL1 = zeros(m,1);
        LL2 = zeros(m,1);
        RR1 = zeros(m,1);
        RR2 = zeros(m,1);
        Lambda_2 = zeros(2*M,m);
        
        for numbInter=1:m
            B = b(:,freq_bin,numbInter);
            B_L = B(ref_mics(1));
            B_R = B(ref_mics(2));
            
            Lambda_2(:,numbInter) = [B.*B_R; -B.*B_L];
            
            e = c*(abs(abs(A_L/A_R).^2 - abs(B_L/B_R).^2));
            
            LL1(numbInter) = abs(B_R)^2;
            LL2(numbInter) = -abs(B_R)^2;
            RR1(numbInter) = -(abs(B_L)^2 + (e*abs(B_R)^2));
            RR2(numbInter) = (abs(B_L)^2 - (e*abs(B_R)^2));
            
            BB(:,:,numbInter) = b(:,freq_bin,numbInter)*b(:,freq_bin,numbInter)';
            
        end
        f_2 = zeros(m,1);
        Lambda = [Lambda_1 Lambda_2];
        f = [f_1; f_2];
        
        tilda_P_n = [P_n zeros(size(P_n)); zeros(size(P_n)) P_n];
        
        
        
        if(c < 1 && c > 0 && m~=0)
            if(Bin_1500(freq_bin))
                for numbInter=1:m
                    BigMatrix1(:,:,numbInter) = [BB(:,:,numbInter).*LL1(numbInter) ,zeros(M);zeros(M), BB(:,:,numbInter).*RR1(numbInter)];
                    BigMatrix2(:,:,numbInter) = [BB(:,:,numbInter).*LL2(numbInter) ,zeros(M);zeros(M), BB(:,:,numbInter).*RR2(numbInter)];
                end
                              
                cvx_begin sdp;
                cvx_solver sedumi
                cvx_precision best;
                cvx_quiet(true);
                variable w_hat(2*M,1) complex;
                variable X(2*M,2*M) complex semidefinite;
                minimize( real(trace(X*tilda_P_n)) )
                subject to
                for number_interferers=1:m
                    real(trace(X*BigMatrix1(:,:,number_interferers))) <= 0;
                    real(trace(X*BigMatrix2(:,:,number_interferers))) <= 0;
                end
                Lambda_1'*w_hat == f_1;
                trace(X*(Lambda_1*Lambda_1')) - w_hat'*Lambda_1*f_1 - f_1'*Lambda_1'*w_hat + f_1'*f_1 == 0;
                0.5*([X w_hat; w_hat' 1] + [X w_hat; w_hat' 1]') >= 0;
                cvx_end
            else
                temp = tilda_P_n \ Lambda;
                w_hat = temp * ( ((Lambda')*temp) \ f );
            end
            w_L(:,freq_bin) = w_hat(1:length(w_hat)/2);
            w_R(:,freq_bin) = w_hat(length(w_hat)/2+1:end);
            
        elseif(c>=1 || m==0)
            % BMVDR
            temp = tilda_P_n \ Lambda_1;
            w_BMVDR = temp * ( ((Lambda_1')*temp) \ f_1 );
            w_L(:,freq_bin) = w_BMVDR(1:length(w_BMVDR)/2);
            w_R(:,freq_bin) = w_BMVDR(length(w_BMVDR)/2+1:end);
        elseif(c==0 && m~=0)
            % JBLCMV
            f = [f_1; zeros(m,1)];
            
            Lambda_2 = zeros(2*M,m);
            for numbInter=1:m
                B = b(:,freq_bin,numbInter);
                B_L = B(ref_mics(1));
                B_R = B(ref_mics(2));
                Lambda_2(:,numbInter) = [B.*B_R; -B.*B_L];
            end
            
            Lambda = [Lambda_1 Lambda_2];
            
            temp = tilda_P_n \ Lambda;
            w_hat = temp * ( ((Lambda')*temp) \ f );
            w_L(:,freq_bin) = w_hat(1:length(w_hat)/2);
            w_R(:,freq_bin) = w_hat(length(w_hat)/2+1:end);
        end
        
    else
        w_L = w_L_prev;
        w_R = w_R_prev;
    end
    
    x_L_fft(freq_bin) = w_L(:,freq_bin)'*y(:,freq_bin);
    x_R_fft(freq_bin) = w_R(:,freq_bin)'*y(:,freq_bin);
    
    %ITF at the input
    ITF_s_in(freq_bin) = a(ref_mics(1),freq_bin)/a(ref_mics(2),freq_bin);
    for numbInter=1:m
        ITF_int_in(freq_bin,numbInter) = b(ref_mics(1),freq_bin,numbInter)/b(ref_mics(2),freq_bin,numbInter);
    end
    
    %ITF at the output
    ITF_s_out(freq_bin) = (w_L(:,freq_bin)'*a(:,freq_bin))/(w_R(:,freq_bin)'*a(:,freq_bin));
    for numbInter=1:m
        ITF_int_out(freq_bin,numbInter) = (w_L(:,freq_bin)'*b(:,freq_bin,numbInter))/(w_R(:,freq_bin)'*b(:,freq_bin,numbInter));
    end
    
    SNR_in_num(freq_bin) = (e_L'*P_x*e_L + e_R'*P_x*e_R);
    SNR_in_den(freq_bin) = (e_L'*P_n*e_L + e_R'*P_n*e_R);
    SNR_out_num(freq_bin) = (w_L(:,freq_bin)'*P_x*w_L(:,freq_bin) + w_R(:,freq_bin)'*P_x*w_R(:,freq_bin));
    SNR_out_den(freq_bin) = (w_L(:,freq_bin)'*P_n*w_L(:,freq_bin) + w_R(:,freq_bin)'*P_n*w_R(:,freq_bin));
end

SNR_in = 10*log10(abs(sum(SNR_in_num)/sum(SNR_in_den)));
SNR_out = 10*log10(abs(sum(SNR_out_num)/sum(SNR_out_den)));

x_L_fft(NFFT/2+2:end) = flipud(conj(x_L_fft(2:end/2)).').';
x_R_fft(NFFT/2+2:end) = flipud(conj(x_R_fft(2:end/2)).').';

x_L = real(ifft(x_L_fft,NFFT,'symmetric'));
x_L = x_L(1:N);  % remove the zero padding
x_L = x_L(:);

x_R = real(ifft(x_R_fft,NFFT,'symmetric'));
x_R = x_R(1:N);  % remove the zero padding
x_R = x_R(:);

end
function [x_L,x_R,w_L,w_R, ITF_s_in, ITF_int_in, ITF_s_out, ITF_int_out, SNR_in, SNR_out] = RBBILD_SDCR_4dB(y,a,b,N,P,Px,Bin_1500,ref_mics,c,w_L_prev,w_R_prev)
% Purpose: This function implements the RJBLCMV_SDCR beamformer.
%
% Author: Andreas Koutrouvelis
[M,NFFT_small] = size(y);
NFFT = 2*NFFT_small-2;
w_L = zeros(M,NFFT/2+1);
w_R = zeros(M,NFFT/2+1);
x_L_fft = zeros(NFFT,1);
x_R_fft = zeros(NFFT,1);
[~,~,r] = size(b);

m = min(2*M -3, r);
c = 10^(c/10);

ITF_s_in = zeros(NFFT/2+1,1);
ITF_int_in = zeros(NFFT/2+1,m);
ITF_s_out = zeros(NFFT/2+1,1);
ITF_int_out = zeros(NFFT/2+1,m);
SNR_in_num = zeros(NFFT/2+1,1);
SNR_in_den = zeros(NFFT/2+1,1);
SNR_out_num = zeros(NFFT/2+1,1);
SNR_out_den = zeros(NFFT/2+1,1);
e_L = zeros(M,1);
e_L(ref_mics(1)) = 1;
e_R = zeros(M,1);
e_R(ref_mics(2)) = 1;

for freq_bin=1:NFFT/2+1
    P_n = P(:,:,freq_bin);
    P_n = 0.5*(P_n + P_n');  % makes sure that the CPSDM is Hermitian
    P_x = Px(:,:,freq_bin);
    P_x = 0.5*(P_x + P_x');
    if(isempty(w_L_prev) || isempty(w_R_prev))
        A = a(:,freq_bin);
        
        A_L = A(ref_mics(1));
        A_R = A(ref_mics(2));
        
        Lambda_1 = [A zeros(size(A)); zeros(size(A)) A];
        f_1 = [A_L A_R]';
        
        BigMatrix1 = zeros(2*M,2*M,m);
        BigMatrix2 = zeros(2*M,2*M,m);
        
        BB = zeros(M,M,m);
        LL1 = zeros(m,1);
        LL2 = zeros(m,1);
        RR1 = zeros(m,1);
        RR2 = zeros(m,1);
        Lambda_2 = zeros(2*M,m);
        
        for numbInter=1:m
            B = b(:,freq_bin,numbInter);
            B_L = B(ref_mics(1));
            B_R = B(ref_mics(2));
            
            Lambda_2(:,numbInter) = [B.*B_R; -B.*B_L];
            
                      
            LL1(numbInter) = abs(B_R)^2;
            LL2(numbInter) = -c*abs(B_R)^2;
            RR1(numbInter) = -c*abs(B_L)^2;
            RR2(numbInter) = abs(B_L)^2;
            
            BB(:,:,numbInter) = b(:,freq_bin,numbInter)*b(:,freq_bin,numbInter)';
            
        end
        f_2 = zeros(m,1);
        Lambda = [Lambda_1 Lambda_2];
        f = [f_1; f_2];
        
        tilda_P_n = [P_n zeros(size(P_n)); zeros(size(P_n)) P_n];
        
        if(Bin_1500(freq_bin))
            for numbInter=1:m
                BigMatrix1(:,:,numbInter) = [BB(:,:,numbInter).*LL1(numbInter) ,zeros(M);zeros(M), BB(:,:,numbInter).*RR1(numbInter)];
                BigMatrix2(:,:,numbInter) = [BB(:,:,numbInter).*LL2(numbInter) ,zeros(M);zeros(M), BB(:,:,numbInter).*RR2(numbInter)];
            end
            
            cvx_begin sdp;
            cvx_solver sedumi
            cvx_precision best;
            cvx_quiet(true);
            variable w_hat(2*M,1) complex;
            variable X(2*M,2*M) complex semidefinite;
            minimize( real(trace(X*tilda_P_n)) )
            subject to
            for number_interferers=1:m
                real(trace(X*BigMatrix1(:,:,number_interferers))) <= 0;
                real(trace(X*BigMatrix2(:,:,number_interferers))) <= 0;
            end
            Lambda_1'*w_hat == f_1;
            trace(X*(Lambda_1*Lambda_1')) - w_hat'*Lambda_1*f_1 - f_1'*Lambda_1'*w_hat + f_1'*f_1 == 0;
            0.5*([X w_hat; w_hat' 1] + [X w_hat; w_hat' 1]') >= 0;
            cvx_end
        else
            temp = tilda_P_n \ Lambda;
            w_hat = temp * ( ((Lambda')*temp) \ f );
        end
        w_L(:,freq_bin) = w_hat(1:length(w_hat)/2);
        w_R(:,freq_bin) = w_hat(length(w_hat)/2+1:end);
    else
        w_L = w_L_prev;
        w_R = w_R_prev;
    end
    
    x_L_fft(freq_bin) = w_L(:,freq_bin)'*y(:,freq_bin);
    x_R_fft(freq_bin) = w_R(:,freq_bin)'*y(:,freq_bin);
    
    %ITF at the input
    ITF_s_in(freq_bin) = a(ref_mics(1),freq_bin)/a(ref_mics(2),freq_bin);
    for numbInter=1:m
        ITF_int_in(freq_bin,numbInter) = b(ref_mics(1),freq_bin,numbInter)/b(ref_mics(2),freq_bin,numbInter);
    end
    
    %ITF at the output
    ITF_s_out(freq_bin) = (w_L(:,freq_bin)'*a(:,freq_bin))/(w_R(:,freq_bin)'*a(:,freq_bin));
    for numbInter=1:m
        ITF_int_out(freq_bin,numbInter) = (w_L(:,freq_bin)'*b(:,freq_bin,numbInter))/(w_R(:,freq_bin)'*b(:,freq_bin,numbInter));
    end
    
    SNR_in_num(freq_bin) = (e_L'*P_x*e_L + e_R'*P_x*e_R);
    SNR_in_den(freq_bin) = (e_L'*P_n*e_L + e_R'*P_n*e_R);
    SNR_out_num(freq_bin) = (w_L(:,freq_bin)'*P_x*w_L(:,freq_bin) + w_R(:,freq_bin)'*P_x*w_R(:,freq_bin));
    SNR_out_den(freq_bin) = (w_L(:,freq_bin)'*P_n*w_L(:,freq_bin) + w_R(:,freq_bin)'*P_n*w_R(:,freq_bin));
end

SNR_in = 10*log10(abs(sum(SNR_in_num)/sum(SNR_in_den)));
SNR_out = 10*log10(abs(sum(SNR_out_num)/sum(SNR_out_den)));

x_L_fft(NFFT/2+2:end) = flipud(conj(x_L_fft(2:end/2)).').';
x_R_fft(NFFT/2+2:end) = flipud(conj(x_R_fft(2:end/2)).').';

x_L = real(ifft(x_L_fft,NFFT,'symmetric'));
x_L = x_L(1:N);  % remove the zero padding
x_L = x_L(:);

x_R = real(ifft(x_R_fft,NFFT,'symmetric'));
x_R = x_R(1:N);  % remove the zero padding
x_R = x_R(:);

end

function [ITF_error_int, ITD_error_int_below, ITD_error_int_above, ILD_error_int_below, ILD_error_int_above, ILD_error_int_dB, ILD_error_int_freq,ILD_error_int_freq_dB, Avg_BMVDR_ILD,Avg_BMVDR_ILD_dB, gsSNR_in, gsSNR_out] = computeMeasures(ITF_s_in,ITF_s_out,ITF_int_in,ITF_int_out,BMVDR_ITF_in,BMVDR_ITF_out,c,SNR_in,SNR_out,frequency)
% Purpose: This function computes the average ITF , ILD, IPD error for source and interferers.
%
% Author: Vasudha Sathyapriyan
Bin_1500 = frequency>=1500;

ITF_error_s = abs(ITF_s_out -ITF_s_in);
ITF_error_s_f = mean(ITF_error_s,[2]);
ITF_error_s = mean(ITF_error_s,[2 1]);
ILD_error_s  = abs(abs(ITF_s_out).^2 -(abs(ITF_s_in).^2));
ILD_error_s_f = mean(ILD_error_s,[2]);
ILD_error_s = sum(mean(ILD_error_s_f(Bin_1500 == 1,:),1));
ILD_error_s_dB  = abs(10*log10(abs(ITF_s_out).^2) - 10*log10(abs(ITF_s_in).^2));
ILD_error_s_f_dB = mean(ILD_error_s_dB,[2]);
ILD_error_s_dB = sum(mean(ILD_error_s_f_dB(Bin_1500 == 1,:),1));


ITD_error_s = abs(angle(ITF_s_out) - angle(ITF_s_in))/pi;
ITD_error_s_f = mean(ITD_error_s,[2]);
ITD_error_s = sum(mean(ITD_error_s_f(Bin_1500 == 0,:),1));

ITF_error_int = abs(ITF_int_out -ITF_int_in);
ITF_error_int_f = mean(ITF_error_int,[3]);
ITF_error_int = sum(mean(ITF_error_int_f,[1]),2);

ILD_error_int  = abs(abs(ITF_int_out).^2 -(abs(ITF_int_in).^2));
ILD_error_int_f = mean(ILD_error_int,[3]);
ILD_error_int_freq = sum(ILD_error_int_f,2);
ILD_error_int_above = sum(mean(ILD_error_int_f(Bin_1500 == 1,:),1));
ILD_error_int_below = sum(mean(ILD_error_int_f(Bin_1500 == 0,:),1));
ILD_error_int_dB  = abs(10*log10(abs(ITF_int_out).^2) - 10*log10(abs(ITF_int_in).^2));
ILD_error_int_f_dB = mean(ILD_error_int_dB,[3]);
ILD_error_int_freq_dB = sum(ILD_error_int_f_dB,2);
ILD_error_int_dB = sum(mean(ILD_error_int_f_dB(Bin_1500 == 1,:),1));

BMVDR_ILD  = c*abs(abs(BMVDR_ITF_out).^2 -(abs(BMVDR_ITF_in).^2));
Avg_BMVDR_ILD = sum(mean(BMVDR_ILD(Bin_1500 == 1,:),1));

BMVDR_ILD_dB = c*abs(10*log10(abs(BMVDR_ITF_out).^2) - 10*log10(abs(BMVDR_ITF_in).^2));
Avg_BMVDR_ILD_dB = sum(mean(BMVDR_ILD_dB(Bin_1500 == 1,:),1));

ITD_error_int = abs(angle(ITF_int_out) - angle(ITF_int_in))/pi;
ITD_error_int_f = mean(ITD_error_int,[3 2]);
ITD_error_int_below =  sum(mean(ITD_error_int_f(Bin_1500 == 0,:),1));
ITD_error_int_above =  sum(mean(ITD_error_int_f(Bin_1500 == 1,:),1));

gsSNR_in = mean(min(max(SNR_in,-20),50));
gsSNR_out = mean(min(max(SNR_out,-20),50));

%To analyse the ILD in frequency for ILD/R_ILD
% Markers = {'-bv','-g*','-kx','-cv','-md','-r^','-ks'};
% Legend  = {'Inter_1','Inter_2','Inter_3','Inter_4','Inter_5','Inter_6','Inter_7'};
% [~,numbInter,~] = size(ITF_int_in);
% figure
% hold on;
% for inter = 1:numbInter
%     plot(frequency(Bin_1500 == 1), ILD_error_int_f(Bin_1500 == 1,inter),Markers{inter});
% end
% title('ITF Error/interferer/freq');
% xlabel('Frequency (Hz)');
% xlim([0 10]);
% legend(Legend{1:numbInter});
% hold off;
end

function [] = plotMeasures(ITF_pro,ITD_pro_below,ITD_pro_above,ILD_pro_below, ILD_pro_above,ILD_pro_dB,ILD_pro_freq,ILD_pro_freq_dB,Avg_BMVDR_ILD,Avg_BMVDR_ILD_dB, gsSNR_in,gsSNR_out,stoiL,stoiR,mbStoi,stoi_unprocessed_L,stoi_unprocessed_R,mb_stoi_unprocessed,fwSegSNR_L, fwSegSNR_R,fwSegSNR_L_in, fwSegSNR_R_in,frequency)
% Purpose: This function plots the average ITF , ILD, IPD error for source and interferers.
%
% Author: Vasudha Sathyapriyan

[numMethods,numbInters] = size(ITF_pro);
Markers = {'-bv','-g*','-kx','-cv','-md','-r^','-ks'};
Legend  = {'Unprocessed','1.BMVDR','2.JBLCMV','3.ILD','4.ILD_{relaxed}','5.ILD_{relaxed}-4dB''6.RBLCMV_{SCO}','7.RBLCMV_{SDCR}','Avg-epsilon'};

for method = 1:numMethods
    if ((method == 4) || (method == 3))
        for r = 1:numbInters
            figure;
            hold on;
            stem(frequency, ILD_pro_freq(method,:,r));
            title(['ILD over frequency for method =' num2str(method) ' r = ' num2str(r)]);
            xlabel('Frequnecy (Hz)');
            ylabel('ILD error');
            hold off;
        end
    end
end

for method = 1:numMethods
    if ((method == 4) || (method == 3))
        for r = 1:numbInters
            figure;
            hold on;
            stem(frequency, ILD_pro_freq_dB(method,:,r));
            title(['ILD(dB) over frequency for method =' num2str(method) ' r = ' num2str(r)]);
            xlabel('Frequnecy (Hz)');
            ylabel('ILD error (dB)');
            hold off;
        end
    end
end
figure
hold on;
for method = 1:numMethods
    plot([1:numbInters], ITF_pro(method,:),Markers{method});
end
title('Total ITF Error');
ylabel('Total ITF Error');
xlabel('No. of interferers (r)');
legend(Legend{2:numMethods+1});
hold off;

figure;
hold on;
for method = 1:numMethods
    plot([1:numbInters], ITD_pro_below(method,:),Markers{method});
end
title('Total ITD Error for f < 1.5kHz');
ylabel('Total ITD Error');
xlabel('No. of interferers (r)');
legend(Legend{2:numMethods+1});

hold off;

figure;
hold on;
for method = 1:numMethods
    plot([1:numbInters], ITD_pro_above(method,:),Markers{method});
end
title('Total ITD Error for f >= 1.5kHz');
ylabel('Total ITD Error');
xlabel('No. of interferers (r)');
legend(Legend{2:numMethods+1});

hold off;

figure;
hold on;
for method = 1:numMethods
    plot([1:numbInters], ILD_pro_below(method,:),Markers{method});
end
title('Total ILD Error for f < 1.5kHz');
ylabel('Total ILD Error');
xlabel('No. of interferers (r)');
legend(Legend{2:numMethods+1});
hold off;

figure;
hold on;
for method = 1:numMethods
    plot([1:numbInters], ILD_pro_above(method,:),Markers{method});
end
plot([1:numbInters], Avg_BMVDR_ILD(1,:),'-g');
title('Total ILD Error for f >= 1.5kHz');
ylabel('Total ILD Error');
xlabel('No. of interferers (r)');
legend(Legend{[2:numMethods+1 ,8]});
hold off;

figure;
hold on;
for method = 1:numMethods
    plot([1:numbInters], ILD_pro_dB(method,:),Markers{method});
end
plot([1:numbInters], Avg_BMVDR_ILD_dB(1,:),'-g');
title('Total ILD Error (Abs)');
ylabel('Total ILD Error(dB)');
xlabel('No. of interferers (r)');
legend(Legend{[2:numMethods+1,8]});
hold off;

figure;
hold on;
plot([1:numbInters],gsSNR_in(1,:),'bo');
for method = 1:numMethods
    plot([1:numbInters], gsSNR_out(method,:),Markers{method});
end
title('gsSNR');
xlabel('No. of interferers (r)');
ylabel('gsSNR (dB)')
legend(Legend{1:numMethods+1});
hold off;

figure;
hold on;
for method = 1:numMethods
    plot([1:numbInters], gsSNR_out(method,:) -gsSNR_in(1,:),Markers{method});
end
title('gsSNR Gain');
xlabel('No. of interferers (r)');
ylabel('gsSNR Gain(dB)')
legend(Legend{2:numMethods+1});
hold off;

figure;
hold on;
plot([1:numbInters],fwSegSNR_L_in(1,:),'bo');
for method = 1:numMethods
    plot([1:numbInters], fwSegSNR_L(method,:),Markers{method});
end
title('fwSegSNR_L');
xlabel('No. of interferers (r)');
ylabel('fwSegSNR (dB)')
legend(Legend{1:numMethods+1});
hold off;

figure;
hold on;
for method = 1:numMethods
    plot([1:numbInters], fwSegSNR_L(method,:) - fwSegSNR_L_in(1,:),Markers{method});
end
title('fwSegSNR_L Gain');
xlabel('No. of interferers (r)');
ylabel('fwSegSNR Gain(dB)')
legend(Legend{2:numMethods+1});
hold off;

figure;
hold on;
plot([1:numbInters],fwSegSNR_R_in(1,:),'bo');
for method = 1:numMethods
    plot([1:numbInters], fwSegSNR_R(method,:),Markers{method});
end
title('fwSegSNR_R');
xlabel('No. of interferers (r)');
ylabel('fwSegSNR (dB)')
legend(Legend{1:numMethods+1});
hold off;

figure;
hold on;
for method = 1:numMethods
    plot([1:numbInters], fwSegSNR_R(method,:) - fwSegSNR_R_in(1,:),Markers{method});
end
title('fwSegSNR_R Gain');
xlabel('No. of interferers (r)');
ylabel('fwSegSNR Gain(dB)')
legend(Legend{2:numMethods+1});
hold off;


figure;
hold on;
for method = 1:numMethods
    plot(ILD_pro_above(method,:), gsSNR_out(method,:),Markers{method});
end
title('Total ILD error Vs gsSNR ');
xlabel('ILD error');
ylabel('gsSNR (dB)')
legend(Legend{2:numMethods+1});
hold off;

figure;
hold on;
for method = 1:numMethods
    plot(ILD_pro_dB(method,:), gsSNR_out(method,:),Markers{method});
end
title('Abs. Total ILD error(dB) Vs gsSNR ');
xlabel('Abs. ILD error(dB)');
ylabel('gsSNR (dB)')
legend(Legend{2:numMethods+1});
hold off;

figure;
hold on;
plot([1:numbInters], stoi_unprocessed_L(1,:),'bo');
for method = 1:numMethods
    plot([1:numbInters], stoiL(method,:),Markers{method});
end
title('STOI_L');
xlabel('No. of interferers (r)');
legend(Legend{1:numMethods+1});
ylim([0 1]);
hold off;

figure;
hold on;
plot([1:numbInters], stoi_unprocessed_R(1,:),'bo');
for method = 1:numMethods
    plot([1:numbInters], stoiR(method,:),Markers{method});
end
title('STOI_R');
xlabel('No. of interferers (r)');
legend(Legend{1:numMethods+1});
ylim([0 1]);
hold off;

figure;
hold on;
plot([1:numbInters], mb_stoi_unprocessed(1,:),'bo');
for method = 1:numMethods
    plot([1:numbInters], mbStoi(method,:),Markers{method});
end
title('MBSTOI');
xlabel('No. of interferers (r)');
legend(Legend{1:numMethods+1});
ylim([0 1]);
hold off;
end

function [] = plotMeasuresC(ILD_pro,gsSNR_out,r)
% Purpose: This function plots the average ITF , ILD, IPD error for source and interferers.
%
% Author: Vasudha Sathyapriyan

figure;
hold on;
plot(ILD_pro, gsSNR_out,'-bv');
title(['Total ILD error Vs gsSNR r =' num2str(r)]);
xlabel('Total ILD error');
ylabel('gsSNR (dB)')
hold off;

end

function [BMVDR_ITF_in,BMVDR_ITF_out] = computBMVDR_ITF(a,b,ref_mics)
[M,NFFT_small,r] = size(b);
NFFT = 2*NFFT_small-2;

m = r;

BMVDR_ITF_in = zeros(NFFT/2+1,m);
BMVDR_ITF_out = zeros(NFFT/2+1,m);

for freq_bin = 1: (NFFT/2 +1)
    A = a(:,freq_bin);
    A_L =   A(ref_mics(1)) ;
    A_R =   A(ref_mics(2)) ;
    for numbInter=1:m
        B = b(:,freq_bin,numbInter);
        B_L = B(ref_mics(1));
        B_R = B(ref_mics(2));
        
        BMVDR_ITF_in(freq_bin,numbInter) = B_L/B_R;
        BMVDR_ITF_out(freq_bin,numbInter) = A_L/A_R;
        
    end
end
end