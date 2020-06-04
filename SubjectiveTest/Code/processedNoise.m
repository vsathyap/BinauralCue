function [Y_N_L, Y_N_R] = processedNoise(E_N,N,NFFT,Ws_L,Ws_R)

[M,Lsig,numbInter] = size(E_N);
[~,~,~,numMethods] = size(Ws_L);
Shift = N/2;
Nfr = floor(Lsig/Shift);
Y_N_L = zeros(Shift*Nfr,numbInter,numMethods);
Y_N_R = zeros(Shift*Nfr,numbInter,numMethods);

for met = 1:numMethods
    for r = 1:numbInter
        slice = 1:N;
        Win = sqrt(hanning(N));
        Shift = N/2;
        Nfr = floor(Lsig/Shift);
        
        Buffer_d_hat_L = zeros(Shift,1);
        Buffer_d_hat_R = zeros(Shift,1);
        tosave = 1:Shift;
        x_hat_L = zeros(Shift*Nfr,1);
        x_hat_R = zeros(Shift*Nfr,1);
        
        x_L_fft = zeros(NFFT,1);
        x_R_fft = zeros(NFFT,1);
        
        %%%%%%%%%%%%%%%%%%% enahancement frame by frame %%%%%%%%%%%%%%%%%%%%%%%
        for l=1:Nfr-1
            N_frame = E_N(:,slice,r);
            for mic_i = 1:M
                N_frame(mic_i,:) = (Win').*N_frame(mic_i,:);
            end
            
            N_frame_fft = obtainFFTs(N_frame,NFFT);
            N_frame_fft = N_frame_fft(:,1:NFFT/2+1);
            
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
            for freq_bin=1:NFFT/2+1
                x_L_fft(freq_bin) = Ws_L(:,freq_bin,l,met)'*N_frame_fft(:,freq_bin);
                x_R_fft(freq_bin) = Ws_R(:,freq_bin,l,met)'*N_frame_fft(:,freq_bin);
            end
            x_L_fft(NFFT/2+2:end) = flipud(conj(x_L_fft(2:end/2)).').';
            x_R_fft(NFFT/2+2:end) = flipud(conj(x_R_fft(2:end/2)).').';
            
            x_L = real(ifft(x_L_fft,NFFT,'symmetric'));
            x_L = x_L(1:N);  % remove the zero padding
            x_L = x_L(:);
            
            x_R = real(ifft(x_R_fft,NFFT,'symmetric'));
            x_R = x_R(1:N);  % remove the zero padding
            x_R = x_R(:);
            
            
            x_hat_frame_win_L = Win.*x_L;
            x_hat_frame_win_R = Win.*x_R;
            
            
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
        Y_N_L(:,r,met) = x_hat_L(:);
        Y_N_R(:,r,met) = x_hat_R(:);
    end
end
end

function N_frame_fft = obtainFFTs(N_frame,NFFT)

N_frame_fft = fft(N_frame.',NFFT).';

end