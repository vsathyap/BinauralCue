function vd = idealVAD(s_in,thresh,block_size)
% Purpose: It detects signal (s_in) presence/absence.
%
% Args:  
%       1) s_in (Lsig x 1):  examined signal.
%
%       2) thresh (1 x 1):   threshold that is used for the detection.
%
%       3) block_size (1 x 1):  the detection outcome is based on 
%                               time-blocks of multiple samples with length
%                               block_size.
%
% Return:
%       1) vd (Lsig x 1):  consists of zeros and ones. Zero means that s_in
%                          is absent, while one means that s_in is present.
%
%
% Author: Andreas Koutrouvelis
%
% Last modified: 28/03/2019
%
% Copyright (C) 2019  Andreas Koutrouvelis
%
% This program is free software: you can redistribute it and/or modify
% it under the terms of the GNU General Public License as published by
% the Free Software Foundation, either version 3 of the License, or
% (at your option) any later version.

% This program is distributed in the hope that it will be useful,
% but WITHOUT ANY WARRANTY; without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
% GNU General Public License for more details.

% You should have received a copy of the GNU General Public License
% along with this program.  If not, see <https://www.gnu.org/licenses/>.
[cols,rows] = size(s_in);

s = zeros(rows,1);

s = s_in(1,:);


vd = zeros(rows,1);

vd(find(abs(s)>thresh))=1;

L_vd = length(vd);

slice = 1:block_size;

Nfr = L_vd/block_size;

for i=1:Nfr
    if(sum(vd(slice)) <= 5)
        vd(slice) = 0;
    else
        vd(slice) = 1;
    end
    
    slice = slice + block_size;
end