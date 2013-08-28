function create_quadrature_filters_for_structure_tensor_computation3d(varargin)
% CREATE_QUADRATURE_FILTERS_FOR_STRUCTURE_TESNOR_COMPUTATION3D 
% Creates 3D quadrature filters for structure tensor estimation
%
% create_quadrature_filters_for_struncture_tensor_computation3d()
%
% Optional input arguments
% 'scale'               - Scale determines for which center frequency the 
%                         filters should be optimized. Available options are:
%                         'coarse','intermediate','fine', corresponding to
%                         pi*1/3, pi*sqrt(2)/3, pi*2/3, where 'intermediate' is default
% 'showFilters'         - true/false
%
%
% See "Advanced filter design" by Knutsson et al for detailed explanation
% of the parameters used.
%
% See also F_weightgensnr, goodsw, quadrature, krnopt

% Copyright (c) 2012 Daniel Forsberg
% Division of Medical Informatics 
% Department of Biomedical Engineering
% Linkoping University
%
% This program is free software: you can redistribute it and/or modify
% it under the terms of the GNU General Public License as published by
% the Free Software Foundation, either version 3 of the License, or
% (at your option) any later version.
%
% This program is distributed in the hope that it will be useful,
% but WITHOUT ANY WARRANTY; without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
% GNU General Public License for more details.
%
% You should have received a copy of the GNU General Public License
% along with this program.  If not, see <http://www.gnu.org/licenses/>.

filePath = which('create_quadrature_filters_for_structure_tensor_computation3d.m');
filePath = filePath(1:strfind(filePath,'create_quadrature_filters_for_structure_tensor_computation3d')-1);
currentPath = pwd;

cd(filePath);

%% Set deafult parameters

scale = 'intermediate';
showFilters = false;

%% Overwrites default parameter
for k=1:2:length(varargin)
  eval([varargin{k},'=varargin{',int2str(k+1),'};']);
end

switch scale
    
    case 'coarse'
        disp('Coarse')
        % Filter parameters
        n = 9;
        spatial_rexp = 2;
        frequency_rexp = -1.0;
        cosexp = 1;
        SNRest = 30;
        DCamp = 1000;
        fw_amp = 30;
        
        % Center frequency and bandwidth (in octaves)
        u0 = pi/3;
        B = 2.0;
    case 'intermediate'
        disp('Intermediate')
        % Filter parameters
        n = 7;
        spatial_rexp = 2;
        frequency_rexp = -1.0;
        cosexp = 1;
        SNRest = 30;
        DCamp = 1000;
        fw_amp = 30;
        
        % Center frequency and bandwidth (in octaves)
        u0 = pi*sqrt(2)/3;
        B = 2.0;
    case 'fine'
        disp('fine')
        % Filter parameters
        n = 7;
        spatial_rexp = 2;
        frequency_rexp = -1.0;
        cosexp = 1;
        SNRest = 30;
        DCamp = 1000;
        fw_amp = 30;
        
        % Center frequency and bandwidth (in octaves)
        u0 = pi*2/3;
        B = 2.0;
    otherwise
        error('Undefined scale')
end

fprintf('Optimizing quadrature filters for structure tensor computation in 3d\n');

% Sizes
spatialSize = [n n n];
frequencySize = 2*spatialSize+1;

% Frequency weights
Fw = F_weightgensnr(frequencySize,frequency_rexp,cosexp,SNRest,DCamp);

% Spatial weights
fw0 = goodsw(spatialSize,spatial_rexp);
fw = fw_amp.*fw0;

% Spatial ideal filter
fi = wa(spatialSize,0);
fi = putorigo(fi,1);

% Spatial mask
fm = wa(spatialSize,0);
fm = putdata(fm,ones(spatialSize));

% Filter directions
a = 2;
b = 1 + sqrt(5);
c = 1/sqrt(10 + 2*sqrt(5));

dir{1} = c * [ 0  a  b];
dir{2} = c * [ 0 -a  b];
dir{3} = c * [ a  b  0];
dir{4} = c * [-a  b  0];
dir{5} = c * [ b  0  a];
dir{6} = c * [ b  0 -a];

%filter_directions_x = [dir{1}(1) dir{2}(1) dir{3}(1) dir{4}(1) dir{5}(1) dir{6}(1)];
%filter_directions_y = [dir{1}(2) dir{2}(2) dir{3}(2) dir{4}(2) dir{5}(2) dir{6}(2)];
%filter_directions_z = [dir{1}(3) dir{2}(3) dir{3}(3) dir{4}(3) dir{5}(3) dir{6}(3)];

filter_directions_x = [dir{1}(2) dir{2}(2) dir{3}(2) dir{4}(2) dir{5}(2) dir{6}(2)];
filter_directions_y = [dir{1}(1) dir{2}(1) dir{3}(1) dir{4}(1) dir{5}(1) dir{6}(1)];
filter_directions_z = [dir{1}(3) dir{2}(3) dir{3}(3) dir{4}(3) dir{5}(3) dir{6}(3)];

%dir{1} = c * [ a  0  b];
%dir{2} = c * [-a  0  b];
%dir{3} = c * [ b  a  0];
%dir{4} = c * [ b  -a  0];
%dir{5} = c * [ 0  b  a];
%dir{6} = c * [ 0  b -a];

% Frequency ideal filters
for k = 1 : length(dir)
    Fi{k} = quadrature(frequencySize,u0,B,dir{k});
end

% Optimize the quadrature filters
for k = 1 : length(dir)
    [f{k},F{k}] = krnopt(Fi{k},Fw,fm,fi,fw);
    if showFilters
        filterr(F{k},Fi{k},Fw,fi,fw,1);
        fprintf('\n')
        sfigure(188); mesh3d( real(dft(f{k} ,[21,21,21])))
        drawnow
        input('Press any key to continue...\n')
    end
end

for k = 1 : length(dir)
    f{k} = getdata(f{k});
end

% Create dual tensors
for k = 1 : length(dir)
    M = 5/4.*dir{k}'*dir{k} - 1/4.*eye(3);
    m11{k} = M(1,1);
    m12{k} = M(1,2);
    m13{k} = M(1,3);
    m22{k} = M(2,2);
    m23{k} = M(2,3);
    m33{k} = M(3,3);
end

% Remember filter parameters
filterInfo.n = n;
filterInfo.spatial_rexp = spatial_rexp;
filterInfo.frequency_rexp = frequency_rexp;
filterInfo.cosexp = cosexp;
filterInfo.SNRest = SNRest;
filterInfo.DCamp = DCamp;
filterInfo.fw_amp = fw_amp;
filterInfo.u0 = u0;
filterInfo.B = B;

f1_nonparametric_registration = f{1};
f2_nonparametric_registration = f{2};
f3_nonparametric_registration = f{3};
f4_nonparametric_registration = f{4};
f5_nonparametric_registration = f{5};
f6_nonparametric_registration = f{6};

m11
m12
m13
m22
m23
m33

m1 = [m11{1} m12{1} m13{1} m22{1} m23{1} m33{1}];
m2 = [m11{2} m12{2} m13{2} m22{2} m23{2} m33{2}];
m3 = [m11{3} m12{3} m13{3} m22{3} m23{3} m33{3}];
m4 = [m11{4} m12{4} m13{4} m22{4} m23{4} m33{4}];
m5 = [m11{5} m12{5} m13{5} m22{5} m23{5} m33{5}];
m6 = [m11{6} m12{6} m13{6} m22{6} m23{6} m33{6}];

save filters_for_nonparametric_registration f1_nonparametric_registration f2_nonparametric_registration f3_nonparametric_registration f4_nonparametric_registration f5_nonparametric_registration f6_nonparametric_registration m1 m2 m3 m4 m5 m6 filter_directions_x filter_directions_y filter_directions_z filterInfo

cd(currentPath)

