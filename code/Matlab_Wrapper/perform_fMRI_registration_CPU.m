function [corrected_volumes registration_parameters, rotations, scalings, filter_response_1, filter_response_2, filter_response_3] = perform_fMRI_registration_CPU(fMRI_volumes,f1,f2,f3,number_of_iterations)

filter_size = size(f1,1);

reference_volume = fMRI_volumes(:,:,:,1);

[sy sx sz st] = size(fMRI_volumes)

corrected_volumes = zeros(sy,sx,sz,st);
corrected_volumes(:,:,:,1) = reference_volume;

registration_parameterss = zeros(st,12);
rotationss = zeros(st,3);
scalingss = zeros(st,3);

% % Filter parameters
% spatial_rexp = 2;
% frequency_rexp = -1;
% cosexp = 1;
% SNRest = 30;
% 
% % Center frequency and bandwidth (in octaves)
% u0 = pi/3;
% B = 1.7;
% 
% % Sizes
% spatial_size = [filter_size filter_size filter_size];
% frequency_size = 2*spatial_size+1;
% 
% % Frequency weights
% Fw = F_weightgensnr(frequency_size,frequency_rexp,cosexp,SNRest);
% 
% % Spatial weights
% fw0 = goodsw(spatial_size,spatial_rexp);
% %wamesh(fw0)
% fw_amp = 30;
% fw = fw_amp.*fw0;
% 
% % Spatial ideal filter
% fi = wa(spatial_size,0);
% fi = putorigo(fi,1);
% 
% % Spatial mask
% fm = wa(spatial_size,0);
% fm = putdata(fi,ones(spatial_size));
% 
% % Filter directions
% dir{1} = [0 1 0]'; % x
% dir{2} = [1 0 0]'; % y
% dir{3} = [0 0 1]'; % z
% 
% % Frequency ideal filters
% for k = 1:3    
%     Fi{k} = quadrature(frequency_size,u0,B,dir{k});
% end
% 
% % Optimize the quadrature filters
% for k = 1:3
%     [f{k},F{k}] = krnopt(Fi{k},Fw,fm,fi,fw);
% end
% 
% filterr(F{1},Fi{1},Fw,fi,fw,1);
% 
% f1 = getdata(f{1});
% f2 = getdata(f{2});
% f3 = getdata(f{3});

F1 = zeros(sy,sx,sz);
F2 = zeros(sy,sx,sz);
F3 = zeros(sy,sx,sz);
% 
% % OBS F�RSKJUTNINGEN BEROR P� FILTERSTORLEKEN!!
% 
% % F�r filterstorlek 3
% %f1_fft((sy - 1)/2 + 1: (sy - 1)/2 + filter_size - 0, (sx - 1)/2 + 1: (sx - 1)/2 + filter_size - 0,(sz - 1)/2 + 1: (sz - 1)/2 + filter_size - 0) = f1; 
% %f2_fft((sy - 1)/2 + 1: (sy - 1)/2 + filter_size - 0, (sx - 1)/2 + 1: (sx - 1)/2 + filter_size - 0,(sz - 1)/2 + 1: (sz - 1)/2 + filter_size - 0) = f2; 
% %f3_fft((sy - 1)/2 + 1: (sy - 1)/2 + filter_size - 0, (sx - 1)/2 + 1: (sx - 1)/2 + filter_size - 0,(sz - 1)/2 + 1: (sz - 1)/2 + filter_size - 0) = f3; 
% 
if filter_size == 5
% 
%     % F�r filterstorlek 5
     F1((sy - 0)/2 - 0: (sy - 0)/2 + filter_size - 1, (sx - 0)/2 - 0: (sx - 0)/2 + filter_size - 1,(sz - 0)/2 - 0: (sz - 0)/2 + filter_size - 1) = f1; 
     F2((sy - 0)/2 - 0: (sy - 0)/2 + filter_size - 1, (sx - 0)/2 - 0: (sx - 0)/2 + filter_size - 1,(sz - 0)/2 - 0: (sz - 0)/2 + filter_size - 1) = f2; 
     F3((sy - 0)/2 - 0: (sy - 0)/2 + filter_size - 1, (sx - 0)/2 - 0: (sx - 0)/2 + filter_size - 1,(sz - 0)/2 - 0: (sz - 0)/2 + filter_size - 1) = f3; 
% 
elseif filter_size == 7
%     
%     % F�r filterstorlek 7
     F1((sy - 0)/2 - 1: (sy - 0)/2 + filter_size - 2, (sx - 0)/2 - 1: (sx - 0)/2 + filter_size - 2,(sz - 0)/2 - 1: (sz - 0)/2 + filter_size - 2) = f1; 
     F2((sy - 0)/2 - 1: (sy - 0)/2 + filter_size - 2, (sx - 0)/2 - 1: (sx - 0)/2 + filter_size - 2,(sz - 0)/2 - 1: (sz - 0)/2 + filter_size - 2) = f2; 
     F3((sy - 0)/2 - 1: (sy - 0)/2 + filter_size - 2, (sx - 0)/2 - 1: (sx - 0)/2 + filter_size - 2,(sz - 0)/2 - 1: (sz - 0)/2 + filter_size - 2) = f3; 
end 
% elseif filter_size == 9
% 
%     % F�r filterstorlek 9
%     F1((sy - 0)/2 - 2: (sy - 0)/2 + filter_size - 3, (sx - 0)/2 - 2: (sx - 0)/2 + filter_size - 3,(sz - 0)/2 - 2: (sz - 0)/2 + filter_size - 3) = f1; 
%     F2((sy - 0)/2 - 2: (sy - 0)/2 + filter_size - 3, (sx - 0)/2 - 2: (sx - 0)/2 + filter_size - 3,(sz - 0)/2 - 2: (sz - 0)/2 + filter_size - 3) = f2; 
%     F3((sy - 0)/2 - 2: (sy - 0)/2 + filter_size - 3, (sx - 0)/2 - 2: (sx - 0)/2 + filter_size - 3,(sz - 0)/2 - 2: (sz - 0)/2 + filter_size - 3) = f3; 
% 
% end

F1 = fftn(F1);
F2 = fftn(F2);
F3 = fftn(F3);

quadrature_filters.F1 = F1;
quadrature_filters.F2 = F2;
quadrature_filters.F3 = F3;

% FFT Convolution
REFERENCE_VOLUME = fftn(reference_volume);

filter_response_1 = fftshift(ifftn(REFERENCE_VOLUME .* F1));
filter_response_2 = fftshift(ifftn(REFERENCE_VOLUME .* F2));
filter_response_3 = fftshift(ifftn(REFERENCE_VOLUME .* F3));

q11 = filter_response_1( (filter_size - 1)/2 + 1:end - (filter_size - 1)/2, (filter_size - 1)/2 + 1:end - (filter_size - 1)/2, (filter_size - 1)/2 + 1:end - (filter_size - 1)/2);
q12 = filter_response_2( (filter_size - 1)/2 + 1:end - (filter_size - 1)/2, (filter_size - 1)/2 + 1:end - (filter_size - 1)/2, (filter_size - 1)/2 + 1:end - (filter_size - 1)/2);
q13 = filter_response_3( (filter_size - 1)/2 + 1:end - (filter_size - 1)/2, (filter_size - 1)/2 + 1:end - (filter_size - 1)/2, (filter_size - 1)/2 + 1:end - (filter_size - 1)/2);
   
quadrature_responses_reference_volume.q11 = q11;
quadrature_responses_reference_volume.q12 = q12;
quadrature_responses_reference_volume.q13 = q13;

[sy sx sz] = size(filter_response_1);
%[x, y, z] = meshgrid(-(sx-1)/2:(sx-1)/2,-(sy-1)/2:(sy-1)/2, -(sz-1)/2:(sz-1)/2);
%B = build_b_matrix(sx,sy,sz,x,y,z);
B = 0;
[sy sx sz] = size(reference_volume);
[x, y, z] = meshgrid(-(sx-1)/2:(sx-1)/2,-(sy-1)/2:(sy-1)/2, -(sz-1)/2:(sz-1)/2);

start_time = clock;

% Make the registrations

for t = 2:st
    t
    altered_volume = fMRI_volumes(:,:,:,t);
    [compensated_volume,registration_parameters,rotations,scalings] = Phasebased_3D_registration_FFT(reference_volume,altered_volume,quadrature_responses_reference_volume,quadrature_filters,x,y,z,number_of_iterations,filter_size);
    corrected_volumes(:,:,:,t) = compensated_volume;
    registration_parameterss(t,:) = registration_parameters;   
    rotationss(t,:) = rotations;
    scalingss(t,:) = scalings;
    t
end    
registration_time = etime(clock,start_time)
time_per_volume = registration_time / (st - 1)



registration_parameters = registration_parameterss;
rotations = rotationss;
scalings = scalingss;


