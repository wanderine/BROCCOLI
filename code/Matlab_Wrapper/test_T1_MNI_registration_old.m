%  	 BROCCOLI: An open source multi-platform software for parallel analysis of fMRI data on many core CPUs and GPUS
%    Copyright (C) <2013>  Anders Eklund, andek034@gmail.com
%
%    This program is free software: you can redistribute it and/or modify
%    it under the terms of the GNU General Public License as published by
%    the Free Software Foundation, either version 3 of the License, or
%    (at your option) any later version.
%
%    This program is distributed in the hope that it will be useful,
%    but WITHOUT ANY WARRANTY; without even the implied warranty of
%    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
%    GNU General Public License for more details.
%
%    You should have received a copy of the GNU General Public License
%    along with this program.  If not, see <http://www.gnu.org/licenses/>.
%-----------------------------------------------------------------------------

%---------------------------------------------------------------------------------------------------------------------
% README
% If you run this code in Windows, your graphics driver might stop working
% for large volumes / large filter sizes. This is not a bug in my code but is due to the
% fact that the Nvidia driver thinks that something is wrong if the GPU
% takes more than 2 seconds to complete a task. This link solved my problem
% https://forums.geforce.com/default/topic/503962/tdr-fix-here-for-nvidia-driver-crashing-randomly-in-firefox/
%---------------------------------------------------------------------------------------------------------------------

clear all
clc
close all

addpath('D:\nifti_matlab')

%mex RegisterT1MNI.cpp -lOpenCL -lBROCCOLI_LIB -IC:/Program' Files'/NVIDIA' GPU Computing Toolkit'/CUDA/v5.0/include -IC:/Program' Files'/NVIDIA' GPU Computing Toolkit'/CUDA/v5.0/include/CL -LC:/Program' Files'/NVIDIA' GPU Computing Toolkit'/CUDA/v5.0/lib/x64 -LC:/users/wande/Documents/Visual' Studio 2010'/Projects/BROCCOLI_LIB/x64/Release/ -IC:/users/wande/Documents/Visual' Studio 2010'/Projects/BROCCOLI_LIB/BROCCOLI_LIB -IC:\Users\wande\Documents\Visual' Studio 2010'\Projects\BROCCOLI_LIB\nifticlib-2.0.0\niftilib  -IC:\Users\wande\Documents\Visual' Studio 2010'\Projects\BROCCOLI_LIB\nifticlib-2.0.0\znzlib

%mex -g RegisterT1MNI.cpp -lOpenCL -lBROCCOLI_LIB -IC:/Program' Files'/NVIDIA' GPU Computing Toolkit'/CUDA/v5.0/include -IC:/Program' Files'/NVIDIA' GPU Computing Toolkit'/CUDA/v5.0/include/CL -LC:/Program' Files'/NVIDIA' GPU Computing Toolkit'/CUDA/v5.0/lib/x64 -LC:/users/wande/Documents/Visual' Studio 2010'/Projects/BROCCOLI_LIB/x64/Debug/ -IC:/users/wande/Documents/Visual' Studio 2010'/Projects/BROCCOLI_LIB/BROCCOLI_LIB -IC:\Users\wande\Documents\Visual' Studio 2010'\Projects\BROCCOLI_LIB\nifticlib-2.0.0\niftilib  -IC:\Users\wande\Documents\Visual' Studio 2010'\Projects\BROCCOLI_LIB\nifticlib-2.0.0\znzlib

T1_nii = load_nii('../../test_data/T1_AE.nii');
T1 = double(T1_nii.img);
T1 = T1(1:182,1:218,105:end);
MNI_nii = load_nii('../../test_data/MNI152_T1_1mm.nii');
MNI = double(MNI_nii.img);
MNI = MNI(:,:,1:end-30);
[sy sx sz] = size(T1)
[sy sx sz] = size(MNI)

mask_nii = load_nii('../../test_data/MNI152_T1_1mm_brain_mask.nii');
mask = double(mask_nii.img);
mask = mask(:,:,1:end-30);

load filters.mat

T1_ = T1(1:4:end,1:4:end,1:4:end);
[sy sx sz] = size(T1_)
MNI_ = MNI(1:4:end,1:4:end,1:4:end);

number_of_iterations_for_motion_correction = 40;
tic
[registered_T1_cpu, registration_parameters_cpu1] = perform_T1_MNI_registration_CPU(T1_,MNI_,f1,f2,f3,number_of_iterations_for_motion_correction);
toc

%tic
%[registered_T1_opencl, registration_parameters_opencl, quadrature_filter_response_1_opencl, quadrature_filter_response_2_opencl, quadrature_filter_response_3_opencl, phase_differences_x_opencl, phase_certainties_x_opencl, phase_gradients_x_opencl] = RegisterT1MNI(T1,MNI,f1,f2,f3,number_of_iterations_for_motion_correction);
%toc

slice = 32;

figure
imagesc([ MNI_(:,:,slice)/norm(MNI_(:)) T1_(:,:,slice)/norm(T1_(:)) ] ); colormap gray
figure
imagesc([ MNI_(:,:,slice)/norm(MNI_(:)) registered_T1_cpu(:,:,slice)/norm(registered_T1_cpu(:)) ] ); colormap gray

T1_ = T1(1:2:end,1:2:end,1:2:end);
MNI_ = MNI(1:2:end,1:2:end,1:2:end);
[sy sx sz] = size(T1_)
p = registration_parameters_cpu1;
p(1) = p(1)*2;
p(2) = p(2)*2;
p(3) = p(3)*2;
pp = p
[x, y, z] = meshgrid(-(sx-1)/2:(sx-1)/2,-(sy-1)/2:(sy-1)/2, -(sz-1)/2:(sz-1)/2);
x_motion_vectors = zeros(sy,sx,sz);
y_motion_vectors = zeros(sy,sx,sz);
z_motion_vectors = zeros(sy,sx,sz);
x_motion_vectors(:) = p(1) + [x(:) y(:) z(:)]*p(4:6);
y_motion_vectors(:) = p(2) + [x(:) y(:) z(:)]*p(7:9);
z_motion_vectors(:) = p(3) + [x(:) y(:) z(:)]*p(10:12);

T1__ = interp3(x,y,z,T1_,x+x_motion_vectors,y+y_motion_vectors,z+z_motion_vectors,'linear');    % Generates NaN's
T1__(isnan(T1__)) = 0;

number_of_iterations_for_motion_correction = 10;
tic
[registered_T1_cpu, registration_parameters_cpu2] = perform_T1_MNI_registration_CPU(T1__,MNI_,f1,f2,f3,number_of_iterations_for_motion_correction);
toc

slice = 64;

figure
imagesc([ MNI_(:,:,slice)/norm(MNI_(:)) T1__(:,:,slice)/norm(T1__(:)) ] ); colormap gray
figure
imagesc([ MNI_(:,:,slice)/norm(MNI_(:)) registered_T1_cpu(:,:,slice)/norm(registered_T1_cpu(:)) ] ); colormap gray


[sy sx sz] = size(T1)
p(1:3) = registration_parameters_cpu1(1:3)*4 + registration_parameters_cpu2(1:3)*2;
p(4:12) = registration_parameters_cpu1(4:12) + registration_parameters_cpu2(4:12);
pp = p
[x, y, z] = meshgrid(-(sx-1)/2:(sx-1)/2,-(sy-1)/2:(sy-1)/2, -(sz-1)/2:(sz-1)/2);
x_motion_vectors = zeros(sy,sx,sz);
y_motion_vectors = zeros(sy,sx,sz);
z_motion_vectors = zeros(sy,sx,sz);
x_motion_vectors(:) = p(1) + [x(:) y(:) z(:)]*p(4:6);
y_motion_vectors(:) = p(2) + [x(:) y(:) z(:)]*p(7:9);
z_motion_vectors(:) = p(3) + [x(:) y(:) z(:)]*p(10:12);

T1__ = interp3(x,y,z,T1,x+x_motion_vectors,y+y_motion_vectors,z+z_motion_vectors,'linear');    % Generates NaN's
T1__(isnan(T1__)) = 0;
 
number_of_iterations_for_motion_correction = 10;
tic
[registered_T1_cpu, registration_parameters_cpu] = perform_T1_MNI_registration_CPU(T1__,MNI,f1,f2,f3,number_of_iterations_for_motion_correction);
toc

slice = 128;

figure
imagesc([ MNI(:,:,slice)/norm(MNI(:)) T1(:,:,slice)/norm(T1(:)) ] ); colormap gray
figure
imagesc([ MNI(:,:,slice)/norm(MNI(:)) T1__(:,:,slice)/norm(T1__(:)) ] ); colormap gray
figure
imagesc([ MNI(:,:,slice)/norm(MNI(:)) registered_T1_cpu(:,:,slice)/norm(registered_T1_cpu(:)) ] ); colormap gray


figure
imagesc([ MNI(:,:,slice)/norm(MNI(:))  ] ); colormap gray
figure
imagesc([ registered_T1_cpu(:,:,slice)/norm(registered_T1_cpu(:)) ]); colormap gray
skullstripped = registered_T1_cpu .* mask;

for slice = 1:size(skullstripped,3)
    figure(11)
    imagesc([ registered_T1_cpu(:,:,slice)] ); colormap gray    
    figure(12)
    imagesc([ skullstripped(:,:,slice)] ); colormap gray
    
    %pause(0.1)
    pause
end



%
% quadrature_filter_response_reference_1_cpu = convn(fMRI_volumes(:,:,:,1),f1,'same');
% quadrature_filter_response_reference_2_cpu = convn(fMRI_volumes(:,:,:,1),f2,'same');
% quadrature_filter_response_reference_3_cpu = convn(fMRI_volumes(:,:,:,1),f3,'same');
%
% quadrature_filter_response_aligned_1_cpu = convn(fMRI_volumes(:,:,:,2),f1,'same');
% quadrature_filter_response_aligned_2_cpu = convn(fMRI_volumes(:,:,:,2),f2,'same');
% quadrature_filter_response_aligned_3_cpu = convn(fMRI_volumes(:,:,:,2),f3,'same');
%
% phase_differences_x_cpu = angle(quadrature_filter_response_reference_1_cpu .* conj(quadrature_filter_response_aligned_1_cpu));
%
% phase_certainties_x_cpu = abs(quadrature_filter_response_reference_1_cpu .* quadrature_filter_response_aligned_1_cpu) .* ((cos(phase_differences_x_cpu/2)).^2);
%
% phase_gradients_x_cpu = zeros(sy,sx,sz);
% phase_gradients_x_cpu(:,2:end-1,:) = angle(quadrature_filter_response_reference_1_cpu(:,3:end,:).*conj(quadrature_filter_response_reference_1_cpu(:,2:end-1,:)) + quadrature_filter_response_reference_1_cpu(:,2:end-1,:).*conj(quadrature_filter_response_reference_1_cpu(:,1:end-2,:)) + ...
%     quadrature_filter_response_aligned_1_cpu(:,3:end,:).*conj(quadrature_filter_response_aligned_1_cpu(:,2:end-1,:)) + quadrature_filter_response_aligned_1_cpu(:,2:end-1,:).*conj(quadrature_filter_response_aligned_1_cpu(:,1:end-2,:)));
%
% %
%
%
% slice = 4;
%
% figure
% imagesc( [real(quadrature_filter_response_reference_1_cpu(:,:,slice)) real(quadrature_filter_response_1_opencl(:,:,slice)) ] )
% figure
% imagesc( [real(quadrature_filter_response_reference_2_cpu(:,:,slice)) real(quadrature_filter_response_2_opencl(:,:,slice)) ] )
% figure
% imagesc( [real(quadrature_filter_response_reference_3_cpu(:,:,slice)) real(quadrature_filter_response_3_opencl(:,:,slice)) ] )
%
% figure
% imagesc( [imag(quadrature_filter_response_reference_1_cpu(:,:,slice)) imag(quadrature_filter_response_1_opencl(:,:,slice)) ] )
% figure
% imagesc( [imag(quadrature_filter_response_reference_2_cpu(:,:,slice)) imag(quadrature_filter_response_2_opencl(:,:,slice)) ] )
% figure
% imagesc( [imag(quadrature_filter_response_reference_3_cpu(:,:,slice)) imag(quadrature_filter_response_3_opencl(:,:,slice)) ] )
%
% figure
% imagesc( [phase_differences_x_cpu(:,:,slice) phase_differences_x_opencl(:,:,slice) ] )
% figure
% imagesc( [phase_certainties_x_cpu(:,:,slice) phase_certainties_x_opencl(:,:,slice) ] )
% %figure
% %imagesc( [phase_certainties_x_cpu(:,:,slice) - phase_certainties_x_opencl(:,:,slice) ] ); colorbar
% figure
% imagesc( [phase_gradients_x_cpu(:,:,slice) phase_gradients_x_opencl(:,:,slice) ] )
% figure
% imagesc( [phase_gradients_x_cpu(2:end-1,2:end-1,slice) - phase_gradients_x_opencl(2:end-1,2:end-1,slice) ] ); colorbar
%
%
%
%
% filter_response_1_real_tot_error = sum(abs(real(quadrature_filter_response_aligned_1_cpu(:)) - real(quadrature_filter_response_1_opencl(:))))
% filter_response_1_real_max_error = max(abs(real(quadrature_filter_response_aligned_1_cpu(:)) - real(quadrature_filter_response_1_opencl(:))))
% filter_response_1_imag_tot_error = sum(abs(imag(quadrature_filter_response_aligned_1_cpu(:)) - imag(quadrature_filter_response_1_opencl(:))))
% filter_response_1_imag_max_error = max(abs(imag(quadrature_filter_response_aligned_1_cpu(:)) - imag(quadrature_filter_response_1_opencl(:))))
%
% filter_response_2_real_tot_error = sum(abs(real(quadrature_filter_response_aligned_2_cpu(:)) - real(quadrature_filter_response_2_opencl(:))))
% filter_response_2_real_max_error = max(abs(real(quadrature_filter_response_aligned_2_cpu(:)) - real(quadrature_filter_response_2_opencl(:))))
% filter_response_2_imag_tot_error = sum(abs(imag(quadrature_filter_response_aligned_2_cpu(:)) - imag(quadrature_filter_response_2_opencl(:))))
% filter_response_2_imag_max_error = max(abs(imag(quadrature_filter_response_aligned_2_cpu(:)) - imag(quadrature_filter_response_2_opencl(:))))
%
% filter_response_3_real_tot_error = sum(abs(real(quadrature_filter_response_aligned_3_cpu(:)) - real(quadrature_filter_response_3_opencl(:))))
% filter_response_3_real_max_error = max(abs(real(quadrature_filter_response_aligned_3_cpu(:)) - real(quadrature_filter_response_3_opencl(:))))
% filter_response_3_imag_tot_error = sum(abs(imag(quadrature_filter_response_aligned_3_cpu(:)) - imag(quadrature_filter_response_3_opencl(:))))
% filter_response_3_imag_max_error = max(abs(imag(quadrature_filter_response_aligned_3_cpu(:)) - imag(quadrature_filter_response_3_opencl(:))))
%
% phase_differences_x_tot_error = sum(abs(phase_differences_x_cpu(:) - phase_differences_x_opencl(:)))
% phase_differences_x_max_error = max(abs(phase_differences_x_cpu(:) - phase_differences_x_opencl(:)))
%
% phase_certainties_x_tot_error = sum(abs(phase_certainties_x_cpu(:) - phase_certainties_x_opencl(:)))
% phase_certainties_x_max_error = max(abs(phase_certainties_x_cpu(:) - phase_certainties_x_opencl(:)))
%
% phase_gradients_x_tot_error = sum(sum(sum(abs(phase_gradients_x_cpu(:,2:end-1,:) - phase_gradients_x_opencl(:,2:end-1,:)))))
% phase_gradients_x_max_error = max(max(max(abs(phase_gradients_x_cpu(:,2:end-1,:) - phase_gradients_x_opencl(:,2:end-1,:)))))
%
%
