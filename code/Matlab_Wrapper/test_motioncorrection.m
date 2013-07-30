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
addpath('D:\BROCCOLI_test_data')

%mex MotionCorrection.cpp -lOpenCL -lBROCCOLI_LIB -IC:/Program' Files'/NVIDIA' GPU Computing Toolkit'/CUDA/v5.0/include -IC:/Program' Files'/NVIDIA' GPU Computing Toolkit'/CUDA/v5.0/include/CL -LC:/Program' Files'/NVIDIA' GPU Computing Toolkit'/CUDA/v5.0/lib/x64 -LC:/users/wande/Documents/Visual' Studio 2010'/Projects/BROCCOLI_LIB/x64/Release/ -IC:/users/wande/Documents/Visual' Studio 2010'/Projects/BROCCOLI_LIB/BROCCOLI_LIB -IC:\Users\wande\Documents\Visual' Studio 2010'\Projects\BROCCOLI_LIB\nifticlib-2.0.0\niftilib  -IC:\Users\wande\Documents\Visual' Studio 2010'\Projects\BROCCOLI_LIB\nifticlib-2.0.0\znzlib

mex -g MotionCorrection.cpp -lOpenCL -lBROCCOLI_LIB -IC:/Program' Files'/NVIDIA' GPU Computing Toolkit'/CUDA/v5.0/include -IC:/Program' Files'/NVIDIA' GPU Computing Toolkit'/CUDA/v5.0/include/CL -LC:/Program' Files'/NVIDIA' GPU Computing Toolkit'/CUDA/v5.0/lib/x64 -LC:/users/wande/Documents/Visual' Studio 2010'/Projects/BROCCOLI_LIB/x64/Debug/ -IC:/users/wande/Documents/Visual' Studio 2010'/Projects/BROCCOLI_LIB/BROCCOLI_LIB -IC:\Users\wande\Documents\Visual' Studio 2010'\Projects\BROCCOLI_LIB\nifticlib-2.0.0\niftilib  -IC:\Users\wande\Documents\Visual' Studio 2010'\Projects\BROCCOLI_LIB\nifticlib-2.0.0\znzlib

%filename = '../../test_data/msit_1.6mm_1.nii';

%load ../../test_data/hand_movements_right.mat
%fMRI_volumes = vol_exp;

subject = 21;
EPI_nii = load_nii(['rest' num2str(subject) '.nii.gz']);
fMRI_volumes = double(EPI_nii.img);
fMRI_volumes = fMRI_volumes/max(fMRI_volumes(:));
%fMRI_volumes = fMRI_volumes(:,:,1:22,:);

reference_volume = fMRI_volumes(:,:,:,1);
[sy sx sz st] = size(fMRI_volumes)
number_of_iterations_for_motion_correction = 5;
load filters.mat
opencl_platform = 0;


%%
% Create random translations, for testing

generated_fMRI_volumes = zeros(size(fMRI_volumes));
generated_fMRI_volumes(:,:,:,1) = fMRI_volumes(:,:,:,1);

x_translations = zeros(st,1);
y_translations = zeros(st,1);
z_translations = zeros(st,1);

x_rotations = zeros(st,1);
y_rotations = zeros(st,1);
z_rotations = zeros(st,1);

factor = 0.15;

for t = 2:st
        
    middle_x = (sx-1)/2;
    middle_y = (sy-1)/2;
    middle_z = (sz-1)/2;
    
    % Translation in 3 directions
    x_translation = factor*randn;
    y_translation = factor*randn;
    z_translation = factor*randn;
    
    x_translations(t) = x_translation;
    y_translations(t) = y_translation;
    z_translations(t) = z_translation;
    
    % Rotation in 3 directions
    x_rotation = factor*randn; % degrees
    y_rotation = factor*randn; % degrees
    z_rotation = factor*randn; % degrees
    
    x_rotations(t) = x_rotation;
    y_rotations(t) = y_rotation;
    z_rotations(t) = z_rotation;
    
    R_x = [1                        0                           0;
        0                        cos(x_rotation*pi/180)      -sin(x_rotation*pi/180);
        0                        sin(x_rotation*pi/180)      cos(x_rotation*pi/180)];
    
    R_y = [cos(y_rotation*pi/180)   0                           sin(y_rotation*pi/180);
        0                        1                           0;
        -sin(y_rotation*pi/180)  0                           cos(y_rotation*pi/180)];
    
    R_z = [cos(z_rotation*pi/180)   -sin(z_rotation*pi/180)     0;
        sin(z_rotation*pi/180)   cos(z_rotation*pi/180)      0;
        0                        0                           1];
    
    Rotation_matrix = R_x * R_y * R_z;
    Rotation_matrix = Rotation_matrix(:);
    
    % Add rotation first
    
    [xi, yi, zi] = meshgrid(-(sx-1)/2:(sx-1)/2,-(sy-1)/2:(sy-1)/2, -(sz-1)/2:(sz-1)/2);
    
    rx_r = zeros(sy,sx,sz);
    ry_r = zeros(sy,sx,sz);
    rz_r = zeros(sy,sx,sz);
    
    rx_r(:) = [xi(:) yi(:) zi(:)]*Rotation_matrix(1:3);
    ry_r(:) = [xi(:) yi(:) zi(:)]*Rotation_matrix(4:6);
    rz_r(:) = [xi(:) yi(:) zi(:)]*Rotation_matrix(7:9);

    rx_t = zeros(sy,sx,sz);
    ry_t = zeros(sy,sx,sz);
    rz_t = zeros(sy,sx,sz);
    
    rx_t(:) = x_translation;
    ry_t(:) = y_translation;
    rz_t(:) = z_translation;
    
    %altered_volume = interp3(xi,yi,zi,original_volume,rx_r,ry_r,rz_r,'cubic');
    altered_volume = interp3(xi,yi,zi,reference_volume,rx_r-rx_t,ry_r-ry_t,rz_r-rz_t,'cubic');
    altered_volume(isnan(altered_volume)) = 0;
    altered_volume = altered_volume + 0.05*randn(size(altered_volume));
    
    % Then add translation
    
    %[xi, yi, zi] = meshgrid(-(sx-1)/2:(sx-1)/2,-(sy-1)/2:(sy-1)/2, -(sz-1)/2:(sz-1)/2);
        
    %altered_volume = interp3(xi,yi,zi,altered_volume,xi - rx_t,yi - ry_t,zi - rz_t,'cubic');
    %altered_volume(isnan(altered_volume)) = 0;
  
    generated_fMRI_volumes(:,:,:,t) = altered_volume;
end


%%

fMRI_volumes = generated_fMRI_volumes;

[motion_corrected_volumes_cpu,motion_parameters_cpu, rotations_cpu, scalings_cpu, quadrature_filter_response_reference_1_cpu, quadrature_filter_response_reference_2_cpu, quadrature_filter_response_reference_3_cpu] = perform_fMRI_registration_CPU(fMRI_volumes,f1,f2,f3,number_of_iterations_for_motion_correction);

tic
[motion_corrected_volumes_opencl,motion_parameters_opencl, quadrature_filter_response_1_opencl, quadrature_filter_response_2_opencl, quadrature_filter_response_3_opencl, phase_differences_x_opencl, phase_certainties_x_opencl, phase_gradients_x_opencl] = MotionCorrection(fMRI_volumes,f1,f2,f3,number_of_iterations_for_motion_correction,opencl_platform);
toc

quadrature_filter_response_reference_1_cpu = convn(fMRI_volumes(:,:,:,1),f1,'same');
quadrature_filter_response_reference_2_cpu = convn(fMRI_volumes(:,:,:,1),f2,'same');
quadrature_filter_response_reference_3_cpu = convn(fMRI_volumes(:,:,:,1),f3,'same');

quadrature_filter_response_aligned_1_cpu = convn(fMRI_volumes(:,:,:,end),f1,'same');
quadrature_filter_response_aligned_2_cpu = convn(fMRI_volumes(:,:,:,end),f2,'same');
quadrature_filter_response_aligned_3_cpu = convn(fMRI_volumes(:,:,:,end),f3,'same');

phase_differences_x_cpu = angle(quadrature_filter_response_reference_1_cpu .* conj(quadrature_filter_response_aligned_1_cpu));

phase_certainties_x_cpu = abs(quadrature_filter_response_reference_1_cpu .* quadrature_filter_response_aligned_1_cpu) .* ((cos(phase_differences_x_cpu/2)).^2);

phase_gradients_x_cpu = zeros(sy,sx,sz);
phase_gradients_x_cpu(:,2:end-1,:) = angle(quadrature_filter_response_reference_1_cpu(:,3:end,:).*conj(quadrature_filter_response_reference_1_cpu(:,2:end-1,:)) + quadrature_filter_response_reference_1_cpu(:,2:end-1,:).*conj(quadrature_filter_response_reference_1_cpu(:,1:end-2,:)) + ...
    quadrature_filter_response_aligned_1_cpu(:,3:end,:).*conj(quadrature_filter_response_aligned_1_cpu(:,2:end-1,:)) + quadrature_filter_response_aligned_1_cpu(:,2:end-1,:).*conj(quadrature_filter_response_aligned_1_cpu(:,1:end-2,:)));

%


slice = 4;
% 
% figure
% imagesc( [real(quadrature_filter_response_aligned_1_cpu(:,:,slice)) real(quadrature_filter_response_1_opencl(:,:,slice)) ] ); colorbar
% figure
% imagesc( [real(quadrature_filter_response_aligned_2_cpu(:,:,slice)) real(quadrature_filter_response_2_opencl(:,:,slice)) ] ); colorbar
% figure
% imagesc( [real(quadrature_filter_response_aligned_3_cpu(:,:,slice)) real(quadrature_filter_response_3_opencl(:,:,slice)) ] ); colorbar
% 
% figure
% imagesc( [imag(quadrature_filter_response_aligned_1_cpu(:,:,slice)) imag(quadrature_filter_response_1_opencl(:,:,slice)) ] ); colorbar
% figure
% imagesc( [imag(quadrature_filter_response_aligned_2_cpu(:,:,slice)) imag(quadrature_filter_response_2_opencl(:,:,slice)) ] ); colorbar
% figure
% imagesc( [imag(quadrature_filter_response_aligned_3_cpu(:,:,slice)) imag(quadrature_filter_response_3_opencl(:,:,slice)) ] ); colorbar
% 
% figure
% imagesc( [phase_differences_x_cpu(:,:,slice) phase_differences_x_opencl(:,:,slice) ] ); colorbar
% figure
% imagesc( [phase_certainties_x_cpu(:,:,slice) phase_certainties_x_opencl(:,:,slice) ] ); colorbar
% %figure
% %imagesc( [phase_certainties_x_cpu(:,:,slice) - phase_certainties_x_opencl(:,:,slice) ] ); colorbar
% figure
% imagesc( [phase_gradients_x_cpu(:,:,slice) phase_gradients_x_opencl(:,:,slice) ] ); colorbar
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

motion_parameters_opencl(1,3) = 0;

figure
plot(x_translations,'g')
hold on
plot(motion_parameters_cpu(:,1),'r')
hold on
plot(motion_parameters_opencl(:,1),'b')
hold off
legend('Applied x translations','Estimated x translations CPU','Estimated x translations OpenCL')

figure
plot(y_translations,'g')
hold on
plot(motion_parameters_cpu(:,2),'r')
hold on
plot(motion_parameters_opencl(:,2),'b')
hold off
legend('Applied y translations','Estimated y translations CPU','Estimated y translations OpenCL')

figure
plot(z_translations,'g')
hold on
plot(motion_parameters_cpu(:,3),'r')
hold on
plot(motion_parameters_opencl(:,3),'b')
hold off
legend('Applied z translations','Estimated z translations CPU','Estimated z translations OpenCL')


figure
plot(x_rotations,'g')
hold on
plot(rotations_cpu(:,1),'r')
hold on
plot(motion_parameters_opencl(:,4),'b')
hold off
legend('Applied x rotations','Estimated x rotations CPU','Estimated x rotations OpenCL')


figure
plot(y_rotations,'g')
hold on
plot(rotations_cpu(:,2),'r')
hold on
plot(motion_parameters_opencl(:,5),'b')
hold off
legend('Applied y rotations','Estimated y rotations CPU','Estimated y rotations OpenCL')


figure
plot(z_rotations,'g')
hold on
plot(rotations_cpu(:,3),'r')
hold on
plot(motion_parameters_opencl(:,6),'b')
hold off
legend('Applied z rotations','Estimated z rotations CPU','Estimated z rotations OpenCL')

slice = 18;

figure
imagesc([motion_corrected_volumes_cpu(:,:,slice,2) - motion_corrected_volumes_opencl(:,:,slice,2)]); colorbar
title('MC cpu - gpu')

% for t = 1:st    
%    figure(5)
%    imagesc([fMRI_volumes(:,:,18,t) motion_corrected_volumes_cpu(:,:,18,t)  motion_corrected_volumes_opencl(:,:,18,t) ])
%    pause(0.1)
% end


