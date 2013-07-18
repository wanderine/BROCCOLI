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

%mex MotionCorrection.cpp -lOpenCL -lBROCCOLI_LIB -IC:/Program' Files'/NVIDIA' GPU Computing Toolkit'/CUDA/v5.0/include -IC:/Program' Files'/NVIDIA' GPU Computing Toolkit'/CUDA/v5.0/include/CL -LC:/Program' Files'/NVIDIA' GPU Computing Toolkit'/CUDA/v5.0/lib/x64 -LC:/users/wande/Documents/Visual' Studio 2010'/Projects/BROCCOLI_LIB/x64/Release/ -IC:/users/wande/Documents/Visual' Studio 2010'/Projects/BROCCOLI_LIB/BROCCOLI_LIB -IC:\Users\wande\Documents\Visual' Studio 2010'\Projects\BROCCOLI_LIB\nifticlib-2.0.0\niftilib  -IC:\Users\wande\Documents\Visual' Studio 2010'\Projects\BROCCOLI_LIB\nifticlib-2.0.0\znzlib

mex -g MotionCorrection.cpp -lOpenCL -lBROCCOLI_LIB -IC:/Program' Files'/NVIDIA' GPU Computing Toolkit'/CUDA/v5.0/include -IC:/Program' Files'/NVIDIA' GPU Computing Toolkit'/CUDA/v5.0/include/CL -LC:/Program' Files'/NVIDIA' GPU Computing Toolkit'/CUDA/v5.0/lib/x64 -LC:/users/wande/Documents/Visual' Studio 2010'/Projects/BROCCOLI_LIB/x64/Debug/ -IC:/users/wande/Documents/Visual' Studio 2010'/Projects/BROCCOLI_LIB/BROCCOLI_LIB -IC:\Users\wande\Documents\Visual' Studio 2010'\Projects\BROCCOLI_LIB\nifticlib-2.0.0\niftilib  -IC:\Users\wande\Documents\Visual' Studio 2010'\Projects\BROCCOLI_LIB\nifticlib-2.0.0\znzlib

%filename = '../../test_data/msit_1.6mm_1.nii';

load ../../test_data/hand_movements_left.mat
fMRI_volumes = vol_exp;
reference_volume = fMRI_volumes(:,:,:,1);
[sy sx sz st] = size(fMRI_volumes)
number_of_iterations_for_motion_correction = 5;
load filters.mat

%%
% Create random translations, for testing

generated_fMRI_volumes = zeros(size(fMRI_volumes));
generated_fMRI_volumes(:,:,:,1) = fMRI_volumes(:,:,:,1);

for t = 2:st
    
    original_volume = fMRI_volumes(:,:,:,2);
    middle_x = (sx-1)/2;
    middle_y = (sy-1)/2;
    middle_z = (sz-1)/2;
    
    % Translation in 3 directions
    x_translation = 2*randn;
    y_translation = 2*randn;
    z_translation = 2*randn;
    
    x_translations(t) = x_translation;
    y_translations(t) = y_translation;
    z_translations(t) = z_translation;
    
    % Rotation in 3 directions
    x_rotation = randn; % degrees
    y_rotation = randn; % degrees
    z_rotation = randn; % degrees
    
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
    
    altered_volume = interp3(xi,yi,zi,original_volume,rx_r,ry_r,rz_r,'cubic');
    altered_volume(isnan(altered_volume)) = 0;
    
    % Then add translation
    
    [xi, yi, zi] = meshgrid(-(sx-1)/2:(sx-1)/2,-(sy-1)/2:(sy-1)/2, -(sz-1)/2:(sz-1)/2);
    
    rx_t = zeros(sy,sx,sz);
    ry_t = zeros(sy,sx,sz);
    rz_t = zeros(sy,sx,sz);
    
    rx_t(:) = x_translation;
    ry_t(:) = y_translation;
    rz_t(:) = z_translation;
    
    altered_volume = interp3(xi,yi,zi,altered_volume,xi - rx_t,yi - ry_t,zi - rz_t,'cubic');
    altered_volume(isnan(altered_volume)) = 0;
  
    generated_fMRI_volumes(:,:,:,t) = altered_volume;
end


%%

fMRI_volumes = generated_fMRI_volumes;

[motion_corrected_volumes_cpu,motion_parameters_cpu, quadrature_filter_response_reference_1_cpu, quadrature_filter_response_reference_2_cpu, quadrature_filter_response_reference_3_cpu] = perform_fMRI_registration_CPU(fMRI_volumes,f1,f2,f3,number_of_iterations_for_motion_correction);

tic
[motion_corrected_volumes_opencl,motion_parameters_opencl, quadrature_filter_response_1_opencl, quadrature_filter_response_2_opencl, quadrature_filter_response_3_opencl, phase_differences_x_opencl, phase_certainties_x_opencl, phase_gradients_x_opencl] = MotionCorrection(fMRI_volumes,f1,f2,f3,number_of_iterations_for_motion_correction);
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

figure
imagesc( [real(quadrature_filter_response_aligned_1_cpu(:,:,slice)) real(quadrature_filter_response_1_opencl(:,:,slice)) ] ); colorbar
figure
imagesc( [real(quadrature_filter_response_aligned_2_cpu(:,:,slice)) real(quadrature_filter_response_2_opencl(:,:,slice)) ] ); colorbar
figure
imagesc( [real(quadrature_filter_response_aligned_3_cpu(:,:,slice)) real(quadrature_filter_response_3_opencl(:,:,slice)) ] ); colorbar

figure
imagesc( [imag(quadrature_filter_response_aligned_1_cpu(:,:,slice)) imag(quadrature_filter_response_1_opencl(:,:,slice)) ] ); colorbar
figure
imagesc( [imag(quadrature_filter_response_aligned_2_cpu(:,:,slice)) imag(quadrature_filter_response_2_opencl(:,:,slice)) ] ); colorbar
figure
imagesc( [imag(quadrature_filter_response_aligned_3_cpu(:,:,slice)) imag(quadrature_filter_response_3_opencl(:,:,slice)) ] ); colorbar

figure
imagesc( [phase_differences_x_cpu(:,:,slice) phase_differences_x_opencl(:,:,slice) ] ); colorbar
figure
imagesc( [phase_certainties_x_cpu(:,:,slice) phase_certainties_x_opencl(:,:,slice) ] ); colorbar
%figure
%imagesc( [phase_certainties_x_cpu(:,:,slice) - phase_certainties_x_opencl(:,:,slice) ] ); colorbar
figure
imagesc( [phase_gradients_x_cpu(:,:,slice) phase_gradients_x_opencl(:,:,slice) ] ); colorbar
figure
imagesc( [phase_gradients_x_cpu(2:end-1,2:end-1,slice) - phase_gradients_x_opencl(2:end-1,2:end-1,slice) ] ); colorbar




filter_response_1_real_tot_error = sum(abs(real(quadrature_filter_response_aligned_1_cpu(:)) - real(quadrature_filter_response_1_opencl(:))))
filter_response_1_real_max_error = max(abs(real(quadrature_filter_response_aligned_1_cpu(:)) - real(quadrature_filter_response_1_opencl(:))))
filter_response_1_imag_tot_error = sum(abs(imag(quadrature_filter_response_aligned_1_cpu(:)) - imag(quadrature_filter_response_1_opencl(:))))
filter_response_1_imag_max_error = max(abs(imag(quadrature_filter_response_aligned_1_cpu(:)) - imag(quadrature_filter_response_1_opencl(:))))

filter_response_2_real_tot_error = sum(abs(real(quadrature_filter_response_aligned_2_cpu(:)) - real(quadrature_filter_response_2_opencl(:))))
filter_response_2_real_max_error = max(abs(real(quadrature_filter_response_aligned_2_cpu(:)) - real(quadrature_filter_response_2_opencl(:))))
filter_response_2_imag_tot_error = sum(abs(imag(quadrature_filter_response_aligned_2_cpu(:)) - imag(quadrature_filter_response_2_opencl(:))))
filter_response_2_imag_max_error = max(abs(imag(quadrature_filter_response_aligned_2_cpu(:)) - imag(quadrature_filter_response_2_opencl(:))))

filter_response_3_real_tot_error = sum(abs(real(quadrature_filter_response_aligned_3_cpu(:)) - real(quadrature_filter_response_3_opencl(:))))
filter_response_3_real_max_error = max(abs(real(quadrature_filter_response_aligned_3_cpu(:)) - real(quadrature_filter_response_3_opencl(:))))
filter_response_3_imag_tot_error = sum(abs(imag(quadrature_filter_response_aligned_3_cpu(:)) - imag(quadrature_filter_response_3_opencl(:))))
filter_response_3_imag_max_error = max(abs(imag(quadrature_filter_response_aligned_3_cpu(:)) - imag(quadrature_filter_response_3_opencl(:))))

phase_differences_x_tot_error = sum(abs(phase_differences_x_cpu(:) - phase_differences_x_opencl(:)))
phase_differences_x_max_error = max(abs(phase_differences_x_cpu(:) - phase_differences_x_opencl(:)))

phase_certainties_x_tot_error = sum(abs(phase_certainties_x_cpu(:) - phase_certainties_x_opencl(:)))
phase_certainties_x_max_error = max(abs(phase_certainties_x_cpu(:) - phase_certainties_x_opencl(:)))

phase_gradients_x_tot_error = sum(sum(sum(abs(phase_gradients_x_cpu(:,2:end-1,:) - phase_gradients_x_opencl(:,2:end-1,:)))))
phase_gradients_x_max_error = max(max(max(abs(phase_gradients_x_cpu(:,2:end-1,:) - phase_gradients_x_opencl(:,2:end-1,:)))))

motion_parameters_opencl(1,3) = 0;


figure
plot(z_translations,'g')
hold on
plot(motion_parameters_cpu(:,3),'r')
hold on
plot(motion_parameters_opencl(:,3),'b')
hold off
legend('Applied translations','Estimated translations CPU','Estimated translations OpenCL')

