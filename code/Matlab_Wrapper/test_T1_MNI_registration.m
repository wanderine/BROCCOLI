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


%mex RegisterT1MNI.cpp -lOpenCL -lBROCCOLI_LIB -IC:/Program' Files'/NVIDIA' GPU Computing Toolkit'/CUDA/v5.0/include -IC:/Program' Files'/NVIDIA' GPU Computing Toolkit'/CUDA/v5.0/include/CL -LC:/Program' Files'/NVIDIA' GPU Computing Toolkit'/CUDA/v5.0/lib/x64 -LC:/users/wande/Documents/Visual' Studio 2010'/Projects/BROCCOLI_LIB/x64/Release/ -IC:/users/wande/Documents/Visual' Studio 2010'/Projects/BROCCOLI_LIB/BROCCOLI_LIB -IC:\Users\wande\Documents\Visual' Studio 2010'\Projects\BROCCOLI_LIB\nifticlib-2.0.0\niftilib  -IC:\Users\wande\Documents\Visual' Studio 2010'\Projects\BROCCOLI_LIB\nifticlib-2.0.0\znzlib

mex -g RegisterT1MNI.cpp -lOpenCL -lBROCCOLI_LIB -IC:/Program' Files'/NVIDIA' GPU Computing Toolkit'/CUDA/v5.0/include -IC:/Program' Files'/NVIDIA' GPU Computing Toolkit'/CUDA/v5.0/include/CL -LC:/Program' Files'/NVIDIA' GPU Computing Toolkit'/CUDA/v5.0/lib/x64 -LC:/users/wande/Documents/Visual' Studio 2010'/Projects/BROCCOLI_LIB/x64/Debug/ -IC:/users/wande/Documents/Visual' Studio 2010'/Projects/BROCCOLI_LIB/BROCCOLI_LIB -IC:\Users\wande\Documents\Visual' Studio 2010'\Projects\BROCCOLI_LIB\nifticlib-2.0.0\niftilib  -IC:\Users\wande\Documents\Visual' Studio 2010'\Projects\BROCCOLI_LIB\nifticlib-2.0.0\znzlib

load filters.mat

%T1_nii = load_nii('../../test_data/T1_AE.nii');
T1_nii = load_nii('mprage_anonymized15.nii.gz');
T1 = double(T1_nii.img);
%T1 = T1(1:182,1:218,105:end);
MNI_nii = load_nii('../../test_data/MNI152_T1_1mm.nii');
MNI = double(MNI_nii.img);
%MNI = MNI(:,:,1:end-30);
[sy_T1 sx_T1 sz_T1] = size(T1)
[sy sx sz] = size(MNI)

% Make sure T1 has same voxel size as MNI
T1_voxel_size_x = T1_nii.hdr.dime.pixdim(1);
T1_voxel_size_y = T1_nii.hdr.dime.pixdim(2);
T1_voxel_size_z = T1_nii.hdr.dime.pixdim(3);

%T1_voxel_size_x = 1.5
%T1_voxel_size_y = 0.55
%T1_voxel_size_z = 0.8


MNI_voxel_size_x = 1.0;
MNI_voxel_size_y = 1.0;
MNI_voxel_size_z = 1.0;

if ( (abs(T1_voxel_size_x - MNI_voxel_size_x) > 0.05) || (abs(T1_voxel_size_y - MNI_voxel_size_y) > 0.05) || (abs(T1_voxel_size_z - MNI_voxel_size_z) > 0.05))
    % Calculate new number of elements
    sx_T1_new = round(sx_T1*T1_voxel_size_x / MNI_voxel_size_x);
    sy_T1_new = round(sy_T1*T1_voxel_size_y / MNI_voxel_size_y);
    sz_T1_new = round(sz_T1*T1_voxel_size_z / MNI_voxel_size_z);
    x = linspace(1,sx_T1,sx_T1_new);
    y = linspace(1,sy_T1,sy_T1_new);
    z = linspace(1,sz_T1,sz_T1_new);
    [xx,yy,zz] = meshgrid(x,y,z);
    temp = interp3(T1,xx,yy,zz,'linear');    % Generates NaN's
    temp(isnan(temp)) = 0;
    T1_new = temp;
    [sy_T1 sx_T1 sz_T1] = size(T1)
end

[registered_T1_opencl, registration_parameters_opencl, quadrature_filter_response_1_opencl, quadrature_filter_response_2_opencl, quadrature_filter_response_3_opencl, phase_differences_x_opencl, phase_certainties_x_opencl, phase_gradients_x_opencl] = RegisterT1MNI(T1,MNI,T1_voxel_size_x,T1_voxel_size_y,T1_voxel_size_z,MNI_voxel_size_x,MNI_voxel_size_y,MNI_voxel_size_z,f1,f2,f3,1);

slice = 60;
figure; imagesc(T1(:,:,slice))
figure; imagesc(T1_new(:,:,slice))
figure; imagesc(registered_T1_opencl(:,:,slice))
figure; imagesc(T1_new(:,:,slice) - registered_T1_opencl(:,:,slice)); colorbar

tot_error = sum(abs(T1_new(:) - registered_T1_opencl(:)))
max_error = max(abs(T1_new(:) - registered_T1_opencl(:)))
mean_error = mean(abs(T1_new(:) - registered_T1_opencl(:)))

% Make sure T1 has same size as MNI
% 
% x_diff = sx_T1 - sx;
% y_diff = sy_T1 - sy;
% z_diff = sz_T1 - sz;
% 
% if z_diff > 0
%     T1 = T1(:,:,z_diff+1:end);
%     % Shift down 10 slices
%     T1(:,:,1:end-10) = T1(:,:,11:end);
%     T1(:,:,end-9:end) = zeros(sy_T1,sx_T1,10);
% else
%     temp = zeros(sy_T1,sx_T1,sz);
%     z_diff = abs(z_diff);
%     temp(:,:,round(z_diff/2)+1:round(z_diff/2)+sz_T1) = T1;
%     T1 = temp;
%     % Shift down 10 slices
%     T1(:,:,1:end-10) = T1(:,:,11:end);
%     T1(:,:,end-9:end) = zeros(sy_T1,sx_T1,10);
% end
% 
% 
% if y_diff > 0
%     T1 = T1(y_diff/2+1:end-y_diff/2,:,:);
% else
%     temp = zeros(sy,sx_T1,sz);
%     y_diff = abs(y_diff);
%     temp(round(y_diff/2)+1:round(y_diff/2)+sy_T1,:,:) = T1;
%     T1 = temp;
% end
% 
% 
% if x_diff > 0
%     T1 = T1(:,x_diff/2+1:end-x_diff/2,:);
% else
%     temp = zeros(sy,sx,sz);
%     x_diff = abs(x_diff);
%     temp(:,round(x_diff/2)+1:round(x_diff/2)+sx_T1,:,:) = T1;
%     T1 = temp;
% end
% 
% [sy_T1 sx_T1 sz_T1] = size(T1)
% 
% 
% 
% %T1 = T1(y_diff/2+1:end-y_diff/2,x_diff/2+1:end-x_diff/2,z_diff+1:end);
% 
% 
% mask_nii = load_nii('../../test_data/MNI152_T1_1mm_brain_mask.nii');
% mask = double(mask_nii.img);
% 
% 
% number_of_iterations_for_motion_correction = 50;
% 
% 
% scales =     [ 8 4    2 1];
% %number_of_iterations_for_motion_correction = [50 50 30 15  3];
% scale = scales(1);
% T1_current = T1(1:scale:end,1:scale:end,1:scale:end);
% MNI_current = MNI(1:scale:end,1:scale:end,1:scale:end);
% [sy sx sz] = size(T1_current)
% 
% total_registration_parameters = zeros(12,1);
% tic
% i = 0;
% for scale = scales
%     i = i + 1;
%     [registered_T1_cpu, registration_parameters_cpu] = perform_T1_MNI_registration_CPU(T1_current,MNI_current,f1,f2,f3,number_of_iterations_for_motion_correction);
%     
%     slice = round(128/scale);
%     
%     figure
%     imagesc([ MNI_current(:,:,slice)/norm(MNI_current(:)) T1_current(:,:,slice)/norm(T1_current(:)) ] ); colormap gray
%     figure
%     imagesc([ MNI_current(:,:,slice)/norm(MNI_current(:)) registered_T1_cpu(:,:,slice)/norm(registered_T1_cpu(:)) ] ); colormap gray
%     drawnow
%     
%     if scale ~= 1
%         T1_current = T1(1:scale/2:end,1:scale/2:end,1:scale/2:end);
%         MNI_current = MNI(1:scale/2:end,1:scale/2:end,1:scale/2:end);
%         [sy sx sz] = size(T1_current)
%         p = registration_parameters_cpu;
%         total_registration_parameters(1:3) = total_registration_parameters(1:3)*scales(i)/scales(i+1) + p(1:3)*scales(i)/scales(i+1);
%         total_registration_parameters(4:12) = total_registration_parameters(4:12) + p(4:12);
%         total_registration_parameters
%         p = total_registration_parameters;
%         [x, y, z] = meshgrid(-(sx-1)/2:(sx-1)/2,-(sy-1)/2:(sy-1)/2, -(sz-1)/2:(sz-1)/2);
%         x_motion_vectors = zeros(sy,sx,sz);
%         y_motion_vectors = zeros(sy,sx,sz);
%         z_motion_vectors = zeros(sy,sx,sz);
%         x_motion_vectors(:) = p(1) + [x(:) y(:) z(:)]*p(4:6);
%         y_motion_vectors(:) = p(2) + [x(:) y(:) z(:)]*p(7:9);
%         z_motion_vectors(:) = p(3) + [x(:) y(:) z(:)]*p(10:12);
%         
%         temp = interp3(x,y,z,T1_current,x+x_motion_vectors,y+y_motion_vectors,z+z_motion_vectors,'linear');    % Generates NaN's
%         temp(isnan(temp)) = 0;
%         T1_current = temp;
%         
%     else
%         total_registration_parameters = total_registration_parameters + registration_parameters_cpu;
%         total_registration_parameters
%     end
%     
%     
% end
% toc
% 
% p = total_registration_parameters;
% [sy sx sz] = size(T1)
% [x, y, z] = meshgrid(-(sx-1)/2:(sx-1)/2,-(sy-1)/2:(sy-1)/2, -(sz-1)/2:(sz-1)/2);
% x_motion_vectors = zeros(sy,sx,sz);
% y_motion_vectors = zeros(sy,sx,sz);
% z_motion_vectors = zeros(sy,sx,sz);
% x_motion_vectors(:) = p(1) + [x(:) y(:) z(:)]*p(4:6);
% y_motion_vectors(:) = p(2) + [x(:) y(:) z(:)]*p(7:9);
% z_motion_vectors(:) = p(3) + [x(:) y(:) z(:)]*p(10:12);
% temp = interp3(x,y,z,T1,x+x_motion_vectors,y+y_motion_vectors,z+z_motion_vectors,'linear');    % Generates NaN's
% temp(isnan(temp)) = 0;
% T1_current = temp;
% 
% 
% % for slice = 1:size(MNI,3)
% %     figure(14)
% %     imagesc([ MNI(:,:,slice)/norm(MNI(:)) registered_T1_cpu(:,:,slice)/norm(registered_T1_cpu(:)) ] ); colormap gray
% %     drawnow
% %
% %     %pause(0.1)
% %     pause
% % end
% 
% skullstripped = registered_T1_cpu .* mask;
% 
% for slice = 1:size(skullstripped,3)
%     figure(11)
%     imagesc([ registered_T1_cpu(:,:,slice)] ); colormap gray
%     figure(12)
%     imagesc([ skullstripped(:,:,slice)] ); colormap gray
%     
%     pause(0.1)
%     %pause
% end
% 
% 
% for slice = 1:size(skullstripped,1)
%     figure(11)
%     imagesc([ squeeze(registered_T1_cpu(slice,:,:))] ); colormap gray
%     figure(12)
%     imagesc([ squeeze(skullstripped(slice,:,:))] ); colormap gray
%     
%     %pause(0.1)
%     pause
% end
% 
% %
% % quadrature_filter_response_reference_1_cpu = convn(fMRI_volumes(:,:,:,1),f1,'same');
% % quadrature_filter_response_reference_2_cpu = convn(fMRI_volumes(:,:,:,1),f2,'same');
% % quadrature_filter_response_reference_3_cpu = convn(fMRI_volumes(:,:,:,1),f3,'same');
% %
% % quadrature_filter_response_aligned_1_cpu = convn(fMRI_volumes(:,:,:,2),f1,'same');
% % quadrature_filter_response_aligned_2_cpu = convn(fMRI_volumes(:,:,:,2),f2,'same');
% % quadrature_filter_response_aligned_3_cpu = convn(fMRI_volumes(:,:,:,2),f3,'same');
% %
% % phase_differences_x_cpu = angle(quadrature_filter_response_reference_1_cpu .* conj(quadrature_filter_response_aligned_1_cpu));
% %
% % phase_certainties_x_cpu = abs(quadrature_filter_response_reference_1_cpu .* quadrature_filter_response_aligned_1_cpu) .* ((cos(phase_differences_x_cpu/2)).^2);
% %
% % phase_gradients_x_cpu = zeros(sy,sx,sz);
% % phase_gradients_x_cpu(:,2:end-1,:) = angle(quadrature_filter_response_reference_1_cpu(:,3:end,:).*conj(quadrature_filter_response_reference_1_cpu(:,2:end-1,:)) + quadrature_filter_response_reference_1_cpu(:,2:end-1,:).*conj(quadrature_filter_response_reference_1_cpu(:,1:end-2,:)) + ...
% %     quadrature_filter_response_aligned_1_cpu(:,3:end,:).*conj(quadrature_filter_response_aligned_1_cpu(:,2:end-1,:)) + quadrature_filter_response_aligned_1_cpu(:,2:end-1,:).*conj(quadrature_filter_response_aligned_1_cpu(:,1:end-2,:)));
% %
% % %
% %
% %
% % slice = 4;
% %
% % figure
% % imagesc( [real(quadrature_filter_response_reference_1_cpu(:,:,slice)) real(quadrature_filter_response_1_opencl(:,:,slice)) ] )
% % figure
% % imagesc( [real(quadrature_filter_response_reference_2_cpu(:,:,slice)) real(quadrature_filter_response_2_opencl(:,:,slice)) ] )
% % figure
% % imagesc( [real(quadrature_filter_response_reference_3_cpu(:,:,slice)) real(quadrature_filter_response_3_opencl(:,:,slice)) ] )
% %
% % figure
% % imagesc( [imag(quadrature_filter_response_reference_1_cpu(:,:,slice)) imag(quadrature_filter_response_1_opencl(:,:,slice)) ] )
% % figure
% % imagesc( [imag(quadrature_filter_response_reference_2_cpu(:,:,slice)) imag(quadrature_filter_response_2_opencl(:,:,slice)) ] )
% % figure
% % imagesc( [imag(quadrature_filter_response_reference_3_cpu(:,:,slice)) imag(quadrature_filter_response_3_opencl(:,:,slice)) ] )
% %
% % figure
% % imagesc( [phase_differences_x_cpu(:,:,slice) phase_differences_x_opencl(:,:,slice) ] )
% % figure
% % imagesc( [phase_certainties_x_cpu(:,:,slice) phase_certainties_x_opencl(:,:,slice) ] )
% % %figure
% % %imagesc( [phase_certainties_x_cpu(:,:,slice) - phase_certainties_x_opencl(:,:,slice) ] ); colorbar
% % figure
% % imagesc( [phase_gradients_x_cpu(:,:,slice) phase_gradients_x_opencl(:,:,slice) ] )
% % figure
% % imagesc( [phase_gradients_x_cpu(2:end-1,2:end-1,slice) - phase_gradients_x_opencl(2:end-1,2:end-1,slice) ] ); colorbar
% %
% %
% %
% %
% % filter_response_1_real_tot_error = sum(abs(real(quadrature_filter_response_aligned_1_cpu(:)) - real(quadrature_filter_response_1_opencl(:))))
% % filter_response_1_real_max_error = max(abs(real(quadrature_filter_response_aligned_1_cpu(:)) - real(quadrature_filter_response_1_opencl(:))))
% % filter_response_1_imag_tot_error = sum(abs(imag(quadrature_filter_response_aligned_1_cpu(:)) - imag(quadrature_filter_response_1_opencl(:))))
% % filter_response_1_imag_max_error = max(abs(imag(quadrature_filter_response_aligned_1_cpu(:)) - imag(quadrature_filter_response_1_opencl(:))))
% %
% % filter_response_2_real_tot_error = sum(abs(real(quadrature_filter_response_aligned_2_cpu(:)) - real(quadrature_filter_response_2_opencl(:))))
% % filter_response_2_real_max_error = max(abs(real(quadrature_filter_response_aligned_2_cpu(:)) - real(quadrature_filter_response_2_opencl(:))))
% % filter_response_2_imag_tot_error = sum(abs(imag(quadrature_filter_response_aligned_2_cpu(:)) - imag(quadrature_filter_response_2_opencl(:))))
% % filter_response_2_imag_max_error = max(abs(imag(quadrature_filter_response_aligned_2_cpu(:)) - imag(quadrature_filter_response_2_opencl(:))))
% %
% % filter_response_3_real_tot_error = sum(abs(real(quadrature_filter_response_aligned_3_cpu(:)) - real(quadrature_filter_response_3_opencl(:))))
% % filter_response_3_real_max_error = max(abs(real(quadrature_filter_response_aligned_3_cpu(:)) - real(quadrature_filter_response_3_opencl(:))))
% % filter_response_3_imag_tot_error = sum(abs(imag(quadrature_filter_response_aligned_3_cpu(:)) - imag(quadrature_filter_response_3_opencl(:))))
% % filter_response_3_imag_max_error = max(abs(imag(quadrature_filter_response_aligned_3_cpu(:)) - imag(quadrature_filter_response_3_opencl(:))))
% %
% % phase_differences_x_tot_error = sum(abs(phase_differences_x_cpu(:) - phase_differences_x_opencl(:)))
% % phase_differences_x_max_error = max(abs(phase_differences_x_cpu(:) - phase_differences_x_opencl(:)))
% %
% % phase_certainties_x_tot_error = sum(abs(phase_certainties_x_cpu(:) - phase_certainties_x_opencl(:)))
% % phase_certainties_x_max_error = max(abs(phase_certainties_x_cpu(:) - phase_certainties_x_opencl(:)))
% %
% % phase_gradients_x_tot_error = sum(sum(sum(abs(phase_gradients_x_cpu(:,2:end-1,:) - phase_gradients_x_opencl(:,2:end-1,:)))))
% % phase_gradients_x_max_error = max(max(max(abs(phase_gradients_x_cpu(:,2:end-1,:) - phase_gradients_x_opencl(:,2:end-1,:)))))
% %
% %
