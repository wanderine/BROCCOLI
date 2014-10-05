%  	 BROCCOLI: Software for Fast fMRI Analysis on Many-Core CPUs and GPUs
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

% Compile the Matlab wrapper to get a mex-file
if ispc
    addpath('D:\nifti_matlab')
    addpath('D:\BROCCOLI_test_data')
    basepath = 'D:\BROCCOLI_test_data\';
    %mex -g MotionCorrection.cpp -lOpenCL -lBROCCOLI_LIB -IC:/Program' Files'/NVIDIA' GPU Computing Toolkit'/CUDA/v5.0/include -IC:/Program' Files'/NVIDIA' GPU Computing Toolkit'/CUDA/v5.0/include/CL -LC:/Program' Files'/NVIDIA' GPU Computing Toolkit'/CUDA/v5.0/lib/x64 -LC:/users/wande/Documents/Visual' Studio 2010'/Projects/BROCCOLI_LIB/x64/Debug/ -IC:/users/wande/Documents/Visual' Studio 2010'/Projects/BROCCOLI_LIB/BROCCOLI_LIB -IC:\Users\wande\Documents\Visual' Studio 2010'\Projects\BROCCOLI_LIB\nifticlib-2.0.0\niftilib  -IC:\Users\wande\Documents\Visual' Studio 2010'\Projects\BROCCOLI_LIB\nifticlib-2.0.0\znzlib -IC:\Users\wande\Documents\Visual' Studio 2010'\Projects\BROCCOLI_LIB\Eigen
    mex MotionCorrection.cpp -lOpenCL -lBROCCOLI_LIB -IC:/Program' Files'/NVIDIA' GPU Computing Toolkit'/CUDA/v5.0/include -IC:/Program' Files'/NVIDIA' GPU Computing Toolkit'/CUDA/v5.0/include/CL -LC:/Program' Files'/NVIDIA' GPU Computing Toolkit'/CUDA/v5.0/lib/x64 -LC:/users/wande/Documents/Visual' Studio 2010'/Projects/BROCCOLI_LIB/x64/Release/ -IC:/users/wande/Documents/Visual' Studio 2010'/Projects/BROCCOLI_LIB/BROCCOLI_LIB -IC:\Users\wande\Documents\Visual' Studio 2010'\Projects\BROCCOLI_LIB\nifticlib-2.0.0\niftilib  -IC:\Users\wande\Documents\Visual' Studio 2010'\Projects\BROCCOLI_LIB\nifticlib-2.0.0\znzlib -IC:\Users\wande\Documents\Visual' Studio 2010'\Projects\BROCCOLI_LIB\Eigen
    
    opencl_platform = 0;
    opencl_device = 0;
elseif isunix
    addpath('/home/andek/Research_projects/nifti_matlab')
    basepath = '/data/andek/BROCCOLI_test_data/';
    %mex -g MotionCorrection.cpp -lOpenCL -lBROCCOLI_LIB -I/usr/local/cuda-5.0/include/ -I/usr/local/cuda-5.0/include/CL -L/usr/lib -I/home/andek/Research_projects/BROCCOLI/BROCCOLI/code/BROCCOLI_LIB/ -L/home/andek/cuda-workspace/BROCCOLI_LIB/Debug
    mex MotionCorrection.cpp -lOpenCL -lBROCCOLI_LIB -I/usr/local/cuda-5.0/include/ -I/usr/local/cuda-5.0/include/CL -L/usr/lib -I/home/andek/Research_projects/BROCCOLI/BROCCOLI/code/BROCCOLI_LIB/ -L/home/andek/cuda-workspace/BROCCOLI_LIB/Release -I/home/andek/Research_projects/BROCCOLI/BROCCOLI/code/BROCCOLI_LIB/Eigen/
    
    opencl_platform = 2; % 0 Intel, 1 AMD, 2 Nvidia
    opencl_device = 0;
end

%study = 'Oulu';
%study = 'ICBM';    
study = 'Cambridge';
%study = 'Beijing';
%study = 'OpenfMRI';
%substudy = 'Mixed';

noise_level = 0.0; % 0.01, 0.02         % Amount of Gaussian noise, the standard deviation is set to noise_level * max_intensity_value
save_test_dataset = 0;                  % Save testing data as a nifti file or not
save_motion_corrected_data_Matlab = 0;  % Save motion corrected data as a Matlab file or not
save_motion_corrected_data_Nifti = 0;   % Save motion corrected data as a Nifti file or not
plot_results = 1;                       % Plot true and estimated motion parameters or not
save_estimated_motion_parameters = 0;   % Save estimated motion parameters as a Matlab file or not
save_true_motion_parameters = 0;        % Save true motion parameters as a Matlab file or not
add_shading = 0;                        % Add shading to each fMRI volume or not
run_Matlab_equivalent = 0;              % Run Matlab equivalent or not, for comparison to OpenCL algorithm

motion_correction_times = zeros(198,1);

% Loop over subjects
for s = 1:1
    
    s
    
    number_of_iterations_for_motion_correction = 5;
    
    % Load fMRI data
    if ispc
        if ( (strcmp(study,'Beijing')) || (strcmp(study,'Cambridge')) || (strcmp(study,'ICBM')) || (strcmp(study,'Oulu')) )
            EPI_nii = load_nii([basepath study '/rest' num2str(s) '.nii.gz']);
        elseif ( strcmp(study,'OpenfMRI'))
            EPI_nii = load_nii([basepath study '\' substudy '/bold' num2str(s) '.nii.gz']);
        end
    elseif isunix
        dirs = dir([basepath study]);
        subject = dirs(s+2).name % Skip . and .. 'folders'
        EPI_nii = load_nii([basepath study '/' subject '/func/rest.nii.gz']);               
    end
    
    voxel_size_x = EPI_nii.hdr.dime.pixdim(2);
    voxel_size_y = EPI_nii.hdr.dime.pixdim(3);
    voxel_size_z = EPI_nii.hdr.dime.pixdim(4);

    fMRI_volumes = double(EPI_nii.img);        
    [sy sx sz st] = size(fMRI_volumes)
    
    % Load quadrature filters
    load filters_for_parametric_registration.mat
            
    %%
    % Create random transformations, for testing
    
    generated_fMRI_volumes = zeros(size(fMRI_volumes));
    generated_fMRI_volumes(:,:,:,1) = fMRI_volumes(:,:,:,1);
    reference_volume = fMRI_volumes(:,:,:,1);
    
    x_translations = zeros(st,1);
    y_translations = zeros(st,1);
    z_translations = zeros(st,1);
    
    x_rotations = zeros(st,1);
    y_rotations = zeros(st,1);
    z_rotations = zeros(st,1);
    
    factor = 0.1; % standard deviation for random translations and rotations    
    
    [xi, yi, zi] = meshgrid(-(sx-1)/2:(sx-1)/2,-(sy-1)/2:(sy-1)/2, -(sz-1)/2:(sz-1)/2);
            
    % Loop over timepoints
    for t = 2:st
        
        middle_x = (sx-1)/2;
        middle_y = (sy-1)/2;
        middle_z = (sz-1)/2;
        
        % Translation in 3 directions
        x_translation = factor*randn; % voxels
        y_translation = factor*randn; % voxels
        z_translation = factor*randn; % voxels
        
        %x_translation = x_translations(t);
        %y_translation = y_translations(t);
        %z_translation = z_translations(t);
        
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
        
        %x_rotation = x_rotations(t);
        %y_rotation = y_rotations(t);
        %z_rotation = z_rotations(t);
        
        % Create rotation matrices around the three axes
        
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
        
        % Add rotation and translation at the same time        
        altered_volume = interp3(xi,yi,zi,reference_volume,rx_r-rx_t,ry_r-ry_t,rz_r-rz_t,'cubic');        
        % Remove 'not are numbers' from interpolation
        altered_volume(isnan(altered_volume)) = 0;        
        
        if add_shading == 1
            % Shading in x direction, 25% of maximum intensity
            shade = repmat(linspace(1,0.25*max(fMRI_volumes(:)),sx),sy,1);
            for z = 1:sz
                altered_volume(:,:,z) =  altered_volume(:,:,z) + shade;
            end
        end
        
        % Add noise
        altered_volume = altered_volume + max(fMRI_volumes(:)) * noise_level * randn(size(altered_volume));
                                       
        generated_fMRI_volumes(:,:,:,t) = altered_volume;
        
    end
    
    % Save testing dataset as nifti file
    if save_test_dataset == 1
        new_file.hdr = EPI_nii.hdr;
        new_file.hdr.dime.datatype = 16;
        new_file.hdr.dime.bitpix = 16;
        new_file.img = single(generated_fMRI_volumes);    
        
        if add_shading == 1
           
            filename = ['/data/andek/BROCCOLI_test_data/Cambridge/with_random_motion/cambridge_rest_subject_' num2str(s) '_with_random_motion_shading.nii'];
            
        else
        
            if noise_level == 0
                filename = ['/data/andek/BROCCOLI_test_data/Cambridge/with_random_motion/cambridge_rest_subject_' num2str(s) '_with_random_motion_no_noise.nii'];                
            elseif noise_level == 0.01
                filename = ['/data/andek/BROCCOLI_test_data/Cambridge/with_random_motion/cambridge_rest_subject_' num2str(s) '_with_random_motion_1percent_noise.nii'];
            elseif noise_level == 0.02
                filename = ['/data/andek/BROCCOLI_test_data/Cambridge/with_random_motion/cambridge_rest_subject_' num2str(s) '_with_random_motion_2percent_noise.nii'];
            end
            
        end
            
        save_nii(new_file,filename);
    end
    
    %%
    
    fMRI_volumes = generated_fMRI_volumes;
    
    % Do motion correction with Matlab implementation, as comparison
    if run_Matlab_equivalent == 1
        [motion_corrected_volumes_cpu,motion_parameters_cpu, rotations_cpu, scalings_cpu, quadrature_filter_response_reference_1_cpu, quadrature_filter_response_reference_2_cpu, quadrature_filter_response_reference_3_cpu] = perform_fMRI_registration_CPU(fMRI_volumes,f1_parametric_registration,f2_parametric_registration,f3_parametric_registration,number_of_iterations_for_motion_correction);
    else
        motion_parameters_cpu = zeros(st,12);
        rotations_cpu = zeros(st,3);
    end
    
    % Do motion correction on OpenCL device
    start = clock;
    [motion_corrected_volumes_opencl,motion_parameters_opencl, quadrature_filter_response_1_opencl, quadrature_filter_response_2_opencl, quadrature_filter_response_3_opencl, ...
        phase_differences_x_opencl, phase_certainties_x_opencl, phase_gradients_x_opencl] = ...
        MotionCorrection(fMRI_volumes,f1_parametric_registration,f2_parametric_registration,f3_parametric_registration,number_of_iterations_for_motion_correction,opencl_platform,opencl_device);
    motion_correction_times(s) = etime(clock,start);
        
    % Save motion corrected data as Matlab file
    if save_motion_corrected_data_Matlab == 1        
        
        if add_shading == 1
        
            filename = ['/data/andek/BROCCOLI_test_data/BROCCOLI/motion_correction/BROCCOLI_motion_corrected_rest_subject_' num2str(s) '_random_motion_shading.mat'];
            
        else
        
            if noise_level == 0
                filename = ['/data/andek/BROCCOLI_test_data/BROCCOLI/motion_correction/BROCCOLI_motion_corrected_rest_subject_' num2str(s) '_random_motion_no_noise.mat'];
            elseif noise_level == 0.01
                filename = ['/data/andek/BROCCOLI_test_data/BROCCOLI/motion_correction/BROCCOLI_motion_corrected_rest_subject_' num2str(s) '_random_motion_1percent_noise.mat'];
            elseif noise_level == 0.02
                filename = ['/data/andek/BROCCOLI_test_data/BROCCOLI/motion_correction/BROCCOLI_motion_corrected_rest_subject_' num2str(s) '_random_motion_2percent_noise.mat'];
            end
        
        end
        
        motion_corrected_volumes_opencl = single(motion_corrected_volumes_opencl);
        save(filename,'motion_corrected_volumes_opencl');
    end
    
    % Save motion corrected data as nifti file
    if save_motion_corrected_data_Nifti == 1
        new_file.hdr = EPI_nii.hdr;
        new_file.hdr.dime.datatype = 16;
        new_file.hdr.dime.bitpix = 16;
        new_file.img = single(motion_corrected_volumes_opencl);    
        
        if add_shading == 1
           
            filename = ['/data/andek/BROCCOLI_test_data/BROCCOLI/motion_correction/BROCCOLI_motion_corrected_rest_subject_' num2str(s) '_random_motion_shading.nii'];
            
        else
        
            if noise_level == 0
                filename = ['/data/andek/BROCCOLI_test_data/BROCCOLI/motion_correction/BROCCOLI_motion_corrected_rest_subject_' num2str(s) '_random_motion_no_noise.nii'];                
            elseif noise_level == 0.01
                filename = ['/data/andek/BROCCOLI_test_data/BROCCOLI/motion_correction/BROCCOLI_motion_corrected_rest_subject_' num2str(s) '_random_motion_1percent_noise.nii'];
            elseif noise_level == 0.02
                filename = ['/data/andek/BROCCOLI_test_data/BROCCOLI/motion_correction/BROCCOLI_motion_corrected_rest_subject_' num2str(s) '_random_motion_2percent_noise.nii'];
            end
            
        end
            
        save_nii(new_file,filename);
    end
    
    % Save estimated motion parameters as Matlab file
    if save_estimated_motion_parameters == 1    
        
        if add_shading == 1
        
            filename = ['/data/andek/BROCCOLI_test_data/BROCCOLI/motion_correction/BROCCOLI_motion_parameters_subject_' num2str(s) '_random_motion_shading.mat'];
            
        else
            
            if noise_level == 0
                filename = ['/data/andek/BROCCOLI_test_data/BROCCOLI/motion_correction/BROCCOLI_motion_parameters_subject_' num2str(s) '_random_motion_no_noise.mat'];
            elseif noise_level == 0.01
                filename = ['/data/andek/BROCCOLI_test_data/BROCCOLI/motion_correction/BROCCOLI_motion_parameters_subject_' num2str(s) '_random_motion_1percent_noise.mat'];
            elseif noise_level == 0.02
                filename = ['/data/andek/BROCCOLI_test_data/BROCCOLI/motion_correction/BROCCOLI_motion_parameters_subject_' num2str(s) '_random_motion_2percent_noise.mat'];
            end
            
        end
        
        save(filename,'motion_parameters_opencl');
    end
    
    % Save true motion parameters as Matlab file
    if save_true_motion_parameters == 1        
        
        if add_shading == 1
            
            filename = ['/data/andek/BROCCOLI_test_data/Cambridge/with_random_motion/true_motion_parameters_subject_' num2str(s) '_random_motion_shading.mat'];
            
        else
            
            if noise_level == 0
                filename = ['/data/andek/BROCCOLI_test_data/Cambridge/with_random_motion/true_motion_parameters_subject_' num2str(s) '_random_motion_no_noise.mat'];
            elseif noise_level == 0.01
                filename = ['/data/andek/BROCCOLI_test_data/Cambridge/with_random_motion/true_motion_parameters_subject_' num2str(s) '_random_motion_1percent_noise.mat'];
            elseif noise_level == 0.02
                filename = ['/data/andek/BROCCOLI_test_data/Cambridge/with_random_motion/true_motion_parameters_subject_' num2str(s) '_random_motion_2percent_noise.mat'];
            end
            
        end
        
        save(filename,'x_translations','y_translations','z_translations','x_rotations','y_rotations','z_rotations');
    end        
    
    if plot_results == 1
        
        % Plot true and estimated translations in the x-direction
        figure
        plot(x_translations*voxel_size_x,'g')
        hold on
        plot(motion_parameters_cpu(:,1),'r')
        hold on
        plot(motion_parameters_opencl(:,1),'b')
        hold off
        legend('Applied x translations','Estimated x translations CPU','Estimated x translations OpenCL')
    
        % Plot true and estimated translations in the y-direction
        figure
        plot(y_translations*voxel_size_y,'g')
        hold on
        plot(motion_parameters_cpu(:,2),'r')
        hold on
        plot(motion_parameters_opencl(:,2),'b')
        hold off
        legend('Applied y translations','Estimated y translations CPU','Estimated y translations OpenCL')
    
        % Plot true and estimated translations in the z-direction
        figure
        plot(z_translations*voxel_size_z,'g')
        hold on
        plot(motion_parameters_cpu(:,3),'r')
        hold on
        plot(motion_parameters_opencl(:,3),'b')
        hold off
        legend('Applied z translations','Estimated z translations CPU','Estimated z translations OpenCL')
        
        % Plot true and estimated rotations in the x-direction
        figure
        plot(x_rotations,'g')
        hold on
        plot(rotations_cpu(:,1),'r')
        hold on
        plot(motion_parameters_opencl(:,4),'b')
        hold off
        legend('Applied x rotations','Estimated x rotations CPU','Estimated x rotations OpenCL')
      
        % Plot true and estimated rotations in the y-direction
        figure
        plot(y_rotations,'g')
        hold on
        plot(rotations_cpu(:,2),'r')
        hold on
        plot(motion_parameters_opencl(:,5),'b')
        hold off
        legend('Applied y rotations','Estimated y rotations CPU','Estimated y rotations OpenCL')
        
        % Plot true and estimated rotations in the z-direction
        figure
        plot(z_rotations,'g')
        hold on
        plot(rotations_cpu(:,3),'r')
        hold on
        plot(motion_parameters_opencl(:,6),'b')
        hold off
        legend('Applied z rotations','Estimated z rotations CPU','Estimated z rotations OpenCL')
    
    end   
    
    % Calculate differences between true parameters and estimated
    % parameters
    x_translations_opencl = squeeze(motion_parameters_opencl(:,1));
    y_translations_opencl = squeeze(motion_parameters_opencl(:,2));
    z_translations_opencl = squeeze(motion_parameters_opencl(:,3));
    
    x_rotations_opencl = squeeze(motion_parameters_opencl(:,4));
    y_rotations_opencl = squeeze(motion_parameters_opencl(:,5));
    z_rotations_opencl = squeeze(motion_parameters_opencl(:,6));
    
    mean_translation_error_x = mean(abs(x_translations(:)*voxel_size_x - x_translations_opencl(:)))
    mean_translation_error_y = mean(abs(y_translations(:)*voxel_size_y - y_translations_opencl(:)))
    mean_translation_error_z = mean(abs(z_translations(:)*voxel_size_z - z_translations_opencl(:)))
      
    mean_rotation_error_x = mean(abs(x_rotations(:) - x_rotations_opencl(:)))
    mean_rotation_error_y = mean(abs(y_rotations(:) - y_rotations_opencl(:)))
    mean_rotation_error_z = mean(abs(z_rotations(:) - z_rotations_opencl(:)))
    
    %pause
    %close all
    
end


%fMRI_volumes = double(EPI_nii.img);        
%[sy sx sz st] = size(fMRI_volumes)
    
% for t = 2:st
%     
%     figure(1)    
%     imagesc([motion_corrected_volumes_opencl(:,:,30,t)  generated_fMRI_volumes(:,:,30,t-1) ]/25); colormap gray; colorbar
%     %image([motion_corrected_volumes_opencl(:,:,30,t) - motion_corrected_volumes_opencl(:,:,30,t-1) ]/10); colormap gray; colorbar
%     %image([generated_fMRI_volumes(:,:,30,t) - generated_fMRI_volumes(:,:,30,t-1) ]); colormap gray; colorbar
%     %image([fMRI_volumes(:,:,30,t) - fMRI_volumes(:,:,30,t-1) ]); colormap gray; colorbar
%     pause(0.15)
%     
% end

filename='motion_parameters_CPU';
motion_parameters_CPU=motion_parameters_opencl;
save(filename,'motion_parameters_CPU');


