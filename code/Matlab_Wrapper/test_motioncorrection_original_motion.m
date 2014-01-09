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

ismiha = 0

% Compile the Matlab wrapper to get a mex-file
if ismiha
    addpath('/home/miha/Delo/BROCCOLI/nifti')
    basepath = '/home/miha/Programiranje/BROCCOLI/test_data/fcon1000/classic/';
    basepath_BROCCOLI = '/data/andek/BROCCOLI_test_data/BROCCOLI/normalization';
    
    mex MotionCorrection.cpp -lOpenCL -lBROCCOLI_LIB -I/opt/cuda/include/ -I/opt/cuda/include/CL -L/usr/lib -I/home/miha/Programiranje/BROCCOLI/code/BROCCOLI_LIB -L/home/miha/Programiranje/BROCCOLI/code/BROCCOLI_LIB    -I/home/miha/Programiranje/BROCCOLI/code/BROCCOLI_LIB/Eigen
    
    opencl_platform = 0;  % 0 Intel, 1 AMD, 2 Nvidia
    opencl_device = 0;
    
    
    %% Only used in Octave for compatibility with Matlab
    if exist('do_braindead_shortcircuit_evaluation', 'builtin')
      do_braindead_shortcircuit_evaluation(1);
      warning('off', 'Octave:possible-matlab-short-circuit-operator');
    end
    
    test_data_dir = '/data/miha/BROCCOLI/motion_correction'
else
    if ispc
        addpath('D:\nifti_matlab')
        addpath('D:\BROCCOLI_test_data')
        basepath = 'D:\BROCCOLI_test_data\';
        %mex -g MotionCorrection.cpp -lOpenCL -lBROCCOLI_LIB -IC:/Program' Files'/NVIDIA' GPU Computing Toolkit'/CUDA/v5.0/include -IC:/Program' Files'/NVIDIA' GPU Computing Toolkit'/CUDA/v5.0/include/CL -LC:/Program' Files'/NVIDIA' GPU Computing Toolkit'/CUDA/v5.0/lib/x64 -LC:/users/wande/Documents/Visual' Studio 2010'/Projects/BROCCOLI_LIB/x64/Debug/ -IC:/users/wande/Documents/Visual' Studio 2010'/Projects/BROCCOLI_LIB/BROCCOLI_LIB -IC:\Users\wande\Documents\Visual' Studio 2010'\Projects\BROCCOLI_LIB\nifticlib-2.0.0\niftilib  -IC:\Users\wande\Documents\Visual' Studio 2010'\Projects\BROCCOLI_LIB\nifticlib-2.0.0\znzlib -IC:\Users\wande\Documents\Visual' Studio 2010'\Projects\BROCCOLI_LIB\Eigen
        mex MotionCorrection.cpp -lOpenCL -lBROCCOLI_LIB -IC:/Program' Files'/NVIDIA' GPU Computing Toolkit'/CUDA/v5.0/include -IC:/Program' Files'/NVIDIA' GPU Computing Toolkit'/CUDA/v5.0/include/CL -LC:/Program' Files'/NVIDIA' GPU Computing Toolkit'/CUDA/v5.0/lib/x64 -LC:/users/wande/Documents/Visual' Studio 2010'/Projects/BROCCOLI_LIB/x64/Release/ -IC:/users/wande/Documents/Visual' Studio 2010'/Projects/BROCCOLI_LIB/BROCCOLI_LIB -IC:\Users\wande\Documents\Visual' Studio 2010'\Projects\BROCCOLI_LIB\nifticlib-2.0.0\niftilib  -IC:\Users\wande\Documents\Visual' Studio 2010'\Projects\BROCCOLI_LIB\nifticlib-2.0.0\znzlib -IC:\Users\wande\Documents\Visual' Studio 2010'\Projects\BROCCOLI_LIB\Eigen
        
        opencl_platform = 0;
        opencl_device = 1;
    elseif isunix
        addpath('/home/andek/Research_projects/nifti_matlab')
        basepath = '/data/andek/BROCCOLI_test_data/';
        %mex -g MotionCorrection.cpp -lOpenCL -lBROCCOLI_LIB -I/usr/local/cuda-5.0/include/ -I/usr/local/cuda-5.0/include/CL -L/usr/lib -I/home/andek/Research_projects/BROCCOLI/BROCCOLI/code/BROCCOLI_LIB/ -L/home/andek/cuda-workspace/BROCCOLI_LIB/Debug
        mex MotionCorrection.cpp -lOpenCL -lBROCCOLI_LIB -I/usr/local/cuda-5.0/include/ -I/usr/local/cuda-5.0/include/CL -L/usr/lib -I/home/andek/Research_projects/BROCCOLI/BROCCOLI/code/BROCCOLI_LIB/ -L/home/andek/cuda-workspace/BROCCOLI_LIB/Release -I/home/andek/Research_projects/BROCCOLI/BROCCOLI/code/BROCCOLI_LIB/Eigen/
        
        opencl_platform = 2; % 0 Intel, 1 AMD, 2 Nvidia
        opencl_device = 0;
    end
    
    test_data_dir = '/data/andek/BROCCOLI_test_data/BROCCOLI/motion_correction/'
end

%study = 'Oulu';
%study = 'ICBM';
study = 'Cambridge';
%study = 'Beijing';
%study = 'OpenfMRI';
%substudy = 'Mixed';

save_motion_corrected_data_Matlab = 0;  % Save motion corrected data as a Matlab file or not
save_motion_corrected_data_Nifti = 1;   % Save motion corrected data as a Nifti file or not
plot_results = 1;                       % Plot true and estimated motion parameters or not
save_estimated_motion_parameters = 0;   % Save estimated motion parameters as a Matlab file or not
run_Matlab_equivalent = 0;              % Run Matlab equivalent or not, for comparison to OpenCL algorithm

motion_correction_times = zeros(198,1);

% Loop over subjects
for s = 1:1
    
    s
    
    number_of_iterations_for_motion_correction = 5;
    
    % Load fMRI data
    if ispc
        if ( (strcmp(study,'Beijing')) || (strcmp(study,'Cambridge')) || (strcmp(study,'ICBM')) || (strcmp(study,'Oulu')) )
            EPI_nii = load_nii([basepath study '/rest' num2str(s) '.nii']);
        elseif ( strcmp(study,'OpenfMRI'))
            EPI_nii = load_nii([basepath study '\' substudy '/bold' num2str(s) '.nii']);
        end
    elseif isunix
        dirs = dir([basepath study]);
        subject = dirs(s+2).name % Skip . and .. 'folders'
        EPI_nii = load_nii([basepath study '/' subject '/func/rest.nii']);                
    end
    
    fMRI_volumes = double(EPI_nii.img);        
    [sy sx sz st] = size(fMRI_volumes)
    
    % Load quadrature filters
    load filters_for_parametric_registration.mat
                
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
        filename = [BROCCOLI_motion_corrected_rest_subject_' num2str(s) '_original_motion.mat'];
        motion_corrected_volumes_opencl = single(motion_corrected_volumes_opencl);
        save(filename,'motion_corrected_volumes_opencl');
    end
    
    % Save motion corrected data as nifti file
    if save_motion_corrected_data_Nifti == 1
        new_file.hdr = EPI_nii.hdr;
        new_file.hdr.dime.datatype = 16;
        new_file.hdr.dime.bitpix = 16;
        new_file.img = single(motion_corrected_volumes_opencl);            
        filename = [test_data_dir '/BROCCOLI_motion_corrected_rest_subject_' num2str(s) '_original_motion.nii'];                        
        save_nii(new_file,filename);
    end
    
    % Save estimated motion parameters as Matlab file
    if save_estimated_motion_parameters == 1            
        filename = [test_data_dir '/BROCCOLI_motion_parameters_subject_' num2str(s) '_original_motion.mat'];        
        save(filename,'motion_parameters_opencl');
    end
        
    if plot_results == 1
        
        % Plot true and estimated translations in the x-direction
        figure
        if  (run_Matlab_equivalent == 1)
            plot(motion_parameters_cpu(:,1),'r')
            hold on
            plot(motion_parameters_opencl(:,1),'b')
            hold off
            legend('Estimated x translations CPU','Estimated x translations OpenCL')
        else
            plot(motion_parameters_opencl(:,1),'b')            
            legend('Estimated x translations OpenCL')
        end
    
        % Plot true and estimated translations in the y-direction
        figure
        if  (run_Matlab_equivalent == 1)
            plot(motion_parameters_cpu(:,2),'r')
            hold on
            plot(motion_parameters_opencl(:,2),'b')
            hold off
            legend('Estimated y translations CPU','Estimated y translations OpenCL')
        else        
            plot(motion_parameters_opencl(:,2),'b')            
            legend('Estimated y translations OpenCL')
        end
    
        % Plot true and estimated translations in the z-direction
        figure
        if  (run_Matlab_equivalent == 1)
            plot(motion_parameters_cpu(:,3),'r')
            hold on
            plot(motion_parameters_opencl(:,3),'b')
            hold off
            legend('Estimated z translations CPU','Estimated z translations OpenCL')
        else
            plot(motion_parameters_opencl(:,3),'b')            
            legend('Estimated z translations OpenCL')
        end
        
        % Plot true and estimated rotations in the x-direction
        figure
        if  (run_Matlab_equivalent == 1)
            plot(rotations_cpu(:,1),'r')
            hold on
            plot(motion_parameters_opencl(:,4),'b')
            hold off
            legend('Estimated x rotations CPU','Estimated x rotations OpenCL')
        else            
            plot(motion_parameters_opencl(:,4),'b')
            legend('Estimated x rotations OpenCL')
        end
      
        % Plot true and estimated rotations in the y-direction
        figure
        if  (run_Matlab_equivalent == 1)
            plot(rotations_cpu(:,2),'r')
            hold on
            plot(motion_parameters_opencl(:,5),'b')
            hold off
            legend('Estimated y rotations CPU','Estimated y rotations OpenCL')    
        else
            plot(motion_parameters_opencl(:,5),'b')            
            legend('Estimated y rotations OpenCL')
        end
        
        % Plot true and estimated rotations in the z-direction
        figure
        if  (run_Matlab_equivalent == 1)
            plot(rotations_cpu(:,3),'r')
            hold on
            plot(motion_parameters_opencl(:,6),'b')
            hold off
            legend('Estimated z rotations CPU','Estimated z rotations OpenCL')
        else
            plot(motion_parameters_opencl(:,6),'b')            
            legend('Estimated z rotations OpenCL')
        end
        
    end   
    
    %pause
    %close all
    
    k = waitforbuttonpress
    
end


