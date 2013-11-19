%--------------------------------------------------------------------------------
% This script compares estimated motion parameters to true ones, 
% for SPM, FSL, AFNI and BROCCOLI
%--------------------------------------------------------------------------------

clear all
close all
clc

if ispc
    addpath('D:\nifti_matlab')
    addpath('D:\BROCCOLI_test_data\FSL')
    basepath = 'D:\';
    basepath_none = 'D:/BROCCOLI_test_data/Cambridge/with_random_motion';
elseif isunix
    basepath_SPM = '/data/andek/BROCCOLI_test_data/SPM/motion_correction';
    basepath_FSL = '/data/andek/BROCCOLI_test_data/FSL/motion_correction';
    basepath_AFNI = '/data/andek/BROCCOLI_test_data/AFNI/motion_correction';
    basepath_BROCCOLI = '/data/andek/BROCCOLI_test_data/BROCCOLI/motion_correction';
    basepath_none = '/data/andek/BROCCOLI_test_data/Cambridge/with_random_motion';
end

% Select noise level in data

noise_level = '_no_noise';
%noise_level = '_2percent_noise';
%noise_level = '_shading';

N = 198;

voxel_size = 3;

show_parameters = 1;       % Show the estimated parameters as plots or not, for SPM, FSL, AFNI and BROCCOLI
show_errors = 0;           % Show the motion errors or not, for SPM, FSL, AFNI and BROCCOLI

%-----------------------------------------------------------------
% SPM Linear interpolation
%-------------------------------------------------------------------

errors_SPM_Linear = zeros(N,1);

dirs = dir([basepath_none]);

% Loop over subjects
for s = 1:N
    
    s
    
    % Load estimated motion parameters
    fid = fopen([basepath_SPM '/rp_cambridge_rest_subject_' num2str(s) '_with_random_motion' noise_level '.txt']);
    text = textscan(fid,'%f%f%f%f%f%f');
    fclose(fid);
            
    transx = text{1};
    transy = text{2};
    transz = text{3};
    
    rotx = text{4};
    roty = text{5};
    rotz = text{6};
    
    % Convert parameters to BROCCOLI coordinate system
    SPM_translations_x = -transy/voxel_size;
    SPM_translations_y = -transx/voxel_size;
    SPM_translations_z = -transz/voxel_size;
    
    SPM_rotations_x = roty*180/pi;
    SPM_rotations_y = -rotx*180/pi;
    SPM_rotations_z = -rotz*180/pi;
    
    % Load true parameters
    load([basepath_none '/true_motion_parameters_subject_' num2str(s) '_random_motion' noise_level '.mat']);
    
    % Calculate errors
    errors = zeros(119,6);
    errors(:,1) = SPM_translations_x - x_translations;
    errors(:,2) = SPM_translations_y - y_translations;
    errors(:,3) = SPM_translations_z - z_translations;
    errors(:,4) = SPM_rotations_x - x_rotations;
    errors(:,5) = SPM_rotations_y - y_rotations;
    errors(:,6) = SPM_rotations_z - z_rotations;
    
    errors_SPM_Linear(s) = sqrt(sum(sum(errors.^2)));
end

%-----------------------------------------------------------------
% SPM B-spline interpolation (default)
%-------------------------------------------------------------------

errors_SPM_Spline = zeros(N,1);

dirs = dir([basepath_none]);

% Loop over subjects
for s = 1:N
    
    s
    
    % Load estimated motion parameters
    fid = fopen([basepath_SPM '/rp_cambridge_rest_subject_' num2str(s) '_with_random_motion' noise_level '_spline.txt']);
    text = textscan(fid,'%f%f%f%f%f%f');
    fclose(fid);
            
    transx = text{1};
    transy = text{2};
    transz = text{3};
    
    rotx = text{4};
    roty = text{5};
    rotz = text{6};
    
    % Convert parameters to BROCCOLI coordinate system
    SPM_translations_x = -transy/voxel_size;
    SPM_translations_y = -transx/voxel_size;
    SPM_translations_z = -transz/voxel_size;
    
    SPM_rotations_x = roty*180/pi;
    SPM_rotations_y = -rotx*180/pi;
    SPM_rotations_z = -rotz*180/pi;
    
    % Load true parameters
    load([basepath_none '/true_motion_parameters_subject_' num2str(s) '_random_motion' noise_level '.mat']);
    
    % Calculate errors
    errors = zeros(119,6);
    errors(:,1) = SPM_translations_x - x_translations;
    errors(:,2) = SPM_translations_y - y_translations;
    errors(:,3) = SPM_translations_z - z_translations;
    errors(:,4) = SPM_rotations_x - x_rotations;
    errors(:,5) = SPM_rotations_y - y_rotations;
    errors(:,6) = SPM_rotations_z - z_rotations;
    
    errors_SPM_Spline(s) = sqrt(sum(sum(errors.^2)));
end

%-----------------------------------------------------------------
% FSL Linear interpolation (default)
%-------------------------------------------------------------------

errors_FSL = zeros(N,1);

% Loop over subjects
for s = 1:N
    
    % Load estimated motion parameters
    fid = fopen([basepath_FSL '/FSL_motion_corrected_subject' num2str(s) '_random_motion' noise_level '.nii.par']);
    text = textscan(fid,'%f%f%f%f%f%f');
    fclose(fid);
       
    rotx = text{1};
    roty = text{2};
    rotz = text{3};
    
    transx = text{4};
    transy = text{5};
    transz = text{6};
    
    % Convert parameters to BROCCOLI coordinate system
    FSL_translations_x = -transy/voxel_size;
    FSL_translations_y = transx/voxel_size;
    FSL_translations_z = -transz/voxel_size;
    FSL_rotations_x = roty*180/pi;
    FSL_rotations_y = -rotx*180/pi;
    FSL_rotations_z = rotz*180/pi;
    
    % Load true parameters
    load([basepath_none '/true_motion_parameters_subject_' num2str(s) '_random_motion' noise_level '.mat']);
    
    % Calculate errors
    errors = zeros(119,6);
    errors(:,1) = FSL_translations_x - x_translations;
    errors(:,2) = FSL_translations_y - y_translations;
    errors(:,3) = FSL_translations_z - z_translations;
    errors(:,4) = FSL_rotations_x - x_rotations;
    errors(:,5) = FSL_rotations_y - y_rotations;
    errors(:,6) = FSL_rotations_z - z_rotations;
    
    errors_FSL(s) = sqrt(sum(sum(errors.^2)));
end



%-----------------------------------------------------------------------
% AFNI Linear interpolation
%-------------------------------------------------------------------

errors_AFNI_Linear = zeros(N,1);

% Loop over subjects
for s = 1:N
    
    % Load estimated motion parameters
    fid = fopen([basepath_AFNI '/AFNI_motion_parameters_subject' num2str(s) '_random_motion' noise_level '.1D']);
    text = textscan(fid,'%f%f%f%f%f%f');
    fclose(fid);
    
    roll = text{1};
    pitch = text{2};
    yaw = text{3};
    
    dS = text{4}; % Superior
    dL = text{5}; % Left
    dP = text{6}; % Posterior
    
    % Convert parameters to BROCCOLI coordinate system
    AFNI_translations_x = dP/voxel_size;
    AFNI_translations_y = dL/voxel_size;
    AFNI_translations_z = -dS/voxel_size;
    AFNI_rotations_x = -yaw;
    AFNI_rotations_y = -pitch;
    AFNI_rotations_z = roll;
    
    % Load true parameters
    load([basepath_none '/true_motion_parameters_subject_' num2str(s) '_random_motion' noise_level '.mat']);
    
    % Calculate errors
    errors = zeros(119,6);
    errors(:,1) = AFNI_translations_x - x_translations;
    errors(:,2) = AFNI_translations_y - y_translations;
    errors(:,3) = AFNI_translations_z - z_translations;
    errors(:,4) = AFNI_rotations_x - x_rotations;
    errors(:,5) = AFNI_rotations_y - y_rotations;
    errors(:,6) = AFNI_rotations_z - z_rotations;
    
    errors_AFNI_Linear(s) = sqrt(sum(sum(errors.^2)));
    
end


%-----------------------------------------------------------------------
% AFNI Fourier interpolation (default)
%-------------------------------------------------------------------

errors_AFNI_Fourier = zeros(N,1);

% Loop over subjects
for s = 1:N
    
    % Load estimated motion parameters
    fid = fopen([basepath_AFNI '/AFNI_motion_parameters_subject' num2str(s) '_random_motion' noise_level '_Fourier.1D']);
    text = textscan(fid,'%f%f%f%f%f%f');
    fclose(fid);
    
    roll = text{1};
    pitch = text{2};
    yaw = text{3};
    
    dS = text{4}; % Superior
    dL = text{5}; % Left
    dP = text{6}; % Posterior
    
    % Convert parameters to BROCCOLI coordinate system
    AFNI_translations_x = dP/voxel_size;
    AFNI_translations_y = dL/voxel_size;
    AFNI_translations_z = -dS/voxel_size;
    AFNI_rotations_x = -yaw;
    AFNI_rotations_y = -pitch;
    AFNI_rotations_z = roll;
    
    % Load true parameters
    load([basepath_none '/true_motion_parameters_subject_' num2str(s) '_random_motion' noise_level '.mat']);
    
    % Calculate errors
    errors = zeros(119,6);
    errors(:,1) = AFNI_translations_x - x_translations;
    errors(:,2) = AFNI_translations_y - y_translations;
    errors(:,3) = AFNI_translations_z - z_translations;
    errors(:,4) = AFNI_rotations_x - x_rotations;
    errors(:,5) = AFNI_rotations_y - y_rotations;
    errors(:,6) = AFNI_rotations_z - z_rotations;
    
    errors_AFNI_Fourier(s) = sqrt(sum(sum(errors.^2)));
    
end



%-------------------------------------------------------------------
% BROCCOLI Linear interpolation (default)
%-------------------------------------------------------------------

errors_BROCCOLI = zeros(N,1);

for s = 1:N
    
    % Load estimated motion parameters
    load([basepath_BROCCOLI '/BROCCOLI_motion_parameters_subject_' num2str(s) '_random_motion' noise_level '.mat']);
    
    BROCCOLI_translations_x = motion_parameters_opencl(:,1);
    BROCCOLI_translations_y = motion_parameters_opencl(:,2);
    BROCCOLI_translations_z = motion_parameters_opencl(:,3);
    BROCCOLI_rotations_x = motion_parameters_opencl(:,4);
    BROCCOLI_rotations_y = motion_parameters_opencl(:,5);
    BROCCOLI_rotations_z = motion_parameters_opencl(:,6);
    
    % Load true parameters
    load([basepath_none '/true_motion_parameters_subject_' num2str(s) '_random_motion' noise_level '.mat']);
    
    % Calculate errors
    errors = zeros(119,6);
    errors(:,1) = BROCCOLI_translations_x - x_translations;
    errors(:,2) = BROCCOLI_translations_y - y_translations;
    errors(:,3) = BROCCOLI_translations_z - z_translations;
    errors(:,4) = BROCCOLI_rotations_x - x_rotations;
    errors(:,5) = BROCCOLI_rotations_y - y_rotations;
    errors(:,6) = BROCCOLI_rotations_z - z_rotations;
    
    errors_BROCCOLI(s) = sqrt(sum(sum(errors.^2)));
        
end

SPM_Linear_meanerror = mean(errors_SPM_Linear)
SPM_Spline_meanerror = mean(errors_SPM_Spline)
FSL_meanerror = mean(errors_FSL)
AFNI_Linear_meanerror = mean(errors_AFNI_Linear)
AFNI_Fourier_meanerror = mean(errors_AFNI_Fourier)
BROCCOLI_meanerror = mean(errors_BROCCOLI)

SPM_Linear_std = std(errors_SPM_Linear)
SPM_Spline_std = std(errors_SPM_Spline)
FSL_std = std(errors_FSL)
AFNI_Linear_std = std(errors_AFNI_Linear)
AFNI_Fourier_std = std(errors_AFNI_Fourier)
BROCCOLI_std = std(errors_BROCCOLI)

% Plot estimated parameters for SPM, FSL, AFNI och BROCCOLI
if show_parameters == 1
    
    % Loop over subjects
    for s = 1:N
        
        %-----------------
        % SPM Spline
        %-----------------
                
        fid = fopen([basepath_SPM '/rp_cambridge_rest_subject_' num2str(s) '_with_random_motion' noise_level '_spline.txt']);
        text = textscan(fid,'%f%f%f%f%f%f');
        fclose(fid);
        
        transx = text{1};
        transy = text{2};
        transz = text{3};
        
        rotx = text{4};
        roty = text{5};
        rotz = text{6};
        
		% Transform coordinates to BROCCOLI space and compare to true
        % parameters        
        SPM_translations_x = -transy/voxel_size;
        SPM_translations_y = -transx/voxel_size;
        SPM_translations_z = -transz/voxel_size;                
        SPM_rotations_x = roty*180/pi;
        SPM_rotations_y = -rotx*180/pi;
        SPM_rotations_z = -rotz*180/pi;
        
        %-----------------
        % FSL
        %-----------------
        
        fid = fopen([basepath_FSL '/FSL_motion_corrected_subject' num2str(s) '_random_motion' noise_level '.nii.par']);
        text = textscan(fid,'%f%f%f%f%f%f');
        fclose(fid);
        
        rotx = text{1};
        roty = text{2};
        rotz = text{3};
        
        transx = text{4};
        transy = text{5};
        transz = text{6};
        
        % Transform coordinates to BROCCOLI space and compare to true
        % parameters
        
        FSL_translations_x = -transy/voxel_size;
        FSL_translations_y = transx/voxel_size;
        FSL_translations_z = -transz/voxel_size;
        FSL_rotations_x = roty*180/pi;
        FSL_rotations_y = -rotx*180/pi;
        FSL_rotations_z = rotz*180/pi;
        
        %-----------------
        % AFNI Fourier
        %-----------------
        
        fid = fopen([basepath_AFNI '/AFNI_motion_parameters_subject' num2str(s) '_random_motion' noise_level '_Fourier.1D']);
        text = textscan(fid,'%f%f%f%f%f%f');
        fclose(fid);
        
        roll = text{1};
        pitch = text{2};
        yaw = text{3};
        
        dS = text{4}; % Superior
        dL = text{5}; % Left
        dP = text{6}; % Posterior
        
        % Transform coordinates to BROCCOLI space and compare to true
        % parameters
        AFNI_translations_x = dP/voxel_size;
        AFNI_translations_y = dL/voxel_size;
        AFNI_translations_z = -dS/voxel_size;
        AFNI_rotations_x = -yaw;
        AFNI_rotations_y = -pitch;
        AFNI_rotations_z = roll;
        
        %-----------------
        % BROCCOLI
        %-----------------
        
        load([basepath_BROCCOLI '/BROCCOLI_motion_parameters_subject_' num2str(s) '_random_motion' noise_level '.mat']);
        
        BROCCOLI_translations_x = motion_parameters_opencl(:,1);
        BROCCOLI_translations_y = motion_parameters_opencl(:,2);
        BROCCOLI_translations_z = motion_parameters_opencl(:,3);
        BROCCOLI_rotations_x = motion_parameters_opencl(:,4);
        BROCCOLI_rotations_y = motion_parameters_opencl(:,5);
        BROCCOLI_rotations_z = motion_parameters_opencl(:,6);
        
        %-----------------
        % True
        %-----------------
        
        % Load true parameters
        load([basepath_none '/true_motion_parameters_subject_' num2str(s) '_random_motion' noise_level '.mat']);
        
        figure(1)
        subplot(3,1,1)
        plot(SPM_translations_x,'c')
        hold on
        plot(FSL_translations_x,'r')
        hold on
        plot(AFNI_translations_x,'g')
        hold on
        plot(BROCCOLI_translations_x,'b')
        hold on
        plot(x_translations,'k')
        hold off
        xlabel('TR')
        ylabel('Voxels')
        title('X translations')
        legend('SPM','FSL','AFNI','BROCCOLI','Ground truth')
        
        subplot(3,1,2)
        plot(SPM_translations_y,'c')
        hold on
        plot(FSL_translations_y,'r')
        hold on
        plot(AFNI_translations_y,'g')
        hold on
        plot(BROCCOLI_translations_y,'b')
        hold on
        plot(y_translations,'k')
        hold off
        xlabel('TR')
        ylabel('Voxels')
        title('Y translations')
        legend('SPM','FSL','AFNI','BROCCOLI','Ground truth')
        
        subplot(3,1,3)
        plot(SPM_translations_z,'c')
        hold on
        plot(FSL_translations_z,'r')
        hold on
        plot(AFNI_translations_z,'g')
        hold on
        plot(BROCCOLI_translations_z,'b')
        hold on
        plot(z_translations,'k')
        hold off
        xlabel('TR')
        ylabel('Voxels')
        title('Z translations')
        legend('SPM','FSL','AFNI','BROCCOLI','Ground truth')
        
        figure(2)
        subplot(3,1,1)
        plot(SPM_rotations_x,'c')
        hold on
        plot(FSL_rotations_x,'r')
        hold on
        plot(AFNI_rotations_x,'g')
        hold on
        plot(BROCCOLI_rotations_x,'b')
        hold on
        plot(x_rotations,'k')
        hold off
        xlabel('TR')
        ylabel('Degrees')
        title('X rotations')
        legend('SPM','FSL','AFNI','BROCCOLI','Ground truth')
        
        subplot(3,1,2)
        plot(SPM_rotations_y,'c')
        hold on
        plot(FSL_rotations_y,'r')
        hold on
        plot(AFNI_rotations_y,'g')
        hold on
        plot(BROCCOLI_rotations_y,'b')
        hold on
        plot(y_rotations,'k')
        hold off
        xlabel('TR')
        ylabel('Degrees')
        title('Y rotations')
        legend('SPM','FSL','AFNI','BROCCOLI','Ground truth')
        
        subplot(3,1,3)
        plot(SPM_rotations_z,'c')
        hold on
        plot(FSL_rotations_z,'r')
        hold on
        plot(AFNI_rotations_z,'g')
        hold on
        plot(BROCCOLI_rotations_z,'b')
        hold on
        plot(z_rotations,'k')
        hold off
        xlabel('TR')
        ylabel('Degrees')
        title('Z rotations')
        legend('SPM','FSL','AFNI','BROCCOLI','Ground truth')
        
        pause
        
        
    end
    
end



% Plot errors for SPM, FSL, AFNI och BROCCOLI
if show_errors == 1
    
    % Loop over subjects
    for s = 1:N
        
        %-----------------
        % SPM Spline
        %-----------------
                
        fid = fopen([basepath_SPM '/rp_cambridge_rest_subject_' num2str(s) '_with_random_motion' noise_level '_spline.txt']);
        text = textscan(fid,'%f%f%f%f%f%f');
        fclose(fid);
        
        transx = text{1};
        transy = text{2};
        transz = text{3};
        
        rotx = text{4};
        roty = text{5};
        rotz = text{6};
        
		% Transform coordinates to BROCCOLI space and compare to true
        % parameters        
        SPM_translations_x = -transy/voxel_size;
        SPM_translations_y = -transx/voxel_size;
        SPM_translations_z = -transz/voxel_size;                
        SPM_rotations_x = roty*180/pi;
        SPM_rotations_y = -rotx*180/pi;
        SPM_rotations_z = -rotz*180/pi;               
        
        %-----------------
        % FSL
        %-----------------
        
        fid = fopen([basepath_FSL '/FSL_motion_corrected_subject' num2str(s) '_random_motion' noise_level '.nii.par']);
        text = textscan(fid,'%f%f%f%f%f%f');
        fclose(fid);
        
        rotx = text{1};
        roty = text{2};
        rotz = text{3};
        
        transx = text{4};
        transy = text{5};
        transz = text{6};
        
        % Transform coordinates to BROCCOLI space and compare to true
        % parameters
        
        FSL_translations_x = -transy/voxel_size;
        FSL_translations_y = transx/voxel_size;
        FSL_translations_z = -transz/voxel_size;
        FSL_rotations_x = roty*180/pi;
        FSL_rotations_y = -rotx*180/pi;
        FSL_rotations_z = rotz*180/pi;
        
        %-----------------
        % AFNI Fourier
        %-----------------
        
        fid = fopen([basepath_AFNI '/AFNI_motion_parameters_subject' num2str(s) '_random_motion' noise_level '_Fourier.1D']);
        text = textscan(fid,'%f%f%f%f%f%f');
        fclose(fid);
        
        roll = text{1};
        pitch = text{2};
        yaw = text{3};
        
        dS = text{4}; % Superior
        dL = text{5}; % Left
        dP = text{6}; % Posterior
        
        % Transform coordinates to BROCCOLI space and compare to true
        % parameters
        AFNI_translations_x = dP/voxel_size;
        AFNI_translations_y = dL/voxel_size;
        AFNI_translations_z = -dS/voxel_size;
        AFNI_rotations_x = -yaw;
        AFNI_rotations_y = -pitch;
        AFNI_rotations_z = roll;
        
        %-----------------
        % BROCCOLI
        %-----------------
        
        load([basepath_BROCCOLI '/BROCCOLI_motion_parameters_subject_' num2str(s) '_random_motion' noise_level '.mat']);
        
        BROCCOLI_translations_x = motion_parameters_opencl(:,1);
        BROCCOLI_translations_y = motion_parameters_opencl(:,2);
        BROCCOLI_translations_z = motion_parameters_opencl(:,3);
        BROCCOLI_rotations_x = motion_parameters_opencl(:,4);
        BROCCOLI_rotations_y = motion_parameters_opencl(:,5);
        BROCCOLI_rotations_z = motion_parameters_opencl(:,6);
        
        %-----------------
        % True
        %-----------------
        
        % Load true parameters
        load([basepath_none '/true_motion_parameters_subject_' num2str(s) '_random_motion' noise_level '.mat']);
        
        %-----------------
        % Errors
        %-----------------
        
        SPM_translations_x_error = SPM_translations_x - x_translations;
        SPM_translations_y_error = SPM_translations_y - y_translations;
        SPM_translations_z_error = SPM_translations_z - z_translations;
        
        SPM_rotations_x_error = SPM_rotations_x - x_rotations;
        SPM_rotations_y_error = SPM_rotations_y - y_rotations;
        SPM_rotations_z_error = SPM_rotations_z - z_rotations;
        
        FSL_translations_x_error = FSL_translations_x - x_translations;
        FSL_translations_y_error = FSL_translations_y - y_translations;
        FSL_translations_z_error = FSL_translations_z - z_translations;
        
        FSL_rotations_x_error = FSL_rotations_x - x_rotations;
        FSL_rotations_y_error = FSL_rotations_y - y_rotations;
        FSL_rotations_z_error = FSL_rotations_z - z_rotations;
        
        AFNI_translations_x_error = AFNI_translations_x - x_translations;
        AFNI_translations_y_error = AFNI_translations_y - y_translations;
        AFNI_translations_z_error = AFNI_translations_z - z_translations;
        
        AFNI_rotations_x_error = AFNI_rotations_x - x_rotations;
        AFNI_rotations_y_error = AFNI_rotations_y - y_rotations;
        AFNI_rotations_z_error = AFNI_rotations_z - z_rotations;
        
        BROCCOLI_translations_x_error = BROCCOLI_translations_x - x_translations;
        BROCCOLI_translations_y_error = BROCCOLI_translations_y - y_translations;
        BROCCOLI_translations_z_error = BROCCOLI_translations_z - z_translations;
        
        BROCCOLI_rotations_x_error = BROCCOLI_rotations_x - x_rotations;
        BROCCOLI_rotations_y_error = BROCCOLI_rotations_y - y_rotations;
        BROCCOLI_rotations_z_error = BROCCOLI_rotations_z - z_rotations;
        
        figure(1)
        subplot(3,1,1)
        plot(SPM_translations_x_error,'c')
        hold on
        plot(FSL_translations_x_error,'r')
        hold on
        plot(AFNI_translations_x_error,'g')
        hold on
        plot(BROCCOLI_translations_x_error,'b')
        hold off
        xlabel('TR')
        ylabel('Voxels')
        title('X translations')
        legend('SPM error','FSL error','AFNI error','BROCCOLI error')
        
        subplot(3,1,2)
        plot(SPM_translations_y_error,'c')
        hold on
        plot(FSL_translations_y_error,'r')
        hold on
        plot(AFNI_translations_y_error,'g')
        hold on
        plot(BROCCOLI_translations_y_error,'b')
        hold on
        xlabel('TR')
        ylabel('Voxels')
        title('Y translations')
        legend('SPM error','FSL error','AFNI error','BROCCOLI error')
        
        subplot(3,1,3)
        plot(SPM_translations_z_error,'c')
        hold on
        plot(FSL_translations_z_error,'r')
        hold on
        plot(AFNI_translations_z_error,'g')
        hold on
        plot(BROCCOLI_translations_z_error,'b')
        hold off
        xlabel('TR')
        ylabel('Voxels')
        title('Z translations')
        legend('SPM error','FSL error','AFNI error','BROCCOLI error')
        
        figure(2)
        subplot(3,1,1)
        plot(SPM_rotations_x_error,'c')
        hold on
        plot(FSL_rotations_x_error,'r')
        hold on
        plot(AFNI_rotations_x_error,'g')
        hold on
        plot(BROCCOLI_rotations_x_error,'b')
        hold off
        xlabel('TR')
        ylabel('Degrees')
        title('X rotations')
        legend('SPM error','FSL error','AFNI error','BROCCOLI error')
        
        subplot(3,1,2)
        plot(SPM_rotations_y_error,'c')
        hold on
        plot(FSL_rotations_y_error,'r')
        hold on
        plot(AFNI_rotations_y_error,'g')
        hold on
        plot(BROCCOLI_rotations_y_error,'b')
        hold off
        xlabel('TR')
        ylabel('Degrees')
        title('Y rotations')
        legend('SPM error','FSL error','AFNI error','BROCCOLI error')
        
        subplot(3,1,3)
        plot(SPM_rotations_z_error,'c')
        hold on
        plot(FSL_rotations_z_error,'r')
        hold on
        plot(AFNI_rotations_z_error,'g')
        hold on
        plot(BROCCOLI_rotations_z_error,'b')
        hold off
        xlabel('TR')
        ylabel('Degrees')
        title('Z rotations')
        legend('SPM error','FSL error','AFNI error','BROCCOLI error')
        
        pause
        
        
    end
    
end

