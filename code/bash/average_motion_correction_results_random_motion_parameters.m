clear all
close all
clc

if ispc
    addpath('D:\nifti_matlab')
    addpath('D:\BROCCOLI_test_data\FSL')
    basepath = 'D:\';
    basepath_none = 'D:/BROCCOLI_test_data/Cambridge/with_random_motion';
elseif isunix
    basepath_SPM = '/data/andek/BROCCOLI_test_data/SPM';
    basepath_FSL = '/data/andek/BROCCOLI_test_data/FSL';
    basepath_AFNI = '/data/andek/BROCCOLI_test_data/AFNI';
    basepath_BROCCOLI = '/data/andek/BROCCOLI_test_data/BROCCOLI/random_motion';
    basepath_none = '/data/andek/BROCCOLI_test_data/Cambridge/with_random_motion';
end

noise_level = '_1percent_noise';
%noise_level = '_no_noise';
%noise_level = '';

N = 198;

voxel_size = 3;

doplots = 1;


%-----------------------------------------------------------------
% SPM
%-------------------------------------------------------------------

errors_SPM = zeros(N,1);

dirs = dir([basepath_none]);
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
    
    errors_SPM(s) = sum(sum(errors.^2));
end

%-----------------------------------------------------------------
% FSL
%-------------------------------------------------------------------

errors_FSL = zeros(N,1);

for s = 1:N
    
    % Load estimated motion parameters
    fid = fopen([basepath_FSL '/FSL_motion_corrected_subject' num2str(s) '_random_motion' noise_level '.nii.par']);
    text = textscan(fid,'%f%f%f%f%f%f');
    fclose(fid);
    
    rotx = text{1}*180/pi;
    roty = text{2}*180/pi;
    rotz = text{3}*180/pi;
    
    transx = text{4};
    transy = text{5};
    transz = text{6};
    
    % Convert parameters to BROCCOLI coordinate system
    FSL_translations_x = -transy/voxel_size;
    FSL_translations_y = transx/voxel_size;
    FSL_translations_z = -transz/voxel_size;
    FSL_rotations_x = roty;
    FSL_rotations_y = -rotx;
    FSL_rotations_z = rotz;
    
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
    
    errors_FSL(s) = sum(sum(errors.^2));
end



%-----------------------------------------------------------------------
% AFNI
%-------------------------------------------------------------------

errors_AFNI = zeros(N,1);

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
    
    errors_AFNI(s) = sum(sum(errors.^2));
    
end



%-------------------------------------------------------------------
% BROCCOLI
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
    
    errors_BROCCOLI(s) = sum(sum(errors.^2));
    
    
end

SPM_meanerror = mean(errors_SPM)
FSL_meanerror = mean(errors_FSL)
AFNI_meanerror = mean(errors_AFNI)
BROCCOLI_meanerror = mean(errors_BROCCOLI)

SPM_std = std(errors_SPM)
FSL_std = std(errors_FSL)
AFNI_std = std(errors_AFNI)
BROCCOLI_std = std(errors_BROCCOLI)


if doplots == 1
    
    % Check results visually
    for s = 1:N
        
        %-----------------
        % SPM
        %-----------------
        
        
        fid = fopen([basepath_SPM '/rp_cambridge_rest_subject_' num2str(s) '_with_random_motion' noise_level '.txt']);
        text = textscan(fid,'%f%f%f%f%f%f');
        fclose(fid);
        
        transx = text{1};
        transy = text{2};
        transz = text{3};
        
        rotx = text{4};
        roty = text{5};
        rotz = text{6};
        
        
        SPM_translations_x = -transy/3;
        SPM_translations_y = -transx/3;
        SPM_translations_z = -transz/3;
        
        
        SPM_rotations_x = roty*180/pi;
        SPM_rotations_y = -rotx*180/pi;
        SPM_rotations_z = -rotz*180/pi;
        
        %-----------------
        % FSL
        %-----------------
        
        fid = fopen([basepath_FSL '/FSL_motion_corrected_subject' num2str(s) '_random_motion' noise_level '.nii.par']);
        text = textscan(fid,'%f%f%f%f%f%f');
        fclose(fid);
        
        rotx = text{1}*180/pi;
        roty = text{2}*180/pi;
        rotz = text{3}*180/pi;
        
        transx = text{4};
        transy = text{5};
        transz = text{6};
        
        % Transform coordinates to BROCCOLI space and compare to true
        % parameters
        
        FSL_translations_x = -transy/3;
        FSL_translations_y = transx/3;
        FSL_translations_z = -transz/3;
        FSL_rotations_x = roty;
        FSL_rotations_y = -rotx;
        FSL_rotations_z = rotz;
        
        %-----------------
        % AFNI
        %-----------------
        
        fid = fopen([basepath_AFNI '/AFNI_motion_parameters_subject' num2str(s) '_random_motion' noise_level '.1D']);
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
        AFNI_translations_x = dP/3;
        AFNI_translations_y = dL/3;
        AFNI_translations_z = -dS/3;
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
        
        pause
        
        
    end
    
end


