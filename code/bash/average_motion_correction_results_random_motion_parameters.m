clear all
close all
clc

if ispc
    addpath('D:\nifti_matlab')
    addpath('D:\BROCCOLI_test_data\FSL')
    basepath = 'D:\';
elseif isunix
    basepath_SPM = '/data/andek/BROCCOLI_test_data/SPM';
    basepath_FSL = '/data/andek/BROCCOLI_test_data/FSL';
    basepath_AFNI = '/data/andek/BROCCOLI_test_data/AFNI';
    basepath_BROCCOLI = '/data/andek/BROCCOLI_test_data/BROCCOLI/random_motion';
    basepath_none = '/data/andek/BROCCOLI_test_data/Cambridge/with_random_motion';
end

noise_level = '_1percent_noise';

N = 1;


%-----------------------------------------------------------------
% SPM
%-------------------------------------------------------------------

errors_SPM = zeros(N,1);

dirs = dir([basepath_none]);
for s = 1:N
    
    subject = dirs(s+2).name;
    
    fid = fopen([basepath_SPM '/rp_' subject(1:end-4) '.txt']);
    text = textscan(fid,'%f%f%f%f%f%f');
    fclose(fid);
    
    transx = text{1};
    transy = text{2};
    transz = text{3};
    
    rotx = text{4};
    roty = text{5};
    rotz = text{6};
    
    SPM_translations_x = zeros(119,1);
    SPM_translations_y = zeros(119,1);
    SPM_translations_z = zeros(119,1);
    SPM_rotations_x = zeros(119,1);
    SPM_rotations_y = zeros(119,1);
    SPM_rotations_z = zeros(119,1);
    
    for t = 1:119
        
        M = zeros(4,4);
        M(1,4) = transx(t);
        M(2,4) = transy(t);
        M(3,4) = transz(t);
        
        M(4,:) = [0 0 0 1];
        
        R_x = [1                        0                           0;
               0                        cos(rotx(t))      -sin(rotx(t));
               0                        sin(rotx(t))      cos(rotx(t))];
        
        R_y = [cos(roty(t))                0              sin(roty(t));
               0                        1              0;
               -sin(roty(t))  0                           cos(roty(t))];
        
        R_z = [cos(rotz(t))   -sin(rotz(t))     0;
               sin(rotz(t))   cos(rotz(t))      0;
               0                        0                           1];
        
        M(1:3,1:3) = R_x * R_y * R_z;
           
        centerMM = 	[1	0	0	72/2-3/2;
                     0	1	0	72/2-3/2;
                     0	0	1	47/2-3/2;
                     0	0	0	        1];
        
        
        % Adjust for different center of rotation
        MSPM = centerMM*M*inv(centerMM);
    
        SPM_translations_x(t) = MSPM(1,4);
        SPM_translations_y(t) = MSPM(2,4);
        SPM_translations_z(t) = MSPM(3,4);
        SPM_rotations_x(t) = MSPM(1,1);
        SPM_rotations_y(t) = MSPM(2,2);
        SPM_rotations_z(t) = MSPM(3,3);
    
    end
    
    
    % Transform coordinates to BROCCOLI space and compare to true
    % parameters
    
    
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

mean(errors_SPM)
mean(errors_FSL)
mean(errors_AFNI)
mean(errors_BROCCOLI)

std(errors_SPM)
std(errors_FSL)
std(errors_AFNI)
std(errors_BROCCOLI)




% Check results visually
for s = 1:1
    
    %-----------------
    % SPM
    %-----------------
    
    subject = dirs(s+2).name;
    
    fid = fopen([basepath_SPM '/rp_' subject(1:end-4) '.txt']);
    text = textscan(fid,'%f%f%f%f%f%f');
    fclose(fid);
    
    transx = text{1};
    transy = text{2};
    transz = text{3};
    
    rotx = text{4};
    roty = text{5};
    rotz = text{6};
    
    MMtoVOX =	[1/3 0	 0	 0;
                 0	 1/3 0	 0;
                 0	 0	 1/3 0;
                 0	 0	 0	 1];
    
    flip    =	[-1   0	 0	 0;
                 0	 1   0	 0;
                 0	 0	 1   0;
                 0	 0	 0	 1];
                 
    centerMM = 	[1	0	0	72/2
                 0	1	0	72/2;
                 0	0	1	47/2;
                 0	0	0	   1];

    shift    = 	[1	0	0	1;
                 0	1	0	1;
                 0	0	1	1;
                 0	0	0	1];
             
    for t = 1:119
        
        M = zeros(4,4);
        M(1,4) = transx(t);
        M(2,4) = transy(t);
        M(3,4) = transz(t);
        
        M(4,:) = [0 0 0 1];
        
        R_x = [1                        0                           0;
               0                        cos(rotx(t))      -sin(rotx(t));
               0                        sin(rotx(t))      cos(rotx(t))];
        
        R_y = [cos(roty(t))                0              sin(roty(t));
               0                        1              0;
               -sin(roty(t))  0                           cos(roty(t))];
        
        R_z = [cos(rotz(t))   -sin(rotz(t))     0;
               sin(rotz(t))   cos(rotz(t))      0;
               0                        0                           1];
        
        M(1:3,1:3) = R_y * R_x * R_z;
           

        
        
        % Adjust for different center of rotation
        MSPM = centerMM*flip*M*shift*MMtoVOX;
    
        SPM_translations_x(t) = MSPM(1,4) - 72/2;
        SPM_translations_y(t) = MSPM(2,4) - 72/2;
        SPM_translations_z(t) = MSPM(3,4) - 72/2;
        SPM_rotations_x(t) = MSPM(1,1);
        SPM_rotations_y(t) = MSPM(2,2);
        SPM_rotations_z(t) = MSPM(3,3);
        
    end
        
    
    %-----------------
    % FSL
    %-----------------
    
    fid = fopen([basepath_FSL '/FSL_motion_corrected_subject' num2str(s) '_random_motion.nii.par']);
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
    
    fid = fopen([basepath_AFNI '/AFNI_motion_parameters_subject' num2str(s) '_random_motion.1D']);
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
    
    load([basepath_BROCCOLI '/BROCCOLI_motion_parameters_subject_' num2str(s) '_random_motion.mat']);
    
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
    load([basepath_none '/true_motion_parameters_subject_' num2str(s) '_random_motion.mat']);
    
    figure(1)
    plot(SPM_translations_z/3+4,'c')
    hold on
    plot(FSL_translations_x,'r')
    hold on
    plot(AFNI_translations_x,'g')
    hold on
    plot(BROCCOLI_translations_x,'b')
    hold on
    plot(x_translations,'k')
    hold off
    %pause
    
%     
%     figure(1)
%     plot(SPM_rotations_y,'y')
%     hold on
%     plot(FSL_rotations_z,'r')
%     hold on
%     plot(AFNI_rotations_z,'g')
%     hold on
%     plot(BROCCOLI_rotations_z,'b')
%     hold on
%     plot(z_rotations,'k')
%     hold off
    %     pause
    
    
end

