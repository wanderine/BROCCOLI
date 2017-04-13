clear all
clc
close all

% Requires nifti matlab package
addpath('/home/andek/Research_projects/nifti_matlab')

basepath = '/data/andek/fcon1000/';

study = 'Cambridge';

noise_level = 0.0; % 0.01, 0.02         % Amount of Gaussian noise, the standard deviation is set to noise_level * max_intensity_value
save_test_dataset = 0;                  % Save testing data as a nifti file or not
plot_motion = 1;                        % Plot random motion parameters
save_true_motion_parameters = 0;        % Save true motion parameters as a Matlab file or not
add_shading = 0;                        % Add shading to each fMRI volume or not

% Loop over subjects
for s = 1:198
    
    s
        
    % Load fMRI data
    if isunix
        dirs = dir([basepath study]);
        subject = dirs(s+2).name % Skip . and .. 'folders'
        EPI_nii = load_nii([basepath study '/' subject '/func/rest.nii.gz']);               
    end
    
    voxel_size_x = EPI_nii.hdr.dime.pixdim(2);
    voxel_size_y = EPI_nii.hdr.dime.pixdim(3);
    voxel_size_z = EPI_nii.hdr.dime.pixdim(4);

    fMRI_volumes = double(EPI_nii.img);        
    [sy sx sz st] = size(fMRI_volumes)
                
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
    
    if plot_motion == 1
        
        figure(1)
        plot(x_translations*voxel_size_x,'g')
        legend('Applied x translations')
    
        figure(2)
        plot(y_translations*voxel_size_y,'g')
        legend('Applied y translations')
    
        figure(3)
        plot(z_translations*voxel_size_z,'g')
        legend('Applied z translations')
        
        figure(4)
        plot(x_rotations,'g')
        legend('Applied x rotations')
      
        figure(5)
        plot(y_rotations,'g')
        legend('Applied y rotations')
        
        figure(6)
        plot(z_rotations,'g')
        legend('Applied z rotations')
    
    end   
    
    
end



