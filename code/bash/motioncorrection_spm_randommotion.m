
clear all
clc
close all

if isunix
    addpath('/data/andek/spm8/')
    data_path = '/data/andek/BROCCOLI_test_data/Cambridge/with_random_motion';
    results_directory = '/data/andek/BROCCOLI_test_data/SPM/';
elseif ispc
    addpath('D:/spm8')
    data_path = 'D:/BROCCOLI_test_data/Cambridge/with_random_motion';
    results_directory = '/data/andek/BROCCOLI_test_data/SPM/';
end

dirs = dir([data_path]);


try
    system(['rm' ' batch_preprocessing.mat']);
end
    
for s = 9:9
    
    

    %% Initialise SPM defaults
    %--------------------------------------------------------------------------
    spm('Defaults','fMRI');
    spm_jobman('initcfg'); % useful in SPM8 only
    
    
    %% WORKING DIRECTORY (useful for .ps only)
    %--------------------------------------------------------------------------
    clear pjobs
    pjobs{1}.util{1}.cdir.directory = cellstr(data_path);

    tic
        
    subject = dirs(s+2).name
    
    filename = [data_path '/' subject];                
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % SPATIAL PREPROCESSING
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    %% Select functional and structural scans
    %--------------------------------------------------------------------------   
    
%     % Select data
%     if file_number < 10
%         f = spm_select('FPList', fullfile(data_path2,['func\subject_000' num2str(file_number) '\']),'func*') ;
%     elseif file_number < 100
%         f = spm_select('FPList', fullfile(data_path2,['func\subject_00' num2str(file_number) '\']),'func*') ;
%     end
%     % Comment if preprocessed files needs to be calculated
%     f = f(1:st,:);
    
    
    %% Motion correction
    %--------------------------------------------------------------------------
    pjobs{2}.spatial{1}.realign{1}.estwrite.data{1} = cellstr(filename);
    % Register to first
    pjobs{2}.spatial{1}.realign{1}.estwrite.eoptions.rtm = 0;  
    % Trilinear interpolation for estimation
    pjobs{2}.spatial{1}.realign{1}.estwrite.eoptions.interp = 1;          
    % Trilinear interpolation for reslice
    pjobs{2}.spatial{1}.realign{1}.estwrite.roptions.interp = 1;          
    % Only reslice images, not mean
    pjobs{2}.spatial{1}.realign{1}.estwrite.roptions.which = [2 0];

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% FINISH
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    save('batch_preprocessing.mat','pjobs');
    
    tic
    error1 = 0;
    try
        %spm_jobman('interactive',pjobs);
        spm_jobman('run',pjobs);        % run preprocessing
    catch err
        err
        error1 = 1;
    end
    toc   
    
    % Move files to results directory 
    motion_corrected_data = ['r' subject ];
    motion_corrected_matrix = ['r' subject(1:end-4) '.mat' ];
    motion_parameters = ['rp_' subject(1:end-4) '.txt'];
    mean_volume = ['mean' subject];
    mat = [subject(1:end-4) '.mat'];
    
    system(['mv ' motion_corrected_data ' ' results_directory]);
    system(['mv ' motion_corrected_matrix ' ' results_directory]);
    system(['mv ' motion_parameters ' ' results_directory]);
    %system(['rm ' mean_volume]);
    system(['rm ' mat]);
    
    s
        
    try
        system(['rm' ' batch_preprocessing.mat']);
    end
    
end








