
clear all
clc
close all

basepath = '/data/andek/BROCCOLI_test_data/Cambridge/with_random_motion';
dirs = dir([basepath]);

for s = 1:198
    
    tic
        
    subject = dirs(s+2).name
    
    filename = [basepath '/cambridge_rest_' subject  '_with_random_motion.nii'];
            
    %% Path containing data
    %--------------------------------------------------------------------------
    data_path = 'E:\rest_fMRI\func_only_1\';
    data_path2 = 'E:\rest_fMRI\';
    
    %% Initialise SPM defaults
    %--------------------------------------------------------------------------
    spm('Defaults','fMRI');
    
    spm_jobman('initcfg'); % useful in SPM8 only
    
    
    %% WORKING DIRECTORY (useful for .ps only)
    %--------------------------------------------------------------------------
    clear pjobs
    pjobs{1}.util{1}.cdir.directory = cellstr(data_path);
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % SPATIAL PREPROCESSING
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    %% Select functional and structural scans
    %--------------------------------------------------------------------------   
    
    % Select data
    if file_number < 10
        f = spm_select('FPList', fullfile(data_path2,['func\subject_000' num2str(file_number) '\']),'func*') ;
    elseif file_number < 100
        f = spm_select('FPList', fullfile(data_path2,['func\subject_00' num2str(file_number) '\']),'func*') ;
    end
    % Comment if preprocessed files needs to be calculated
    f = f(1:st,:);
    
    
    %% REALIGN
    %--------------------------------------------------------------------------
    pjobs{2}.spatial{1}.realign{1}.estwrite.data{1} = cellstr(f);
               
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% FINISH
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    save('rest_batch_preprocessing.mat','pjobs');
    
    error1 = 0;
    try
        spm_jobman('run',pjobs);        % run preprocessing
    catch err
        err
        error1 = 1;
    end
           
    file_number
    
    
    toc
    
end








