% for slice = 1:33
%     statistical_maps_no_flipped(:,:,slice) = flipud(statistical_maps_no(:,:,slice));
% end
% 
% new_file.hdr = EPI_nii.hdr;
% new_file.hdr.dime.dim = [3 64 64 33 1 1 1 1];
% new_file.hdr.dime.vox_offset = 352;
% new_file.hdr.scl_slope = 1;
% new_file.hdr.dime.cal_max = max(statistical_maps_no(:));
% new_file.hdr.dime.cal_min = min(statistical_maps_no(:));
% new_file.hdr.dime.gl_max = max(statistical_maps_no(:));
% new_file.hdr.dime.gl_min = min(statistical_maps_no(:));
% new_file.hdr.dime.datatype = 16;
% new_file.hdr.dime.bitpix = 32;
% 
% new_file.original.hdr.dime.dim = [3 64 64 33 1 1 1 1];
% new_file.original.hdr.dime.vox_offset = 352;
% new_file.original.hdr.scl_slope = 1;
% new_file.original.hdr.dime.cal_max = max(statistical_maps_no(:));
% new_file.original.hdr.dime.cal_min = min(statistical_maps_no(:));
% new_file.original.hdr.dime.gl_max = max(statistical_maps_no(:));
% new_file.original.hdr.dime.gl_min = min(statistical_maps_no(:));
% new_file.original.hdr.dime.datatype = 16;
% new_file.original.hdr.dime.bitpix = 32;
% 
% new_file.img = single(statistical_maps_no);    
% filename = ['BROCCOLI_statistical_map_no_whitening.nii'];
% save_nii(new_file,filename);

% 
% new_file.hdr = EPI_nii.hdr;
% new_file.hdr.dime.dim = [4 64 64 33 1 1 1 1];
% new_file.hdr.dime.vox_offset = 0;
% new_file.hdr.dime.cal_max = 50;
% new_file.hdr.dime.gl_max = 50;
% new_file.hdr.dime.datatype = 16;
% new_file.hdr.dime.bitpix = 16;
% new_file.img = single(statistical_maps);    
% filename = ['BROCCOLI_statistical_map_whitening.nii'];
% save_nii(new_file,filename);

load BROCCOLI_whitening
load BROCCOLI_nowhitening

fsl = load_nii( 'C:/Users/wande/Downloads/tstat1_whitening.nii.gz')
fsl = double(fsl.img);
fsl_no = load_nii( 'C:/Users/wande/Downloads/tstat1_nowhitening.nii.gz')
fsl_no = double(fsl_no.img);


fsl = load_nii( 'C:/Users/wande/Downloads/tstat3_whitening.nii.gz')
fsl = double(fsl.img);
fsl_no = load_nii( 'C:/Users/wande/Downloads/tstat3_nowhitening.nii.gz')
fsl_no = double(fsl_no.img);


fsl = load_nii( 'C:/Users/wande/Downloads/tstat6_whitening.nii.gz')
fsl = double(fsl.img);
fsl_no = load_nii( 'C:/Users/wande/Downloads/tstat6_nowhitening.nii.gz')
fsl_no = double(fsl_no.img);

fsl_beta_no = load_nii( 'C:/Users/wande/Downloads/beta6_nowhitening.nii.gz')
fsl_beta_no = double(fsl_beta_no.img);

close all

image([fMRI_volumes(:,:,slice,1)/100 fMRI_volumes(:,:,slice,1)/100    ]) ; colorbar
hold on
image([statistical_maps(:,:,slice) fsl(:,:,slice)    ]) ; colorbar
hold off

image([fMRI_volumes(:,:,slice,1)/100 fMRI_volumes(:,:,slice,1)/100    ]) ; colorbar
hold on
image([statistical_maps_no(:,:,slice) fsl_no(:,:,slice)    ]) ; colorbar
hold off


