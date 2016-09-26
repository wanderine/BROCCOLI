close all
clear all
clc

% Get current directory
BROCCOLI_DIRECTORY=pwd;
% Remove '\code\Matlab_Wrapper'
BROCCOLI_DIRECTORY=BROCCOLI_DIRECTORY(1:end-20);

if ispc
    
    BROCCOLI_LIBRARY_DIRECTORY=[BROCCOLI_DIRECTORY '\compiled\BROCCOLI_LIB\Windows\Release'];
    BROCCOLI_HEADER_DIRECTORY=[BROCCOLI_DIRECTORY '\code\BROCCOLI_LIB'];
    
    OPENCL_INCLUDE_DIRECTORY1='C:/Program'' Files''/NVIDIA'' GPU Computing Toolkit''/CUDA/v5.0/include';
    OPENCL_INCLUDE_DIRECTORY2='C:/Program'' Files''/NVIDIA'' GPU Computing Toolkit''/CUDA/v5.0/include/CL';
    OPENCL_LIBRARY_DIRECTORY='C:/Program'' Files''/NVIDIA'' GPU Computing Toolkit''/CUDA/v5.0/lib/x64';
    
    EIGEN_DIRECTORY=[BROCCOLI_DIRECTORY '\code\BROCCOLI_LIB\Eigen'];
    CLBLAS_DIRECTORY=[BROCCOLI_DIRECTORY '\code\BROCCOLI_LIB\clBLASWindows'];
    
    error = 0;
    
    try
        disp('Compiling GetOpenCLInfo.cpp')
        cmd = sprintf('mex GetOpenCLInfo.cpp -lBROCCOLI_LIB -lOpenCL -I%s -I%s -L%s -L%s -I%s -I%s -L%s',OPENCL_INCLUDE_DIRECTORY1, OPENCL_INCLUDE_DIRECTORY2, OPENCL_LIBRARY_DIRECTORY, BROCCOLI_LIBRARY_DIRECTORY, BROCCOLI_HEADER_DIRECTORY, EIGEN_DIRECTORY);
        eval(cmd)
    catch
        error = 1;
        disp('Failed to compile GetOpenCLInfo.cpp')
    end
    
    try
        disp('Compiling SmoothingMex.cpp')
        cmd = sprintf('mex SmoothingMex.cpp -lBROCCOLI_LIB -lOpenCL -I%s -I%s -L%s -L%s -I%s -I%s',OPENCL_INCLUDE_DIRECTORY1, OPENCL_INCLUDE_DIRECTORY2, OPENCL_LIBRARY_DIRECTORY, BROCCOLI_LIBRARY_DIRECTORY, BROCCOLI_HEADER_DIRECTORY, EIGEN_DIRECTORY);
        eval(cmd)
    catch
        error = 1;
        disp('Failed to compile SmoothingMex.cpp')
    end
    
    try
        disp('Compiling MotionCorrectionMex.cpp')
        cmd = sprintf('mex MotionCorrectionMex.cpp -lBROCCOLI_LIB -lOpenCL -I%s -I%s -L%s -L%s -I%s -I%s',OPENCL_INCLUDE_DIRECTORY1, OPENCL_INCLUDE_DIRECTORY2, OPENCL_LIBRARY_DIRECTORY, BROCCOLI_LIBRARY_DIRECTORY, BROCCOLI_HEADER_DIRECTORY, EIGEN_DIRECTORY);
        eval(cmd)
    catch
        error = 1;
        disp('Failed to compile MotionCorrectionMex.cpp')
    end
    
    try
        disp('Compiling RegisterTwoVolumesMex.cpp')
        cmd = sprintf('mex RegisterTwoVolumesMex.cpp -lBROCCOLI_LIB -lOpenCL -I%s -I%s -L%s -L%s -I%s -I%s',OPENCL_INCLUDE_DIRECTORY1, OPENCL_INCLUDE_DIRECTORY2, OPENCL_LIBRARY_DIRECTORY, BROCCOLI_LIBRARY_DIRECTORY, BROCCOLI_HEADER_DIRECTORY, EIGEN_DIRECTORY);
        eval(cmd)
    catch
        error = 1;
        disp('Failed to compile RegisterTwoVolumesMex.cpp')
    end
    %
    %     try
    %         disp('Compiling SliceTimingCorrectionMex.cpp')
    %         cmd = sprintf('mex SliceTimingCorrectionMex.cpp -lOpenCL -lBROCCOLI_LIB -I%s -I%s -L%s -L%s -I%s -I%s  -I%s -I%s',OPENCL_INCLUDE_DIRECTORY1, OPENCL_INCLUDE_DIRECTORY2, OPENCL_LIBRARY_DIRECTORY, BROCCOLI_LIBRARY_DIRECTORY, BROCCOLI_HEADER_DIRECTORY, NIFTI_DIRECTORY, ZNZ_DIRECTORY, EIGEN_DIRECTORY);
    %         eval(cmd)
    %     catch
    %         error = 1;
    %         disp('Failed to compile SliceTimingCorrectionMex.cpp')
    %     end
    %
    %     try
    %         %disp('Compiling RandomiseGroupLevelMex.cpp')
    %         cmd = sprintf('mex GLMTTest_SecondLevel_Permutation.cpp -lOpenCL -lBROCCOLI_LIB -I%s -I%s -L%s -L%s -I%s -I%s  -I%s -I%s',OPENCL_INCLUDE_DIRECTORY1, OPENCL_INCLUDE_DIRECTORY2, OPENCL_LIBRARY_DIRECTORY, BROCCOLI_LIBRARY_DIRECTORY, BROCCOLI_HEADER_DIRECTORY, NIFTI_DIRECTORY, ZNZ_DIRECTORY, EIGEN_DIRECTORY);
    %         eval(cmd)
    %     catch
    %         error = 1;
    %     end
    
    %disp('Compiling FirstLevelAnalysisMex.cpp')
    %cmd = sprintf('mex FirstLevelAnalysisMex.cpp -lOpenCL -lBROCCOLI_LIB -I%s -I%s -L%s -L%s -I%s -I%s  -I%s -I%s',OPENCL_INCLUDE_DIRECTORY1, OPENCL_INCLUDE_DIRECTORY2, OPENCL_LIBRARY_DIRECTORY, BROCCOLI_LIBRARY_DIRECTORY, BROCCOLI_HEADER_DIRECTORY, NIFTI_DIRECTORY, ZNZ_DIRECTORY, EIGEN_DIRECTORY);
    %eval(cmd)
    
    if error == 0
        disp('Successfully compiled all wrappers!')
    end
    
elseif isunix
    
    BROCCOLI_LIBRARY_DIRECTORY=[BROCCOLI_DIRECTORY '/compiled/BROCCOLI_LIB/Linux/Release'];
    BROCCOLI_HEADER_DIRECTORY=[BROCCOLI_DIRECTORY '/code/BROCCOLI_LIB'];
    
    OPENCL_INCLUDE_DIRECTORY1='/opt/intel/opencl-sdk/include';
    OPENCL_INCLUDE_DIRECTORY2='/opt/intel/opencl-sdk/include/CL';
    OPENCL_LIBRARY_DIRECTORY='/opt/intel/opencl/lib64';
    
    EIGEN_DIRECTORY=[BROCCOLI_DIRECTORY '/code/BROCCOLI_LIB/Eigen'];
    CLBLAS_DIRECTORY=[BROCCOLI_DIRECTORY '/code/BROCCOLI_LIB/clBLASLinux'];
    
    error = 0;
    
    try
        disp('Compiling GetOpenCLInfo.cpp')
        cmd = sprintf('mex GetOpenCLInfo.cpp -lBROCCOLI_LIB -lOpenCL -lclBLAS -lgomp -I%s -I%s -L%s -L%s -I%s -I%s -L%s',OPENCL_INCLUDE_DIRECTORY1, OPENCL_INCLUDE_DIRECTORY2, OPENCL_LIBRARY_DIRECTORY, BROCCOLI_LIBRARY_DIRECTORY, BROCCOLI_HEADER_DIRECTORY, EIGEN_DIRECTORY, CLBLAS_DIRECTORY);
        eval(cmd)
    catch
        error = 1;
        disp('Failed to compile GetOpenCLInfo.cpp')
    end
    
    try
        disp('Compiling SmoothingMex.cpp')
        cmd = sprintf('mex SmoothingMex.cpp -lBROCCOLI_LIB -lOpenCL -lclBLAS -lgomp -I%s -I%s -L%s -L%s -I%s -I%s -L%s',OPENCL_INCLUDE_DIRECTORY1, OPENCL_INCLUDE_DIRECTORY2, OPENCL_LIBRARY_DIRECTORY, BROCCOLI_LIBRARY_DIRECTORY, BROCCOLI_HEADER_DIRECTORY, EIGEN_DIRECTORY, CLBLAS_DIRECTORY);
        eval(cmd)
    catch
        error = 1;
        disp('Failed to compile SmoothingMex.cpp')
    end
    
    try
        disp('Compiling MotionCorrectionMex.cpp')
        cmd = sprintf('mex MotionCorrectionMex.cpp -lBROCCOLI_LIB -lOpenCL -lclBLAS -lgomp -I%s -I%s -L%s -L%s -I%s -I%s -L%s',OPENCL_INCLUDE_DIRECTORY1, OPENCL_INCLUDE_DIRECTORY2, OPENCL_LIBRARY_DIRECTORY, BROCCOLI_LIBRARY_DIRECTORY, BROCCOLI_HEADER_DIRECTORY, EIGEN_DIRECTORY, CLBLAS_DIRECTORY);
        eval(cmd)
    catch
        error = 1;
        disp('Failed to compile MotionCorrectionMex.cpp')
    end
    
    try
        disp('Compiling RegisterTwoVolumesMex.cpp')
        cmd = sprintf('mex RegisterTwoVolumesMex.cpp -lBROCCOLI_LIB -lOpenCL -lclBLAS -lgomp -I%s -I%s -L%s -L%s -I%s -I%s -L%s',OPENCL_INCLUDE_DIRECTORY1, OPENCL_INCLUDE_DIRECTORY2, OPENCL_LIBRARY_DIRECTORY, BROCCOLI_LIBRARY_DIRECTORY, BROCCOLI_HEADER_DIRECTORY, EIGEN_DIRECTORY, CLBLAS_DIRECTORY);
        eval(cmd)
    catch
        error = 1;
        disp('Failed to compile RegisterTwoVolumesMex.cpp')
    end
    
    try
        disp('Compiling RandomiseGroupLevelMex.cpp')
        cmd = sprintf('mex RandomiseGroupLevelMex.cpp -lBROCCOLI_LIB -lOpenCL -lclBLAS -lgomp -I%s -I%s -L%s -L%s -I%s -I%s -L%s',OPENCL_INCLUDE_DIRECTORY1, OPENCL_INCLUDE_DIRECTORY2, OPENCL_LIBRARY_DIRECTORY, BROCCOLI_LIBRARY_DIRECTORY, BROCCOLI_HEADER_DIRECTORY, EIGEN_DIRECTORY, CLBLAS_DIRECTORY);
        eval(cmd)
    catch
        error = 1;
        disp('Failed to compile RandomiseGroupLevelMex.cpp')
    end
    
    %
    %     try
    %         disp('Compiling SliceTimingCorrectionMex.cpp')
    %         cmd = sprintf('mex SliceTimingCorrectionMex.cpp -lOpenCL -lBROCCOLI_LIB -I%s -I%s -L%s -L%s -I%s -I%s  -I%s -I%s',OPENCL_INCLUDE_DIRECTORY1, OPENCL_INCLUDE_DIRECTORY2, OPENCL_LIBRARY_DIRECTORY, BROCCOLI_LIBRARY_DIRECTORY, BROCCOLI_HEADER_DIRECTORY, NIFTI_DIRECTORY, ZNZ_DIRECTORY, EIGEN_DIRECTORY);
    %         eval(cmd)
    %     catch
    %         error = 1;
    %         disp('Failed to compile SliceTimingCorrectionMex.cpp')
    %     end
    %
    
    
    %disp('Compiling FirstLevelAnalysisMex.cpp')
    %cmd = sprintf('mex FirstLevelAnalysisMex.cpp -lOpenCL -lBROCCOLI_LIB -I%s -I%s -L%s -L%s -I%s -I%s  -I%s -I%s',OPENCL_INCLUDE_DIRECTORY1, OPENCL_INCLUDE_DIRECTORY2, OPENCL_LIBRARY_DIRECTORY, BROCCOLI_LIBRARY_DIRECTORY, BROCCOLI_HEADER_DIRECTORY, NIFTI_DIRECTORY, ZNZ_DIRECTORY, EIGEN_DIRECTORY);
    %eval(cmd)
    
    if error == 0
        disp('Successfully compiled all wrappers!')
    end
    
elseif ismac
    
end





