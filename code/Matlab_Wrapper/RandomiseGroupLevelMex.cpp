/*
 * BROCCOLI: Software for fast fMRI analysis on many core CPUs and GPUs
 * Copyright (C) <2013>  Anders Eklund, andek034@gmail.com
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

#include "mex.h"
#include "help_functions.cpp"
#include "broccoli_lib.h"

double factorial(unsigned long int n)
{
    // Stirling approximation
    return (n == 1 || n == 0) ? 1.0 : round( sqrt(2.0*3.14*(double)n) * pow( ((double)n / 2.7183), double(n) ) );
}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    //-----------------------
    // Input 
    
    double		    *h_First_Level_Results_double, *h_Mask_double;
    float           *h_First_Level_Results, *h_Mask; 
    
    double		    *h_X_GLM_double, *h_xtxxt_GLM_double, *h_Contrasts_double, *h_ctxtxc_GLM_double;
    float           *h_X_GLM, *h_xtxxt_GLM, *h_Contrasts, *h_ctxtxc_GLM;  
            
    int             DATA_W, DATA_H, DATA_D, NUMBER_OF_SUBJECTS, NUMBER_OF_PERMUTATIONS; 
                    
	int				ANALYSIS_TYPE;
    int             NUMBER_OF_GLM_REGRESSORS, NUMBER_OF_CONTRASTS, INFERENCE_MODE; 
    float           CLUSTER_DEFINING_THRESHOLD;
    
    int             OPENCL_PLATFORM, OPENCL_DEVICE;
    
    int             NUMBER_OF_DIMENSIONS;
	        
	const char* 	BROCCOLI_LOCATION;

	size_t			NUMBER_OF_PERMUTATIONS_PER_CONTRAST[1000];
	bool 			CORRELATION_DESIGN[1000];
	bool 			TWOSAMPLE_DESIGN[1000];
	bool 			MEAN_DESIGN[1000];
	int  			GROUP_DESIGNS[1000];
	bool 			DO_ALL_PERMUTATIONS = false;
	int	 			NUMBER_OF_STATISTICAL_MAPS = 1;
	float 			SIGNIFICANCE_LEVEL = 0.05f;

	for (int i = 0; i < 1000; i++)
	{
		TWOSAMPLE_DESIGN[i] = false;
		CORRELATION_DESIGN[i] = false;
		MEAN_DESIGN[i] = false;
		GROUP_DESIGNS[i] = 100;
		NUMBER_OF_PERMUTATIONS_PER_CONTRAST[i] = 5000;
	}

	bool ANALYZE_GROUP_MEAN = false;
	bool ANALYZE_TTEST = false;
	bool ANALYZE_FTEST = false;

	int NUMBER_OF_SUBJECTS_IN_GROUP1[1000], NUMBER_OF_SUBJECTS_IN_GROUP2[1000];

	unsigned short int        **h_Permutation_Matrices, *h_Permutation_Matrix;
	float					   *h_Sign_Matrix;

    //-----------------------
    // Output
    
    double     		*h_Statistical_Maps_double, *h_P_Values_double;    
    float           *h_Statistical_Maps, *h_P_Values, *h_Permutation_Distribution, **h_Permutation_Distributions;        
    
    //---------------------
    
    /* Check the number of input and output arguments. */
    if(nrhs<13)
    {
        mexErrMsgTxt("Too few input arguments.");
    }
    if(nrhs>13)
    {
        mexErrMsgTxt("Too many input arguments.");
    }
    if(nlhs<2)
    {
        mexErrMsgTxt("Too few output arguments.");
    }
    if(nlhs>2)
    {
        mexErrMsgTxt("Too many output arguments.");
    }
    
    /* Input arguments */
    
    // The data
    h_First_Level_Results_double =  (double*)mxGetData(prhs[0]);
    h_Mask_double = (double*)mxGetData(prhs[1]);
          
	ANALYSIS_TYPE = (int)mxGetScalar(prhs[2]);        

    h_X_GLM_double =  (double*)mxGetData(prhs[3]);    
    h_xtxxt_GLM_double =  (double*)mxGetData(prhs[4]);
    h_Contrasts_double = (double*)mxGetData(prhs[5]);
    h_ctxtxc_GLM_double = (double*)mxGetData(prhs[6]);    
    
    NUMBER_OF_PERMUTATIONS = (int)mxGetScalar(prhs[7]);        
    INFERENCE_MODE = (int)mxGetScalar(prhs[8]);        
    CLUSTER_DEFINING_THRESHOLD = (float)mxGetScalar(prhs[9]);    
    
    OPENCL_PLATFORM  = (int)mxGetScalar(prhs[10]);
    OPENCL_DEVICE = (int)mxGetScalar(prhs[11]);

	BROCCOLI_LOCATION  = mxArrayToString(prhs[12]);
    
	// t-test
	if (ANALYSIS_TYPE == 0)
	{
		ANALYZE_TTEST = true;
	}
	// F-test
	else if (ANALYSIS_TYPE == 1)
	{
		ANALYZE_FTEST = true;
	}
	// Group mean
	else if (ANALYSIS_TYPE == 2)
	{
		ANALYZE_GROUP_MEAN = true;
	}

    const int *ARRAY_DIMENSIONS_FIRST_LEVEL_RESULTS = mxGetDimensions(prhs[0]);
    const int *ARRAY_DIMENSIONS_MNI = mxGetDimensions(prhs[1]);
    const int *ARRAY_DIMENSIONS_GLM = mxGetDimensions(prhs[3]);    
    const int *ARRAY_DIMENSIONS_CONTRAST = mxGetDimensions(prhs[5]);        
      
    NUMBER_OF_SUBJECTS = ARRAY_DIMENSIONS_FIRST_LEVEL_RESULTS[3];
    
    DATA_H = ARRAY_DIMENSIONS_MNI[0];
    DATA_W = ARRAY_DIMENSIONS_MNI[1];
    DATA_D = ARRAY_DIMENSIONS_MNI[2];
            
    NUMBER_OF_GLM_REGRESSORS = ARRAY_DIMENSIONS_GLM[1];
    NUMBER_OF_CONTRASTS = ARRAY_DIMENSIONS_CONTRAST[1];
    
    int FIRST_LEVEL_RESULTS_DATA_SIZE = DATA_W * DATA_H * DATA_D * NUMBER_OF_SUBJECTS * sizeof(float);
    int VOLUME_SIZE = DATA_W * DATA_H * DATA_D * sizeof(float);
        
    int GLM_SIZE = NUMBER_OF_SUBJECTS * NUMBER_OF_GLM_REGRESSORS * sizeof(float);
    int CONTRAST_SIZE = NUMBER_OF_GLM_REGRESSORS * NUMBER_OF_CONTRASTS * sizeof(float);
    int CONTRAST_MATRIX_SIZE = NUMBER_OF_CONTRASTS * NUMBER_OF_CONTRASTS * sizeof(float);
    int DESIGN_MATRIX_SIZE = NUMBER_OF_GLM_REGRESSORS * NUMBER_OF_SUBJECTS * sizeof(float);
            
    int STATISTICAL_MAPS_DATA_SIZE = DATA_W * DATA_H * DATA_D * NUMBER_OF_CONTRASTS * sizeof(float);    
       
    int NULL_DISTRIBUTION_SIZE = NUMBER_OF_PERMUTATIONS * sizeof(float);
    
	size_t			allocatedHostMemory = 0;

    mexPrintf("Data size : %i x %i x %i x %i \n", DATA_W, DATA_H, DATA_D, NUMBER_OF_SUBJECTS);
    mexPrintf("Number of GLM regressors : %i \n",  NUMBER_OF_GLM_REGRESSORS);
    mexPrintf("Number of contrasts : %i \n",  NUMBER_OF_CONTRASTS);
    mexPrintf("Number of permutations : %i \n",  NUMBER_OF_PERMUTATIONS);
    
    //-------------------------------------------------
    // Output to Matlab
    
    // Create pointer for volumes to Matlab        

    NUMBER_OF_DIMENSIONS = 4;
    int ARRAY_DIMENSIONS_OUT_STATISTICAL_MAPS[4];
    ARRAY_DIMENSIONS_OUT_STATISTICAL_MAPS[0] = DATA_H;
    ARRAY_DIMENSIONS_OUT_STATISTICAL_MAPS[1] = DATA_W;
    ARRAY_DIMENSIONS_OUT_STATISTICAL_MAPS[2] = DATA_D;            
    ARRAY_DIMENSIONS_OUT_STATISTICAL_MAPS[3] = NUMBER_OF_CONTRASTS;            
    
    plhs[0] = mxCreateNumericArray(NUMBER_OF_DIMENSIONS,ARRAY_DIMENSIONS_OUT_STATISTICAL_MAPS,mxDOUBLE_CLASS, mxREAL);
    h_Statistical_Maps_double = mxGetPr(plhs[0]);
    
    plhs[1] = mxCreateNumericArray(NUMBER_OF_DIMENSIONS,ARRAY_DIMENSIONS_OUT_STATISTICAL_MAPS,mxDOUBLE_CLASS, mxREAL);
    h_P_Values_double = mxGetPr(plhs[1]);                                               
    
    // ------------------------------------------------
    
    // Allocate memory on the host
    h_First_Level_Results               = (float *)mxMalloc(FIRST_LEVEL_RESULTS_DATA_SIZE);
    
    h_Mask                    			= (float *)mxMalloc(VOLUME_SIZE);
                
    h_X_GLM                        		= (float *)mxMalloc(GLM_SIZE);
    h_xtxxt_GLM                    		= (float *)mxMalloc(GLM_SIZE);
    h_Contrasts                    		= (float *)mxMalloc(CONTRAST_SIZE);
    h_ctxtxc_GLM                  		= (float *)mxMalloc(CONTRAST_MATRIX_SIZE);
         
    h_Statistical_Maps                  = (float *)mxMalloc(STATISTICAL_MAPS_DATA_SIZE);
    h_P_Values                          = (float *)mxMalloc(STATISTICAL_MAPS_DATA_SIZE);

	h_Permutation_Distributions = (float**)mxMalloc(NUMBER_OF_CONTRASTS * sizeof(float*));
	h_Permutation_Matrices = (unsigned short int**)mxMalloc(NUMBER_OF_CONTRASTS * sizeof(unsigned short int*));
    
    // Reorder and cast data
    pack_double2float_volumes(h_First_Level_Results, h_First_Level_Results_double, DATA_W, DATA_H, DATA_D, NUMBER_OF_SUBJECTS);
    pack_double2float_volume(h_Mask, h_Mask_double, DATA_W, DATA_H, DATA_D);
    
    pack_double2float(h_X_GLM, h_X_GLM_double, NUMBER_OF_GLM_REGRESSORS * NUMBER_OF_SUBJECTS);
    pack_double2float(h_xtxxt_GLM, h_xtxxt_GLM_double, NUMBER_OF_GLM_REGRESSORS * NUMBER_OF_SUBJECTS);    
    pack_double2float(h_Contrasts, h_Contrasts_double, NUMBER_OF_GLM_REGRESSORS * NUMBER_OF_CONTRASTS);    
    pack_double2float(h_ctxtxc_GLM, h_ctxtxc_GLM_double, NUMBER_OF_CONTRASTS);       
    
	// ------------------------------------------------
	// Check if design matrix is two sample test or correlation, for each contrast
    // ------------------------------------------------

	Eigen::MatrixXd X(NUMBER_OF_SUBJECTS,NUMBER_OF_GLM_REGRESSORS);
	
	if (!ANALYZE_GROUP_MEAN)
    {
		// Put design matrix into Eigen object 
		for (size_t s = 0; s < NUMBER_OF_SUBJECTS; s++)
		{
			for (size_t r = 0; r < NUMBER_OF_GLM_REGRESSORS; r++)
			{
				X(s,r) = (double)h_X_GLM[s + r * NUMBER_OF_SUBJECTS];
			}
		}
	}
	else if (ANALYZE_GROUP_MEAN)
	{
		// Create regressor with all ones
		for (size_t s = 0; s < NUMBER_OF_SUBJECTS; s++)
		{
			X(s) = 1.0;
		}
	}

	Eigen::MatrixXd Contrasts(NUMBER_OF_CONTRASTS,NUMBER_OF_GLM_REGRESSORS);

	// Put contrast vectors into eigen object
	for (size_t c = 0; c < NUMBER_OF_CONTRASTS; c++)
	{
		for (size_t r = 0; r < NUMBER_OF_GLM_REGRESSORS; r++)
		{
			Contrasts(c,r) = h_Contrasts[r + c * NUMBER_OF_GLM_REGRESSORS];		
		}
	}

	if (ANALYZE_FTEST)
	{
		// Check if contrast matrix has full rank
		Eigen::FullPivLU<Eigen::MatrixXd> luA(Contrasts);
		int rank = luA.rank();
		if (rank < NUMBER_OF_CONTRASTS)
		{
	        mexErrMsgTxt("Contrast matrix does not have full rank, at least one contrast can be written as a linear combination of other contrasts, not OK for F-test, aborting!\n");      	
		}
	}

	if (ANALYZE_TTEST)
	{    
		// Calculate current contrast
		for (size_t c = 0; c < NUMBER_OF_CONTRASTS; c++)
		{
			Eigen::MatrixXd currentContrast = Contrasts.row(c);
			Eigen::MatrixXd currentVector = X * currentContrast.transpose();		

			mexPrintf("Current contrast is ");
			for (int r = 0; r < NUMBER_OF_GLM_REGRESSORS; r++)
			{
				mexPrintf(" %f ",currentContrast(r));
			}
			mexPrintf("\n");

			// Check how many unique values there are
			std::vector<float> allUniqueValues;        

	        for (size_t s = 0; s < NUMBER_OF_SUBJECTS; s++)
	        {
				float value = currentVector(s);
				bool unique = true;
				// Check if value is in list of unique values
				for (size_t i = 0; i < allUniqueValues.size(); i++)
				{
					if (value == allUniqueValues[i])
					{
						unique = false;
						break;
					}
				}

				if (unique)
				{
					allUniqueValues.push_back(value);
				}
	 		}
    
			if ( allUniqueValues.size() == 2 )
			{
				TWOSAMPLE_DESIGN[c] = true;
				GROUP_DESIGNS[c] = 0;
			}
			else if ( allUniqueValues.size() > 2 )
			{
				CORRELATION_DESIGN[c] = true;
				GROUP_DESIGNS[c] = 1;
			}
			else if ( allUniqueValues.size() == 1 )
			{
				MEAN_DESIGN[c] = true;
				GROUP_DESIGNS[c] = 2;
			}

			if (TWOSAMPLE_DESIGN[c])
			{
				NUMBER_OF_SUBJECTS_IN_GROUP1[c] = 0;
				for (size_t s = 0; s < NUMBER_OF_SUBJECTS; s++)
				{
					if (currentVector(s) == allUniqueValues[0])
					{
						NUMBER_OF_SUBJECTS_IN_GROUP1[c]++;
					}
				}
				NUMBER_OF_SUBJECTS_IN_GROUP2[c] = NUMBER_OF_SUBJECTS - NUMBER_OF_SUBJECTS_IN_GROUP1[c];
		        mexPrintf("Two sample t-test design detected for t-contrast %zu, %i subjects in group 1 and %i subjects in group 2\n",c+1,NUMBER_OF_SUBJECTS_IN_GROUP1[c],NUMBER_OF_SUBJECTS_IN_GROUP2[c]);
			}
			else if (CORRELATION_DESIGN[c])
			{
			   mexPrintf("Correlation design detected for t-contrast %zu\n",c+1);
			}	
			else if (MEAN_DESIGN[c])
			{
			   mexPrintf("Mean design detected for t-contrast %zu\n",c+1);
			}	

		}
	}
	else if (ANALYZE_FTEST)
	{
		// Calculate design matrix * contrast
		Eigen::MatrixXd temp = X * Contrasts.transpose();
		
		// Check how many unique values there are
		std::vector<float> allUniqueValues;        

        for (size_t s = 0; s < NUMBER_OF_SUBJECTS; s++)
        {
	        for (size_t c = 0; c < NUMBER_OF_CONTRASTS; c++)
    	    {
				float value = temp(s,c);
				bool unique = true;
				// Check if value is in list of unique values
				for (size_t i = 0; i < allUniqueValues.size(); i++)
				{
					if (value == allUniqueValues[i])
					{
						unique = false;
						break;
					}
				}

				if (unique)
				{
					allUniqueValues.push_back(value);
				}
	 		}
		}
    
		if ( allUniqueValues.size() == 2 )
		{
			TWOSAMPLE_DESIGN[0] = true;
			GROUP_DESIGNS[0] = 0;
		}
		else if ( allUniqueValues.size() > 2 )
		{
			CORRELATION_DESIGN[0] = true;
			GROUP_DESIGNS[0] = 1;
		}
		else if ( allUniqueValues.size() == 1 )
		{
			MEAN_DESIGN[0] = true;
			GROUP_DESIGNS[0] = 2;
		}
		

		if (TWOSAMPLE_DESIGN[0])
		{
			NUMBER_OF_SUBJECTS_IN_GROUP1[0] = 0;
			for (size_t s = 0; s < NUMBER_OF_SUBJECTS; s++)
			{
				if (temp(s,0) == allUniqueValues[0])
				{
					NUMBER_OF_SUBJECTS_IN_GROUP1[0]++;
				}
			}
			NUMBER_OF_SUBJECTS_IN_GROUP2[0] = NUMBER_OF_SUBJECTS - NUMBER_OF_SUBJECTS_IN_GROUP1[0];
	        mexPrintf("\nTwo sample t-test design detected for F-test, %i subjects in group 1 and %i subjects in group 2\n\n",NUMBER_OF_SUBJECTS_IN_GROUP1[0],NUMBER_OF_SUBJECTS_IN_GROUP2[0]);
		}
		else if (CORRELATION_DESIGN[0])
		{
		   mexPrintf("\nCorrelation design detected for F-test\n\n");
		}
	}


	// Check if requested number of permutations is larger than number of possible sign flips
	if (ANALYZE_GROUP_MEAN)
	{
		NUMBER_OF_PERMUTATIONS_PER_CONTRAST[0] = NUMBER_OF_PERMUTATIONS;

		// Calculate maximum number of sign flips
		double MAX_SIGN_FLIPS = pow(2.0, (double)NUMBER_OF_SUBJECTS);
		if ((double)NUMBER_OF_PERMUTATIONS > MAX_SIGN_FLIPS)
		{
			mexPrintf("Warning: Number of possible sign flips for group mean is %g, but %g permutations were requested. Lowering number of permutations to number of possible sign flips. \n",MAX_SIGN_FLIPS,(double)NUMBER_OF_PERMUTATIONS);
			NUMBER_OF_PERMUTATIONS_PER_CONTRAST[0] = (int)MAX_SIGN_FLIPS;
			DO_ALL_PERMUTATIONS = true;
		}
		else if ((double)NUMBER_OF_PERMUTATIONS == MAX_SIGN_FLIPS)
		{
			DO_ALL_PERMUTATIONS = true;
			mexPrintf("Max number of sign flips is %g \n",MAX_SIGN_FLIPS);
		}
		else
		{
			mexPrintf("Max number of sign flips is %g \n",MAX_SIGN_FLIPS);
		}
	}

	if (ANALYZE_GROUP_MEAN)
	{
		NUMBER_OF_STATISTICAL_MAPS = 1;
	}
	if (ANALYZE_TTEST)
	{
		NUMBER_OF_STATISTICAL_MAPS = NUMBER_OF_CONTRASTS;
	}
	else if (ANALYZE_FTEST)
	{
		NUMBER_OF_STATISTICAL_MAPS = 1;
	}

	if (!ANALYZE_GROUP_MEAN)
	{
		for (size_t c = 0; c < NUMBER_OF_STATISTICAL_MAPS; c++)
		{
			NUMBER_OF_PERMUTATIONS_PER_CONTRAST[c] = NUMBER_OF_PERMUTATIONS;

			if (TWOSAMPLE_DESIGN[c])
			{
		       	double MAX_PERMS = round(exp(lgamma(NUMBER_OF_SUBJECTS+1)-lgamma(NUMBER_OF_SUBJECTS-NUMBER_OF_SUBJECTS_IN_GROUP2[c]+1)-lgamma(NUMBER_OF_SUBJECTS_IN_GROUP2[c]+1)));
				if ((double)NUMBER_OF_PERMUTATIONS > MAX_PERMS)
				{
					mexPrintf("Warning: Number of possible permutations for your design is %g for contrast %zu, but %g permutations were requested. Lowering number of permutations to number of possible permutations. \n",MAX_PERMS,c+1,(double)NUMBER_OF_PERMUTATIONS);
					NUMBER_OF_PERMUTATIONS_PER_CONTRAST[c] = (int)MAX_PERMS; 
					DO_ALL_PERMUTATIONS = true;
				}
				else if ((double)NUMBER_OF_PERMUTATIONS == MAX_PERMS)
				{
					DO_ALL_PERMUTATIONS = true;
					mexPrintf("Max number of permutations for contrast %zu is %g \n",c+1,MAX_PERMS);
				}
				else
				{
					mexPrintf("Max number of permutations for contrast %zu is %g \n",c+1,MAX_PERMS);
				}
			}
			else if (CORRELATION_DESIGN[c])
			{
				double MAX_PERMS = factorial(NUMBER_OF_SUBJECTS);
				if ((double)NUMBER_OF_PERMUTATIONS > MAX_PERMS)
				{
					mexPrintf("Warning: Number of possible permutations for your design is %g for contrast %zu, but %g permutations were requested. Lowering number of permutations to number of possible permutations. \n",MAX_PERMS,c+1,(double)NUMBER_OF_PERMUTATIONS);
					NUMBER_OF_PERMUTATIONS_PER_CONTRAST[c] = (int)MAX_PERMS; 
					DO_ALL_PERMUTATIONS = true;
				}
				else if ((double)NUMBER_OF_PERMUTATIONS == MAX_PERMS)
				{
					DO_ALL_PERMUTATIONS = true;
					mexPrintf("Max number of permutations for contrast %zu is %g \n",c+1,MAX_PERMS);
				}
				else
				{
					mexPrintf("Max number of permutations for contrast %zu is %g \n",c+1,MAX_PERMS);
				}
			}
			else if (MEAN_DESIGN[c])
			{
				mexPrintf("Warning: Contrast %zu leads to a simple mean value, use group mean option instead!\n",c+1);
				NUMBER_OF_PERMUTATIONS_PER_CONTRAST[c] = 1; 
			}
		}
	}

    // ------------------------------------------------
	
	size_t SIGN_MATRIX_SIZE = NUMBER_OF_PERMUTATIONS_PER_CONTRAST[0] * NUMBER_OF_SUBJECTS * sizeof(float);

	h_Sign_Matrix = (float *)mxMalloc(SIGN_MATRIX_SIZE);

	for (size_t c = 0; c < NUMBER_OF_STATISTICAL_MAPS; c++)
	{ 
		size_t PERMUTATION_MATRIX_SIZE = NUMBER_OF_PERMUTATIONS_PER_CONTRAST[c] * NUMBER_OF_SUBJECTS * sizeof(unsigned short int);
	    size_t NULL_DISTRIBUTION_SIZE = NUMBER_OF_PERMUTATIONS_PER_CONTRAST[c] * sizeof(float);

 	    h_Permutation_Matrix  = (unsigned short int *)mxMalloc(PERMUTATION_MATRIX_SIZE);
 	    h_Permutation_Distribution  = (float *)mxMalloc(NULL_DISTRIBUTION_SIZE);

		h_Permutation_Matrices[c] = h_Permutation_Matrix;
		h_Permutation_Distributions[c] = h_Permutation_Distribution;
	}

    // ------------------------------------------------
    
    BROCCOLI_LIB BROCCOLI(OPENCL_PLATFORM,OPENCL_DEVICE,BROCCOLI_LOCATION); 
        
    // Print build info to file (always)
	std::vector<std::string> buildInfo = BROCCOLI.GetOpenCLBuildInfo();
	std::vector<std::string> kernelFileNames = BROCCOLI.GetKernelFileNames();

	std::string buildInfoPath;
	buildInfoPath.append(BROCCOLI_LOCATION);
	buildInfoPath.append("compiled/Kernels/");
    
    FILE *fp = NULL;
    
    for (int k = 0; k < BROCCOLI.GetNumberOfKernelFiles(); k++)
	{
		std::string temp = buildInfoPath;
		temp.append("buildInfo_");
		temp.append(BROCCOLI.GetOpenCLPlatformName());
		temp.append("_");	
		temp.append(BROCCOLI.GetOpenCLDeviceName());
		temp.append("_");	
		std::string name = kernelFileNames[k];
		// Remove "kernel" and ".cpp" from kernel filename
		name = name.substr(0,name.size()-4);
		name = name.substr(6,name.size());
		temp.append(name);
		temp.append(".txt");
		fp = fopen(temp.c_str(),"w");
		if (fp == NULL)
		{     
		    mexPrintf("Could not open %s for writing ! \n",temp.c_str());
		}
		else
		{	
			if (buildInfo[k].c_str() != NULL)
			{
			    int error = fputs(buildInfo[k].c_str(),fp);
			    if (error == EOF)
			    {
			        mexPrintf("Could not write to %s ! \n",temp.c_str());
			    }
			}
			fclose(fp);
		}
	}
                
    // Something went wrong...
    if (!BROCCOLI.GetOpenCLInitiated())
    { 
        mexPrintf("Initialization error is \"%s\" \n",BROCCOLI.GetOpenCLInitializationError().c_str());
		mexPrintf("OpenCL error is \"%s\" \n",BROCCOLI.GetOpenCLError());

        // Print create kernel errors
        int* createKernelErrors = BROCCOLI.GetOpenCLCreateKernelErrors();
        for (int i = 0; i < BROCCOLI.GetNumberOfOpenCLKernels(); i++)
        {
            if (createKernelErrors[i] != 0)
            {
                mexPrintf("Create kernel error for kernel '%s' is '%s' \n",BROCCOLI.GetOpenCLKernelName(i),BROCCOLI.GetOpenCLErrorMessage(createKernelErrors[i]));
            }
        }                        
                
        mexPrintf("OpenCL initialization failed, aborting! \nSee buildInfo* for output of OpenCL compilation!\n");         
    }
    else 
    {     
        BROCCOLI.SetInputFirstLevelResults(h_First_Level_Results);        
        BROCCOLI.SetInputMNIBrainMask(h_Mask);        
        BROCCOLI.SetMNIWidth(DATA_W);
        BROCCOLI.SetMNIHeight(DATA_H);
        BROCCOLI.SetMNIDepth(DATA_D);                

		BROCCOLI.SetAllocatedHostMemory(allocatedHostMemory);

        BROCCOLI.SetInferenceMode(INFERENCE_MODE);        
        BROCCOLI.SetClusterDefiningThreshold(CLUSTER_DEFINING_THRESHOLD);
        BROCCOLI.SetSignificanceLevel(SIGNIFICANCE_LEVEL);		
        BROCCOLI.SetNumberOfSubjects(NUMBER_OF_SUBJECTS);
        BROCCOLI.SetNumberOfSubjectsGroup1(NUMBER_OF_SUBJECTS_IN_GROUP1);
        BROCCOLI.SetNumberOfSubjectsGroup2(NUMBER_OF_SUBJECTS_IN_GROUP2);
        BROCCOLI.SetNumberOfPermutations(NUMBER_OF_PERMUTATIONS);
        BROCCOLI.SetNumberOfGroupPermutations(NUMBER_OF_PERMUTATIONS_PER_CONTRAST);
        BROCCOLI.SetNumberOfGLMRegressors(NUMBER_OF_GLM_REGRESSORS);
        BROCCOLI.SetNumberOfContrasts(NUMBER_OF_CONTRASTS);    
        BROCCOLI.SetDesignMatrix(h_X_GLM, h_xtxxt_GLM);
        BROCCOLI.SetContrasts(h_Contrasts);
        BROCCOLI.SetGLMScalars(h_ctxtxc_GLM);

        BROCCOLI.SetOutputStatisticalMapsMNI(h_Statistical_Maps);        
        BROCCOLI.SetOutputPermutationDistributions(h_Permutation_Distributions);
        BROCCOLI.SetOutputPValuesMNI(h_P_Values);        

		BROCCOLI.SetDoAllPermutations(DO_ALL_PERMUTATIONS);

		BROCCOLI.SetPermutationFileUsage(false);
		BROCCOLI.SetPrint(false);

		BROCCOLI.SetGroupDesigns(GROUP_DESIGNS);

        // Run the permutation test

		if (ANALYZE_GROUP_MEAN)
		{
		    BROCCOLI.SetSignMatrix(h_Sign_Matrix);          
	        BROCCOLI.SetStatisticalTest(2); // Group mean
			BROCCOLI.PerformMeanSecondLevelPermutationWrapper();                            
		}
		else if (ANALYZE_TTEST)
		{
	        BROCCOLI.SetPermutationMatrices(h_Permutation_Matrices);        
	        BROCCOLI.SetStatisticalTest(0); // t-test
	        BROCCOLI.PerformGLMTTestSecondLevelPermutationWrapper();                            
		}
		else if (ANALYZE_FTEST)
		{
	        BROCCOLI.SetPermutationMatrices(h_Permutation_Matrices);        
	        BROCCOLI.SetStatisticalTest(1); // F-test
	        BROCCOLI.PerformGLMFTestSecondLevelPermutationWrapper();                            
		}

		// Print create buffer errors
        int* createBufferErrors = BROCCOLI.GetOpenCLCreateBufferErrors();
        for (int i = 0; i < BROCCOLI.GetNumberOfOpenCLKernels(); i++)
        {
            if (createBufferErrors[i] != 0)
            {
                mexPrintf("Create buffer error %i is %s \n",i,BROCCOLI.GetOpenCLErrorMessage(createBufferErrors[i]));
            }
        }
        
        // Print create kernel errors
        int* createKernelErrors = BROCCOLI.GetOpenCLCreateKernelErrors();
        for (int i = 0; i < BROCCOLI.GetNumberOfOpenCLKernels(); i++)
        {
            if (createKernelErrors[i] != 0)
            {
                mexPrintf("Create kernel error for kernel '%s' is '%s' \n",BROCCOLI.GetOpenCLKernelName(i),BROCCOLI.GetOpenCLErrorMessage(createKernelErrors[i]));
            }
        } 

        // Print run kernel errors
        int* runKernelErrors = BROCCOLI.GetOpenCLRunKernelErrors();
        for (int i = 0; i < BROCCOLI.GetNumberOfOpenCLKernels(); i++)
        {
            if (runKernelErrors[i] != 0)
            {
                mexPrintf("Run kernel error for kernel '%s' is '%s' \n",BROCCOLI.GetOpenCLKernelName(i),BROCCOLI.GetOpenCLErrorMessage(runKernelErrors[i]));
            }
        }                 
    }
    
	for (size_t c = 0; c < NUMBER_OF_STATISTICAL_MAPS; c++)
	{
		h_Permutation_Distribution = h_Permutation_Distributions[c];

		std::vector<float> max_values (h_Permutation_Distribution, h_Permutation_Distribution + NUMBER_OF_PERMUTATIONS_PER_CONTRAST[c]);
        std::sort (max_values.begin(), max_values.begin() + NUMBER_OF_PERMUTATIONS_PER_CONTRAST[c]);
   
        // Find the threshold for the specified significance level
        float SIGNIFICANCE_THRESHOLD = max_values[(int)(ceil((1.0f - SIGNIFICANCE_LEVEL) * (float)NUMBER_OF_PERMUTATIONS_PER_CONTRAST[c]))-1];

        mexPrintf("\nPermutation threshold for contrast %zu for a significance level of %f is %f \n",c+1,SIGNIFICANCE_LEVEL, SIGNIFICANCE_THRESHOLD);
    }
	mexPrintf("\n");

    // Unpack results to Matlab
    unpack_float2double_volumes(h_Statistical_Maps_double, h_Statistical_Maps, DATA_W, DATA_H, DATA_D, NUMBER_OF_CONTRASTS); 
    unpack_float2double_volumes(h_P_Values_double, h_P_Values, DATA_W, DATA_H, DATA_D, NUMBER_OF_CONTRASTS); 
    //unpack_float2double(h_Permutation_Distribution_double, h_Permutation_Distribution, NUMBER_OF_PERMUTATIONS);  
            
    // Free all the allocated memory on the host
        
    mxFree(h_First_Level_Results);   
    mxFree(h_Mask);
        
    mxFree(h_X_GLM);
    mxFree(h_xtxxt_GLM);
    mxFree(h_Contrasts);
    mxFree(h_ctxtxc_GLM);    
            
    mxFree(h_Statistical_Maps);
    mxFree(h_P_Values);

	mxFree(h_Sign_Matrix);

	for (size_t c = 0; c < NUMBER_OF_STATISTICAL_MAPS; c++)
	{ 
		h_Permutation_Matrix = h_Permutation_Matrices[c];
		h_Permutation_Distribution = h_Permutation_Distributions[c];

		mxFree(h_Permutation_Matrix);
		mxFree(h_Permutation_Distribution);
	}

	mxFree(h_Permutation_Distributions);
	mxFree(h_Permutation_Matrices);
	    
    return;
}



