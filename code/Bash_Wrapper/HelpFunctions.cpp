
#include <time.h>
#include <sys/time.h>

void LowpassFilterRegressor(float* h_LowpassFiltered_Regressor, float* h_Regressor, int DATA_T, int HIGHRES_FACTOR, float TR)
{
	// Allocate memory for lowpass filter
	int FILTER_LENGTH = 151;
	float* h_Filter = (float*)malloc(FILTER_LENGTH * sizeof(float));

	// Create lowpass filter
	int halfSize = (FILTER_LENGTH - 1) / 2;
	double smoothing_FWHM = 150.0;
	double sigma = smoothing_FWHM / 2.354 / (double)TR;
	double sigma2 = 2.0 * sigma * sigma;

	double u;
	float sum = 0.0f;
	for (int i = 0; i < FILTER_LENGTH; i++)
	{
		u = (double)(i - halfSize);
		h_Filter[i] = (float)exp(-pow(u,2.0) / sigma2);
		sum += h_Filter[i];
	}

	// Normalize
	for (int i = 0; i < FILTER_LENGTH; i++)
	{
		h_Filter[i] /= sum;
	}

	// Convolve regressor with filter
	for (int t = 0; t < (DATA_T * HIGHRES_FACTOR); t++)
	{
		h_LowpassFiltered_Regressor[t] = 0.0f;

		// 1D convolution
		//int offset = -(int)(((float)HRF_LENGTH - 1.0f)/2.0f);
		int offset = -(int)(((float)FILTER_LENGTH - 1.0f)/2.0f);
		for (int tt = FILTER_LENGTH - 1; tt >= 0; tt--)
		{
			if ( ((t + offset) >= 0) && ((t + offset) < (DATA_T * HIGHRES_FACTOR)) )
			{
				h_LowpassFiltered_Regressor[t] += h_Regressor[t + offset] * h_Filter[tt];
			}
			offset++;
		}
	}

	free(h_Filter);
}


void ConvertFloat2ToFloats(float* Real, float* Imag, cl_float2* Complex, int DATA_W, int DATA_H, int DATA_D)
{
    for (int z = 0; z < DATA_D; z++)
    {
        for (int y = 0; y < DATA_H; y++)
        {   
           for (int x = 0; x < DATA_W; x++)
           {    
               Real[x + y * DATA_W + z * DATA_W * DATA_H] = Complex[x + y * DATA_W + z * DATA_W * DATA_H].x;
               Imag[x + y * DATA_W + z * DATA_W * DATA_H] = Complex[x + y * DATA_W + z * DATA_W * DATA_H].y;
           }
        }
    }
    
}

void FreeAllMemory(void **pointers, int N)
{
    for (int i = 0; i < N; i++)
    {
        if (pointers[i] != NULL)
        {
            free(pointers[i]);
        }
    }
}

void FreeAllNiftiImages(nifti_image **niftiImages, int N)
{
    for (int i = 0; i < N; i++)
    {
		if (niftiImages[i] != NULL)
		{
			nifti_image_free(niftiImages[i]);
		}
    }
}

void ReadBinaryFile(float* pointer, int size, const char* filename, void** pointers, int& Npointers, nifti_image** niftiImages, int Nimages)
{
	if (pointer == NULL)
    {
        printf("The provided pointer for file %s is NULL, aborting! \n",filename);
        FreeAllMemory(pointers,Npointers);
		FreeAllNiftiImages(niftiImages,Nimages);
        exit(EXIT_FAILURE);
	}	

	FILE *fp = NULL; 
	fp = fopen(filename,"rb");

    if (fp != NULL)
    {
        fread(pointer,sizeof(float),size,fp);
        fclose(fp);
    }
    else
    {
        printf("Could not open %s , aborting! \n",filename);
        FreeAllMemory(pointers,Npointers);
		FreeAllNiftiImages(niftiImages,Nimages);
        exit(EXIT_FAILURE);
    }
}

void AllocateMemory(float *& pointer, size_t size, void** pointers, int& Npointers, nifti_image** niftiImages, int Nimages, size_t& allocatedMemory, const char* variable)
{
    pointer = (float*)malloc(size);
    if (pointer != NULL)
    {
        pointers[Npointers] = (void*)pointer;
        Npointers++;
		allocatedMemory += size;
    }
    else
    {
   		perror ("The following error occurred");
	    printf("Could not allocate host memory for variable %s ! \n",variable);     
	 	FreeAllMemory(pointers, Npointers);
		FreeAllNiftiImages(niftiImages, Nimages);
		exit(EXIT_FAILURE);        
    }
}

void AllocateMemoryInt(unsigned short int *& pointer, size_t size, void** pointers, int& Npointers, nifti_image** niftiImages, int Nimages, size_t allocatedMemory, const char* variable)
{
    pointer = (unsigned short int*)malloc(size);
    if (pointer != NULL)
    {
        pointers[Npointers] = (void*)pointer;
        Npointers++;
		allocatedMemory += size;
    }
    else
    {
		perror ("The following error occurred");
        printf("Could not allocate host memory for variable %s ! \n",variable);        
		FreeAllMemory(pointers, Npointers);
		FreeAllNiftiImages(niftiImages, Nimages);
		exit(EXIT_FAILURE);        
    }
}

    
void AllocateMemoryFloat2(cl_float2 *& pointer, int size, void** pointers, int& Npointers, nifti_image** niftiImages, int Nimages, size_t allocatedMemory, const char* variable)
{
    pointer = (cl_float2*)malloc(size);
    if (pointer != NULL)
    {
        pointers[Npointers] = (void*)pointer;
        Npointers++;
		allocatedMemory += size;
    }
    else
    {
		perror ("The following error occurred");
        printf("Could not allocate host memory for variable %s ! \n",variable);        
		FreeAllMemory(pointers, Npointers);
		FreeAllNiftiImages(niftiImages, Nimages);
		exit(EXIT_FAILURE);        
    }
}

float mymax(float* data, int N)
{
	float max = -100000.0f;
	for (int i = 0; i < N; i++)
	{
		if (data[i] > max)
			max = data[i];
	}

	return max;
}


float mymin(float* data, int N)
{
	float min = 100000.0f;
	for (int i = 0; i < N; i++)
	{
		if (data[i] < min)
			min = data[i];
	}

	return min;
}

bool WriteNifti(nifti_image* inputNifti, float* data, const char* filename, bool addFilename, bool checkFilename)
{       
	if (data == NULL)
    {
        printf("The provided data pointer for file %s is NULL, aborting writing nifti file! \n",filename);
		return false;
	}	
	if (inputNifti == NULL)
    {
        printf("The provided nifti pointer for file %s is NULL, aborting writing nifti file! \n",filename);
		return false;
	}	


    char* filenameWithExtension;
    
    // Add the provided filename extension to the original filename, before the dot
    if (addFilename)
    {
        // Find the dot in the original filename
        const char* p = inputNifti->fname;
        int dotPosition = 0;
        while ( (p != NULL) && ((*p) != '.') )
        {
            p++;
            dotPosition++;
        }
    
        // Allocate temporary array
        filenameWithExtension = (char*)malloc(strlen(inputNifti->fname) + strlen(filename) + 1);
        if (filenameWithExtension == NULL)
        {
            printf("Could not allocate temporary host memory! \n");      
            return false;
        }
    
        // Copy filename to the dot
        strncpy(filenameWithExtension,inputNifti->fname,dotPosition);
        filenameWithExtension[dotPosition] = '\0';
        // Add the extension
        strcat(filenameWithExtension,filename);
        // Add the rest of the original filename
        strcat(filenameWithExtension,inputNifti->fname+dotPosition);    
    }
        
    // Copy information from input data
    nifti_image *outputNifti = nifti_copy_nim_info(inputNifti);    
    // Set data pointer 
    outputNifti->data = (void*)data;        
    // Set data type to float
    outputNifti->datatype = DT_FLOAT;
    outputNifti->nbyper = 4;    
    
	// Change cal_min and cal_max, to get the scaling right in AFNI and FSL
    int N = inputNifti->nx * inputNifti->ny * inputNifti->nz * inputNifti->nt;
	outputNifti->cal_min = mymin(data,N);
	outputNifti->cal_max = mymax(data,N);

    // Change filename and write
    bool written = false;
    if (addFilename)
    {
        if ( nifti_set_filenames(outputNifti, filenameWithExtension, checkFilename, 1) == 0)
        {
            nifti_image_write(outputNifti);
            written = true;
        }
    }
    else if (!addFilename)
    {
        if ( nifti_set_filenames(outputNifti, filename, checkFilename, 1) == 0)
        {
            nifti_image_write(outputNifti);
            written = true;
        }                
    }    
    
    outputNifti->data = NULL;
    nifti_image_free(outputNifti);

    if (addFilename)
    {
        free(filenameWithExtension);
    } 
        
    if (written)
    {      
        return true;
    }
    else
    {
        return false;
    }                        
}

double GetWallTime()
{
    struct timeval time;
    if (gettimeofday(&time,NULL))
    {
        //  Handle error
        return 0;
    }
    return (double)time.tv_sec + (double)time.tv_usec * .000001;
}

