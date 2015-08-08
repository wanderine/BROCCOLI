/*
 * BROCCOLI: Software for fast fMRI analysis on many-core CPUs and GPUs
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

#include "broccoli_lib.h"
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <iomanip>

int main(int argc, char **argv)
{
    // Default parameters
    int     OPENCL_PLATFORM = 0;
    int     OPENCL_DEVICE = 0;

	bool	FOUND_PLATFORM = false;
	bool 	FOUND_DEVICE = false;

    // No inputs, so print help text
    if (argc == 1)
    {        
        printf("Usage:\n\n");
        printf("GetBandwidthPerformance -platform x -device y\n\n");
        printf(" -platform           The OpenCL platform to use \n");
        printf(" -device             The OpenCL device to use for the specificed platform  \n");
        printf("\n\n");
        
        return EXIT_SUCCESS;
    }

 	// Loop over additional inputs
    int i = 1;
    while (i < argc)
    {
        char *input = argv[i];
        char *p;
        if (strcmp(input,"-platform") == 0)
        {
			if ( (i+1) >= argc  )
			{
			    printf("Unable to read value after -platform !\n");
                return EXIT_FAILURE;
			}

            OPENCL_PLATFORM = (int)strtol(argv[i+1], &p, 10);
			FOUND_PLATFORM = true;

			if (!isspace(*p) && *p != 0)
		    {
		        printf("OpenCL platform must be an integer! You provided %s \n",argv[i+1]);
				return EXIT_FAILURE;
		    }
            else if (OPENCL_PLATFORM < 0)
            {
                printf("OpenCL platform must be >= 0!\n");
                return EXIT_FAILURE;
            }
            i += 2;
        }
        else if (strcmp(input,"-device") == 0)
        {
			if ( (i+1) >= argc  )
			{
			    printf("Unable to read value after -device !\n");
                return EXIT_FAILURE;
			}

            OPENCL_DEVICE = (int)strtol(argv[i+1], &p, 10);
			FOUND_DEVICE = true;

			if (!isspace(*p) && *p != 0)
		    {
		        printf("OpenCL device must be an integer! You provided %s \n",argv[i+1]);
				return EXIT_FAILURE;
		    }
            else if (OPENCL_DEVICE < 0)
            {
                printf("OpenCL device must be >= 0!\n");
                return EXIT_FAILURE;
            }
            i += 2;
        }
        else
        {
            printf("Unrecognized option! %s \n",argv[i]);
            return EXIT_FAILURE;
        }   
	}

	if (!FOUND_PLATFORM)
	{
        printf("No OpenCL platform given, aborting!\n");
        return EXIT_FAILURE;
	}

	if (!FOUND_DEVICE)
	{
        printf("No OpenCL device given, aborting!\n");
        return EXIT_FAILURE;
	}

	BROCCOLI_LIB BROCCOLI(OPENCL_PLATFORM,OPENCL_DEVICE,2,false); // 2 = Bash wrapper

	BROCCOLI.GetBandwidth();
    
            
    return EXIT_SUCCESS;
}

