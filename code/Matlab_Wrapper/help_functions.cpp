#include "broccoli_lib.h"


void pack_double2float(float* output_float, double* input_double, int SIZE)
{
    for (int i = 0; i < SIZE ; i++)
    {
        output_float[i] = (float)input_double[i];       
    }
}


void pack_double2float_image(float* output_float, double* input_double, int DATA_W, int DATA_H)
{
    int i = 0;
    for (int x = 0; x < DATA_W ; x++)
    {
        for (int y = 0; y < DATA_H ; y++)
        {
            output_float[x + y * DATA_W] = (float)input_double[i];
            i++;
        }
    }
}


void unpack_float2double_image(double* output_double, float* input_float, int DATA_W, int DATA_H)
{
    int i = 0;
    for (int x = 0; x < DATA_W ; x++)
    {
        for (int y = 0; y < DATA_H ; y++)
        {
            output_double[i] = (double)input_float[x + y * DATA_W];
            i++;
        }
    }
}

void pack_double2float_volume(float* output_float, double* input_double, int DATA_W, int DATA_H, int DATA_D)
{
    int i = 0;
    for (int z = 0; z < DATA_D ; z++)
    {
        for (int x = 0; x < DATA_W ; x++)
        {
            for (int y = 0; y < DATA_H ; y++)
            {
                output_float[x + y * DATA_W + z * DATA_W * DATA_H] = (float)input_double[i];
                i++;
            }
        }
    }
}

void unpack_float2double(double* output_double, float* input_float, int SIZE)
{
    for (int i = 0; i < SIZE ; i++)
    {
        output_double[i] = (double)input_float[i];
    }
}

void unpack_float2double_volume(double* output_double, float* input_float, int DATA_W, int DATA_H, int DATA_D)
{
    int i = 0;
    for (int z = 0; z < DATA_D ; z++)
    {
        for (int x = 0; x < DATA_W ; x++)
        {
            for (int y = 0; y < DATA_H ; y++)
            {
                output_double[i] = (double)input_float[x + y * DATA_W + z * DATA_W * DATA_H];
                i++;
            }
        }
    }
}

void unpack_int2int_volume(int* output_int, int* input_int, int DATA_W, int DATA_H, int DATA_D)
{
    int i = 0;
    for (int z = 0; z < DATA_D ; z++)
    {
        for (int x = 0; x < DATA_W ; x++)
        {
            for (int y = 0; y < DATA_H ; y++)
            {
                output_int[i] = input_int[x + y * DATA_W + z * DATA_W * DATA_H];
                i++;
            }
        }
    }
}


void pack_double2float_volumes(float* output_float, double* input_double, int DATA_W, int DATA_H, int DATA_D, int DATA_T)
{
    int i = 0;
    for (int t = 0; t < DATA_T; t++)
    {
        for (int z = 0; z < DATA_D ; z++)
        {
            for (int x = 0; x < DATA_W ; x++)
            {
                for (int y = 0; y < DATA_H ; y++)
                {
                    output_float[x + y * DATA_W + z * DATA_W * DATA_H + t * DATA_W * DATA_H * DATA_D] = (float)input_double[i];
                    i++;
                }
            }
        }
    }
}



void unpack_float2double_volumes(double* output_double, float* input_float, int DATA_W, int DATA_H, int DATA_D, int DATA_T)
{
    int i = 0;
    for (int t = 0; t < DATA_T; t++)
    {
        for (int z = 0; z < DATA_D ; z++)
        {
            for (int x = 0; x < DATA_W ; x++)
            {
                for (int y = 0; y < DATA_H ; y++)
                {
                    output_double[i] = (double)input_float[x + y * DATA_W + z * DATA_W * DATA_H + t * DATA_W * DATA_H * DATA_D];
                    i++;
                }
            }
        }
    }
}


void pack_c2c_volume_(cl_float2 *output_float, double *input_re, double *input_im, int DATA_W, int DATA_H, int DATA_D)
{
	int i = 0;
	for (int z = 0; z < DATA_D ; z++)
	{
		for (int x = 0; x < DATA_W ; x++)
		{
			for (int y = 0; y < DATA_H ; y++)
			{
            	output_float[x + y * DATA_W + z * DATA_W * DATA_H].s[0] = (float)input_re[i];
               	output_float[x + y * DATA_W + z * DATA_W * DATA_H].s[1] = (float)input_im[i];                
				i++;
	      	}
		}
	}
}

void unpack_c2c_volume_(double *output_re, double *output_im, cl_float2 *input_float, int DATA_W, int DATA_H, int DATA_D)
{
    int i = 0;
    for (int z = 0; z < DATA_D ; z++)
    {
        for (int x = 0; x < DATA_W ; x++)
        {
            for (int y = 0; y < DATA_H ; y++)
            {
                output_re[i] = (double)input_float[x + y * DATA_W + z * DATA_W * DATA_H].s[0];
                output_im[i] = (double)input_float[x + y * DATA_W + z * DATA_W * DATA_H].s[1];
                i++;
            }
        }
    }    
}

void pack_c2c_volume(float2 *output_float, double *input_re, double *input_im, int DATA_W, int DATA_H, int DATA_D)
{
	int i = 0;
	for (int z = 0; z < DATA_D ; z++)
	{
		for (int x = 0; x < DATA_W ; x++)
		{
			for (int y = 0; y < DATA_H ; y++)
			{
            	output_float[x + y * DATA_W + z * DATA_W * DATA_H].x = (float)input_re[i];
               	output_float[x + y * DATA_W + z * DATA_W * DATA_H].y = (float)input_im[i];                
				i++;
	      	}
		}
	}
}

void unpack_c2c_volume(double *output_re, double *output_im, float2 *input_float, int DATA_W, int DATA_H, int DATA_D)
{
    int i = 0;
    for (int z = 0; z < DATA_D ; z++)
    {
        for (int x = 0; x < DATA_W ; x++)
        {
            for (int y = 0; y < DATA_H ; y++)
            {
                output_re[i] = (double)input_float[x + y * DATA_W + z * DATA_W * DATA_H].x;
                output_im[i] = (double)input_float[x + y * DATA_W + z * DATA_W * DATA_H].y;
                i++;
            }
        }
    }    
}



