	/*
 * BROCCOLI: Software for Fast fMRI Analysis on Many-Core CPUs and GPUs
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

// OpenGL Graphics Includes
#include <GL/glew.h>
    #include <GL/gl.h>
    #include <GL/freeglut.h>
       #include <GL/glx.h>

// Includes
#include <iostream>
#include <string>

#include <sstream>
#include <stdio.h>
#include <stdlib.h>
#include "nifti1_io.h"
#include <fstream>
#include <iomanip>
#include <time.h>
#include <sys/time.h>
#include <opencl.h>
#include <vector>

#if defined (__APPLE__) || defined(MACOSX)
   #define GL_SHARING_EXTENSION "cl_APPLE_gl_sharing"
#else
   #define GL_SHARING_EXTENSION "cl_khr_gl_sharing"
#endif

// Constants, defines, typedefs and global declarations
//*****************************************************************************
#define REFRESH_DELAY	  10 //ms

int *pArgc = NULL;
char **pArgv = NULL;

typedef unsigned int uint;
typedef unsigned char uchar;

uint width = 512, height = 512;
size_t gridSize[2] = {width, height};

#define LOCAL_SIZE_X 16
#define LOCAL_SIZE_Y 16

float viewRotation[3];
float viewTranslation[3] = {0.0, 0.0, -4.0f};
float invViewMatrix[12];

float density = 0.05f;
float brightness = 1.0f;
float transferOffset = 0.0f;
float transferScale = 2.0f;
bool linearFiltering = true;

GLuint pbo = 0;                 // OpenGL pixel buffer object
int iGLUTWindowHandle;          // handle to the GLUT window

// OpenCL vars
cl_platform_id cpPlatform;
cl_uint uiNumDevices;
cl_device_id* cdDevices;
cl_uint uiDeviceUsed;
cl_uint uiDevCount;
cl_context cxGPUContext;
cl_device_id device;
cl_command_queue cqCommandQueue;
cl_program cpProgram;
cl_kernel ckKernel;
cl_int ciErrNum;
cl_mem pbo_cl;
cl_mem d_volumeArray;
cl_mem d_transferFuncArray;
cl_mem d_invViewMatrix;
const char* cPathAndName = "volumeRender.cl";          // var for full paths to data, src, etc.
char* cSourceCL;                    // Buffer to hold source for compilation 
const char* cExecutableName = NULL;
cl_bool g_bImageSupport;
cl_sampler volumeSamplerLinear;
cl_sampler volumeSamplerNearest;
cl_sampler transferFuncSampler;
cl_bool g_glInterop = false;

int iFrameCount = 0;                // FPS count for averaging
int iFrameTrigger = 20;             // FPS trigger for sampling
int iFramesPerSec = 0;              // frames per second
int iTestSets = 3;
int g_Index = 0;
bool bNoPrompt = false;		// false = normal GL loop, true = Finite period of GL loop (a few seconds)
bool bQATest = false;			// false = normal GL loop, true = run No-GL test sequence  
bool g_bFBODisplay = false;
int ox, oy;                         // mouse location vars
int buttonState = 0;         



// OpenCL Functions
void initPixelBuffer();
void render();
void createCLContext(int argc, const char** argv);
void initCLVolume(float *h_Volume, int DATA_W, int DATA_H, int DATA_D);

// OpenGL functionality
void InitGL(int* argc, char** argv);
void DisplayGL();
void Reshape(int w, int h);
void Idle(void);
void KeyboardGL(unsigned char key, int x, int y);
void timerEvent(int value);
void motion(int x, int y);
void mouse(int button, int state, int x, int y);

// Helpers
void Cleanup(int iExitCode);
void (*pCleanup)(int) = &Cleanup;
void TestNoGL();


// render image using OpenCL
//*****************************************************************************
void render()
{
    ciErrNum = CL_SUCCESS;

    // Transfer ownership of buffer from GL to CL

	if( g_glInterop ) 
	{
		// Acquire PBO for OpenCL writing
		glFlush();
		ciErrNum |= clEnqueueAcquireGLObjects(cqCommandQueue, 1, &pbo_cl, 0, 0, 0);
		//printf("Enqueue acquired GL objects error is %i \n",ciErrNum);
	}

	ciErrNum |= clEnqueueWriteBuffer(cqCommandQueue,d_invViewMatrix,CL_FALSE, 0,12*sizeof(float), invViewMatrix, 0, 0, 0);	
	//printf("Write buffer error is %i \n",ciErrNum);

    // execute OpenCL kernel, writing results to PBO
    size_t localSize[] = {LOCAL_SIZE_X,LOCAL_SIZE_Y};

    ciErrNum |= clSetKernelArg(ckKernel, 3, sizeof(float), &density);
    ciErrNum |= clSetKernelArg(ckKernel, 4, sizeof(float), &brightness);
    ciErrNum |= clSetKernelArg(ckKernel, 5, sizeof(float), &transferOffset);
    ciErrNum |= clSetKernelArg(ckKernel, 6, sizeof(float), &transferScale);
    ciErrNum |= clEnqueueNDRangeKernel(cqCommandQueue, ckKernel, 2, NULL, gridSize, localSize, 0, 0, 0);
	//printf("Enqueue ND range kernel error is %i \n",ciErrNum);
    ////oclCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);
		clFinish( cqCommandQueue );

	if( g_glInterop ) 
	{
		// Transfer ownership of buffer back from CL to GL    
		ciErrNum |= clEnqueueReleaseGLObjects(cqCommandQueue, 1, &pbo_cl, 0, 0, 0);
		//printf("Release GL object error is %i \n",ciErrNum);
		////oclCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);
		clFinish( cqCommandQueue );
	} 
	else 
	{
		// Explicit Copy 
		// map the PBO to copy data from the CL buffer via host
		glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, pbo);    

		// map the buffer object into client's memory
		GLubyte* ptr = (GLubyte*)glMapBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB,
			GL_WRITE_ONLY_ARB);
		clEnqueueReadBuffer(cqCommandQueue, pbo_cl, CL_TRUE, 0, sizeof(unsigned int) * height * width, ptr, 0, NULL, NULL);        
		////oclCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);
		glUnmapBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB); 
	}
}

// Display callback for GLUT main loop
//*****************************************************************************
void DisplayGL()
{
    // use OpenGL to build view matrix
    GLfloat modelView[16];
    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
    glLoadIdentity();
    glRotatef(-viewRotation[0], 1.0, 0.0, 0.0);
    glRotatef(-viewRotation[1], 0.0, 1.0, 0.0);
    glTranslatef(-viewTranslation[0], -viewTranslation[1], -viewTranslation[2]);
    glGetFloatv(GL_MODELVIEW_MATRIX, modelView);
    glPopMatrix();

    invViewMatrix[0] = modelView[0]; invViewMatrix[1] = modelView[4]; invViewMatrix[2] = modelView[8]; invViewMatrix[3] = modelView[12];
    invViewMatrix[4] = modelView[1]; invViewMatrix[5] = modelView[5]; invViewMatrix[6] = modelView[9]; invViewMatrix[7] = modelView[13];
    invViewMatrix[8] = modelView[2]; invViewMatrix[9] = modelView[6]; invViewMatrix[10] = modelView[10]; invViewMatrix[11] = modelView[14];

    // start timer 0 if it's update time
    double dProcessingTime = 0.0;
    if (iFrameCount >= iFrameTrigger)
    {
        //shrDeltaT(0); 
    }

     // process 
    render();

	//printf("Framecount is %i \n",iFrameTrigger);

    // get processing time from timer 0, if it's update time
    if (iFrameCount >= iFrameTrigger)
    {
        //dProcessingTime = shrDeltaT(0); 
    }

    // draw image from PBO
    glClear(GL_COLOR_BUFFER_BIT);
    glDisable(GL_DEPTH_TEST);
    glRasterPos2i(0, 0);
    glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, pbo);
    glDrawPixels(width, height, GL_RGBA, GL_UNSIGNED_BYTE, 0);
    glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, 0);

    // flip backbuffer to screen
    glutSwapBuffers();

    // Increment the frame counter, and do fps and Q/A stuff if it's time
    if (iFrameCount++ > iFrameTrigger) 
    {
        // set GLUT Window Title
        char cFPS[256];
        //iFramesPerSec = (int)((double)iFrameCount/shrDeltaT(1));
		iFramesPerSec = 20;
#ifdef GPU_PROFILING
        #ifdef _WIN32
            sprintf_s(cFPS, 256, "Volume Render %ux%u | %i fps | Proc.t = %.3f s | %.3f MP/s", 
                width, height, iFramesPerSec, dProcessingTime, (1.0e-6 * width * height)/dProcessingTime);
        #else 
            sprintf(cFPS, "Volume Render %ux%u |  %i fps | Proc. t = %.3f s | %.3f MP/s", 
                width, height, iFramesPerSec, dProcessingTime, (1.0e-6 * width * height)/dProcessingTime);
        #endif
#else
        #ifdef _WIN32
            sprintf_s(cFPS, 256, "Volume Render | W: %u  H: %u", width, height);
        #else 
            sprintf(cFPS, "Volume Render | W: %u  H: %u", width, height);
        #endif
#endif
        glutSetWindowTitle(cFPS);

        // if doing quick test, exit
        if ((bNoPrompt) && (!--iTestSets))
        {
            // Cleanup up and quit
            Cleanup(EXIT_SUCCESS);
        }

        // reset framecount, trigger and timer
        iFrameCount = 0; 
        iFrameTrigger = (iFramesPerSec > 1) ? iFramesPerSec * 2 : 1;
    }
}

// GL Idle time callback
//*****************************************************************************
void Idle()
{
}

void timerEvent(int value)
{
    glutPostRedisplay();
	glutTimerFunc(REFRESH_DELAY, timerEvent,0);
}

// Keyboard event handler callback
//*****************************************************************************
void KeyboardGL(unsigned char key, int /*x*/, int /*y*/)
{
    switch(key) 
    {
        case '-':
            density -= 0.01f;
			printf("Density--\n");
            break;
        case '+':
            density += 0.01f;
			printf("Density++\n");
            break;
        case ']':
            brightness += 0.1f;
			printf("Brightness++\n");
            break;
        case '[':
            brightness -= 0.1f;
			printf("Brightness--\n");
            break;
        case ';':
            transferOffset += 0.01f;
			printf("TransferOffset++\n");
            break;
        case '\'':
            transferOffset -= 0.01f;
			printf("TransferOffset--\n");
            break;
        case '.':
            transferScale += 0.01f;
			printf("TransferScale++\n");
            break;
        case ',':
            transferScale -= 0.01f;
			printf("TransferScale--\n");
            break;
        case '\033': // escape quits
        case '\015': // Enter quits    
        case 'Q':    // Q quits
        case 'q':    // q (or escape) quits
            // Cleanup up and quit
            bNoPrompt = true;
            Cleanup(EXIT_SUCCESS);
            break;
        case 'F':
        case 'f':
                    linearFiltering = !linearFiltering;
                    ciErrNum = clSetKernelArg(ckKernel, 10, sizeof(cl_sampler), linearFiltering ? &volumeSamplerLinear : &volumeSamplerNearest);
                    ////shrlog("\nLinear Filtering Toggled %s...\n", linearFiltering ? "ON" : "OFF");
                    //oclCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);
                    break;
        default:
            break;
    }
    ////shrlog("density = %.2f, brightness = %.2f, transferOffset = %.2f, transferScale = %.2f\n", density, brightness, transferOffset, transferScale);
}

// Mouse event handlers
//*****************************************************************************
void mouse(int button, int state, int x, int y)
{
    if (state == GLUT_DOWN)
        buttonState |= 1<<button;
    else if (state == GLUT_UP)
        buttonState = 0;

    ox = x; 
    oy = y;
}

void motion(int x, int y)
{
    float dx, dy;
    dx = (float)(x - ox);
    dy = (float)(y - oy);

    if (buttonState == 3) 	
	{
        // left+middle = zoom
        viewTranslation[2] += dy / 100.0f;
    } 
    else if (buttonState & 2) 
	{
        // middle = translate
        viewTranslation[0] += dx / 100.0f;
        viewTranslation[1] -= dy / 100.0f;
    }
    else if (buttonState & 1) 
	{
        // left = rotate
        viewRotation[0] += dy / 5.0f;
        viewRotation[1] += dx / 5.0f;
    }

    ox = x; 
    oy = y;
}

// Window resize handler callback
//*****************************************************************************
void Reshape(int x, int y)
{
    width = x; height = y;
    initPixelBuffer();

    glViewport(0, 0, x, y);

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0.0, 1.0, 0.0, 1.0, 0.0, 1.0); 
}

// Intitialize OpenCL
//*****************************************************************************
void createCLContext(int PLATFORM, int DEVICE) 
{
	cl_int error;

	cl_uint platformIdCount = 0;
	clGetPlatformIDs (0, NULL, &platformIdCount);
	std::vector<cl_platform_id> platformIds (platformIdCount);
	clGetPlatformIDs (platformIdCount, platformIds.data(), NULL);

	cpPlatform = platformIds[PLATFORM];

    // Get the number of GPU devices available to the platform
    ciErrNum = clGetDeviceIDs(cpPlatform, CL_DEVICE_TYPE_GPU, 0, NULL, &uiDevCount);
	printf("Number of GPU devices is %i \n",uiDevCount);
    //oclCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);

    // Create the device list
    unsigned int uiEndDev = uiDevCount - 1;
    cdDevices = new cl_device_id [uiDevCount];
    ciErrNum = clGetDeviceIDs(cpPlatform, CL_DEVICE_TYPE_GPU, uiDevCount, cdDevices, NULL);

	// Get device name
	char* value;
	size_t valueSize;

	clGetDeviceInfo(cdDevices[DEVICE], CL_DEVICE_NAME, 0, NULL, &valueSize);
	value = (char*) malloc(valueSize);
	clGetDeviceInfo(cdDevices[DEVICE], CL_DEVICE_NAME, valueSize, value, NULL);    
	printf("Devide name is %s \n",value);
	free(value);

	// Check if the requested device (or any of the devices if none requested) supports context sharing with OpenGL
    if(g_glInterop)
    {
        bool bSharingSupported = false;
        for (unsigned int i = uiDeviceUsed; (!bSharingSupported && (i <= uiEndDev)); ++i) 
        {
            size_t extensionSize;
            ciErrNum = clGetDeviceInfo(cdDevices[i], CL_DEVICE_EXTENSIONS, 0, NULL, &extensionSize );
            //oclCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);
            if(extensionSize > 0) 
            {
                char* extensions = (char*)malloc(extensionSize);
                ciErrNum = clGetDeviceInfo(cdDevices[i], CL_DEVICE_EXTENSIONS, extensionSize, extensions, &extensionSize);
				printf("Get device info error is %i \n",ciErrNum);
                //oclCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);
                std::string stdDevString(extensions);
                free(extensions);

                size_t szOldPos = 0;
                size_t szSpacePos = stdDevString.find(' ', szOldPos); // extensions string is space delimited
                while (szSpacePos != stdDevString.npos)
                {
                    if( strcmp(GL_SHARING_EXTENSION, stdDevString.substr(szOldPos, szSpacePos - szOldPos).c_str()) == 0 ) 
                    {
                        // Device supports context sharing with OpenGL
                        uiDeviceUsed = i;
                        bSharingSupported = true;
						printf("Sharing supported!\n");
                        break;
                    }
                    do 
                    {
                        szOldPos = szSpacePos + 1;
                        szSpacePos = stdDevString.find(' ', szOldPos);
                    } 
                    while (szSpacePos == szOldPos);
                }
            }
        }    

 // Define OS-specific context properties and create the OpenCL context
        
        cl_context_properties props[] = 
                {
                    CL_GL_CONTEXT_KHR, (cl_context_properties)glXGetCurrentContext(), 
                    CL_GLX_DISPLAY_KHR, (cl_context_properties)glXGetCurrentDisplay(), 
                    CL_CONTEXT_PLATFORM, (cl_context_properties)cpPlatform, 
                    0
                };
                cxGPUContext = clCreateContext(props, 1, &cdDevices[uiDeviceUsed], NULL, NULL, &ciErrNum);
				printf("Create context error is %i\n",ciErrNum);

	}
	else 
    {
		// No GL interop
        cl_context_properties props[] = {CL_CONTEXT_PLATFORM, (cl_context_properties)PLATFORM, 0};
        cxGPUContext = clCreateContext(props, 1, &cdDevices[uiDeviceUsed], NULL, NULL, &error);
		printf("Error 7 is %i \n",error);
		g_glInterop = false;
    }
    //oclCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);
}

void initCLVolume(float *h_Volume, int DATA_W, int DATA_H, int DATA_D)
{
    ciErrNum = CL_SUCCESS;

	printf("b image support is %i \n",g_bImageSupport);

	if (g_bImageSupport)
	//if (true)  
    {
		// create 3D array and copy data to device
		cl_image_format volume_format;
        volume_format.image_channel_order = CL_RGBA;
        volume_format.image_channel_data_type = CL_UNORM_INT8;
        unsigned char* h_tempVolume = (unsigned char*)malloc(DATA_W * DATA_H * DATA_D * sizeof(unsigned char) * 4);
        for(int i = 0; i <(int)(DATA_W * DATA_H * DATA_D); i++)
        {
            h_tempVolume[4 * i] = (unsigned char)(h_Volume[i] / 10.0f);
			//h_tempVolume[4 * i] = h_Volume[i]/10.0f;
			//h_tempVolume[4 * i + 0] = 10;
			//h_tempVolume[4 * i + 1] = 10;
			//h_tempVolume[4 * i + 2] = 10;
			//h_tempVolume[4 * i + 3] = 10;
        }
		int error;
        d_volumeArray = clCreateImage3D(cxGPUContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, &volume_format, 
                                        DATA_W, DATA_H, DATA_D,
                                        (DATA_W * 4), (DATA_W * DATA_H * 4),
										//0, 0,
                                        h_tempVolume, &error);

		printf("Create image 3D error is %i \n",error);
		//oclCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);
        free (h_tempVolume);

		// create transfer function texture
		float transferFunc[] = {
			 0.0, 0.0, 0.0, 0.0, 
			 1.0, 0.0, 0.0, 1.0, 
			 1.0, 0.5, 0.0, 1.0, 
			 1.0, 1.0, 0.0, 1.0, 
			 0.0, 1.0, 0.0, 1.0, 
			 0.0, 1.0, 1.0, 1.0, 
			 0.0, 0.0, 1.0, 1.0, 
			 1.0, 0.0, 1.0, 1.0, 
			 0.0, 0.0, 0.0, 0.0, 
		};

		cl_image_format transferFunc_format;
		transferFunc_format.image_channel_order = CL_RGBA;
		transferFunc_format.image_channel_data_type = CL_FLOAT;
		d_transferFuncArray = clCreateImage2D(cxGPUContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, &transferFunc_format,
											  9, 1, sizeof(float) * 9 * 4,
											  transferFunc, &ciErrNum);                                          
		printf("Error 8 is %i \n",ciErrNum);
		//oclCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);

        // Create samplers for transfer function, linear interpolation and nearest interpolation 
        transferFuncSampler = clCreateSampler(cxGPUContext, true, CL_ADDRESS_CLAMP_TO_EDGE, CL_FILTER_LINEAR, &ciErrNum);
		printf("Error 9 is %i \n",ciErrNum);
        //oclCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);
        volumeSamplerLinear = clCreateSampler(cxGPUContext, true, CL_ADDRESS_REPEAT, CL_FILTER_LINEAR, &ciErrNum);
		printf("Error 10 is %i \n",ciErrNum);
        //oclCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);
        volumeSamplerNearest = clCreateSampler(cxGPUContext, true, CL_ADDRESS_REPEAT, CL_FILTER_NEAREST, &ciErrNum);
		printf("Error 11 is %i \n",ciErrNum);
        //oclCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);

        // set image and sampler args
        ciErrNum = clSetKernelArg(ckKernel, 8, sizeof(cl_mem), (void *) &d_volumeArray);
		ciErrNum |= clSetKernelArg(ckKernel, 9, sizeof(cl_mem), (void *) &d_transferFuncArray);
        ciErrNum |= clSetKernelArg(ckKernel, 10, sizeof(cl_sampler), linearFiltering ? &volumeSamplerLinear : &volumeSamplerNearest);
        ciErrNum |= clSetKernelArg(ckKernel, 11, sizeof(cl_sampler), &transferFuncSampler);
		//oclCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);
		printf("Error 12 is %i \n",ciErrNum);
	}

    // init invViewMatrix
    d_invViewMatrix = clCreateBuffer(cxGPUContext, CL_MEM_READ_ONLY, 12 * sizeof(float), 0, &ciErrNum);
    ciErrNum |= clSetKernelArg(ckKernel, 7, sizeof(cl_mem), (void *) &d_invViewMatrix);
	printf("Error 13 is %i \n",ciErrNum);
    //oclCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);
}

// Initialize GL
//*****************************************************************************
void InitGL(int* argc, char **argv)
{
    // initialize GLUT 
    glutInit(argc, argv);
    glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE);
    glutInitWindowPosition (glutGet(GLUT_SCREEN_WIDTH)/2 - width/2, 
                            glutGet(GLUT_SCREEN_HEIGHT)/2 - height/2);
    glutInitWindowSize(width, height);
    iGLUTWindowHandle = glutCreateWindow("OpenCL volume rendering");
#if !(defined (__APPLE__) || defined(MACOSX))
    glutSetOption(GLUT_ACTION_ON_WINDOW_CLOSE, GLUT_ACTION_GLUTMAINLOOP_RETURNS);
#endif

    // register glut callbacks
    glutDisplayFunc(DisplayGL);
    glutKeyboardFunc(KeyboardGL);
    glutMouseFunc(mouse);
    glutMotionFunc(motion);
    glutReshapeFunc(Reshape);
    glutIdleFunc(Idle);
	glutTimerFunc(REFRESH_DELAY, timerEvent,0);

	// initialize necessary OpenGL extensions
    glewInit();
    GLboolean bGLEW = glewIsSupported("GL_VERSION_2_0 GL_ARB_pixel_buffer_object"); 
    //oclCheckErrorEX(bGLEW, true, pCleanup);

	g_glInterop = true;
}

// Initialize GL
//*****************************************************************************
void initPixelBuffer()
{
     ciErrNum = CL_SUCCESS;

    if (pbo) {
        // delete old buffer
        clReleaseMemObject(pbo_cl);
        glDeleteBuffersARB(1, &pbo);
    }

    // create pixel buffer object for display
    glGenBuffersARB(1, &pbo);
	glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, pbo);
	glBufferDataARB(GL_PIXEL_UNPACK_BUFFER_ARB, width * height * sizeof(GLubyte) * 4, 0, GL_STREAM_DRAW_ARB);
	glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, 0);

	if( g_glInterop ) 
	{
		// create OpenCL buffer from GL PBO
		pbo_cl = clCreateFromGLBuffer(cxGPUContext,CL_MEM_WRITE_ONLY, pbo, &ciErrNum);
		//oclCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);
		printf("Create from GL buffer error %i\n",ciErrNum);		
	} 
	else 
	{
		pbo_cl = clCreateBuffer(cxGPUContext, CL_MEM_WRITE_ONLY, width * height * sizeof(GLubyte) * 4, NULL, &ciErrNum);
	}

    // calculate new grid size
	//gridSize[0] = floor(width/LOCAL_SIZE_X);
	//gridSize[1] = floor(height/LOCAL_SIZE_Y);

	gridSize[0] = width;
	gridSize[1] = height;

	int xBlocks = (size_t)ceil((float)width / (float)LOCAL_SIZE_X);
	int yBlocks = (size_t)ceil((float)height / (float)LOCAL_SIZE_Y);

	// Calculate total number of threads (this is done to guarantee that total number of threads is multiple of local work size, required by OpenCL)
	gridSize[0] = xBlocks * LOCAL_SIZE_X;
	gridSize[1] = yBlocks * LOCAL_SIZE_Y;

    ciErrNum |= clSetKernelArg(ckKernel, 0, sizeof(cl_mem), (void *) &pbo_cl);
    ciErrNum |= clSetKernelArg(ckKernel, 1, sizeof(unsigned int), &width);
    ciErrNum |= clSetKernelArg(ckKernel, 2, sizeof(unsigned int), &height);
	printf("Error first kernel args is %i\n",ciErrNum);

    //oclCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);
}

// Run a test sequence without any GL 
//*****************************************************************************
void TestNoGL()
{
    // execute OpenCL kernel without GL interaction
    float modelView[16] = 
        {
            1.0f, 0.0f, 0.0f, 0.0f,
            0.0f, 1.0f, 0.0f, 0.0f,
            0.0f, 0.0f, 1.0f, 0.0f,
            0.0f, 0.0f, 4.0f, 1.0f
        };
    
    invViewMatrix[0] = modelView[0]; invViewMatrix[1] = modelView[4]; invViewMatrix[2] = modelView[8]; invViewMatrix[3] = modelView[12];
    invViewMatrix[4] = modelView[1]; invViewMatrix[5] = modelView[5]; invViewMatrix[6] = modelView[9]; invViewMatrix[7] = modelView[13];
    invViewMatrix[8] = modelView[2]; invViewMatrix[9] = modelView[6]; invViewMatrix[10] = modelView[10]; invViewMatrix[11] = modelView[14];
    
    pbo_cl = clCreateBuffer(cxGPUContext, CL_MEM_WRITE_ONLY,  width*height*sizeof(GLubyte)*4, NULL, &ciErrNum);
    ciErrNum |= clEnqueueWriteBuffer(cqCommandQueue,d_invViewMatrix,CL_FALSE, 0,12*sizeof(float), invViewMatrix, 0, 0, 0);

    gridSize[0] = width;
    gridSize[1] = height;

    ciErrNum |= clSetKernelArg(ckKernel, 0, sizeof(cl_mem), (void *) &pbo_cl);
    ciErrNum |= clSetKernelArg(ckKernel, 1, sizeof(unsigned int), &width);
    ciErrNum |= clSetKernelArg(ckKernel, 2, sizeof(unsigned int), &height);
    ciErrNum |= clSetKernelArg(ckKernel, 3, sizeof(float), &density);
    ciErrNum |= clSetKernelArg(ckKernel, 4, sizeof(float), &brightness);
    ciErrNum |= clSetKernelArg(ckKernel, 5, sizeof(float), &transferOffset);
    ciErrNum |= clSetKernelArg(ckKernel, 6, sizeof(float), &transferScale);
    
				printf("Error more kernel args is %i\n",ciErrNum);

    // Warmup
    int iCycles = 20;
    size_t localSize[] = {LOCAL_SIZE_X,LOCAL_SIZE_Y};
    for (int i = 0; i < iCycles ; i++)
    {
        ciErrNum |= clEnqueueNDRangeKernel(cqCommandQueue, ckKernel, 2, NULL, gridSize, localSize, 0, 0, 0);
        //oclCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);
    }
    clFinish(cqCommandQueue);
    
    // Start timer 0 and process n loops on the GPU 
    //shrDeltaT(0); 
    for (int i = 0; i < iCycles ; i++)
    {
        ciErrNum |= clEnqueueNDRangeKernel(cqCommandQueue, ckKernel, 2, NULL, gridSize, localSize, 0, 0, 0);
        //oclCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);
    }
    clFinish(cqCommandQueue);
    
    // Get elapsed time and throughput, then log to sample and master logs
    //double dAvgTime = shrDeltaT(0)/(double)iCycles;
	double dAvgTime = 0.01;
    //shrlogEx(LOGBOTH | MASTER, 0, "oclVolumeRender, Throughput = %.4f MTexels/s, Time = %.5f s, Size = %u Texels, NumDevsUsed = %u, Workgroup = %u\n", 
    //       (1.0e-6 * width * height)/dAvgTime, dAvgTime, (width * height), 1, localSize[0] * localSize[1]); 

    // Cleanup and exit
    Cleanup(EXIT_SUCCESS);
}

// Function to clean up and exit
//*****************************************************************************
void Cleanup(int iExitCode)
{
    // cleanup allocated objects
    //shrlog("\nStarting Cleanup...\n\n");
    //if(cPathAndName)free(cPathAndName);
    if(cSourceCL)free(cSourceCL);
	if(ckKernel)clReleaseKernel(ckKernel);  
    if(cpProgram)clReleaseProgram(cpProgram);
    if(volumeSamplerLinear)clReleaseSampler(volumeSamplerLinear);
    if(volumeSamplerNearest)clReleaseSampler(volumeSamplerNearest);
    if(transferFuncSampler)clReleaseSampler(transferFuncSampler);
    if(d_volumeArray)clReleaseMemObject(d_volumeArray);
    if(d_transferFuncArray)clReleaseMemObject(d_transferFuncArray);
    if(pbo_cl)clReleaseMemObject(pbo_cl);    
    if(d_invViewMatrix)clReleaseMemObject(d_invViewMatrix);    
    if(cqCommandQueue)clReleaseCommandQueue(cqCommandQueue);
    if(cxGPUContext)clReleaseContext(cxGPUContext);

    glDeleteBuffersARB(1, &pbo);

    //shrQAFinish2(bQATest, *pArgc, (const char **)pArgv, (iExitCode == 0) ? QA_PASSED : QA_FAILED); 

    // finalize logs and leave
    if (bNoPrompt || bQATest)
    {
        //shrlogEx(LOGBOTH | CLOSELOG, 0, "%s Exiting...\n", cExecutableName);
    }
    else 
    {
        //shrlogEx(LOGBOTH | CLOSELOG, 0, "%s Exiting...\nPress <Enter> to Quit\n", cExecutableName);
        #ifdef WIN32
            getchar();
        #endif
    }
    exit (iExitCode);
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

void AllocateMemory(float *& pointer, int size, void** pointers, int& Npointers, nifti_image** niftiImages, int Nimages, const char* variable)
{
    pointer = (float*)malloc(size);
    if (pointer != NULL)
    {
        pointers[Npointers] = (void*)pointer;
        Npointers++;
    }
    else
    {
        printf("Could not allocate host memory for variable %s ! \n",variable);        
		FreeAllMemory(pointers, Npointers);
		FreeAllNiftiImages(niftiImages, Nimages);
		exit(EXIT_FAILURE);        
    }
}
    
void AllocateMemoryFloat2(cl_float2 *& pointer, int size, void** pointers, int& Npointers, nifti_image** niftiImages, int Nimages, const char* variable)
{
    pointer = (cl_float2*)malloc(size);
    if (pointer != NULL)
    {
        pointers[Npointers] = (void*)pointer;
        Npointers++;
    }
    else
    {
        printf("Could not allocate host memory for variable %s ! \n",variable);        
		FreeAllMemory(pointers, Npointers);
		FreeAllNiftiImages(niftiImages, Nimages);
		exit(EXIT_FAILURE);        
    }
}



int main(int argc, char **argv)
{
    //-----------------------
    // Input pointers
    
    float           *h_Volume;
      
    void*           allMemoryPointers[500];
    int             numberOfMemoryPointers = 0;

	nifti_image*	allNiftiImages[500];
	int				numberOfNiftiImages = 0;           
    
    int             OPENCL_PLATFORM = 0;
    int             OPENCL_DEVICE = 0;

    // Size parameters

    int             DATA_H, DATA_W, DATA_D;
	float			VOXEL_SIZE_X, VOXEL_SIZE_Y, VOXEL_SIZE_Z;
        
    //---------------------    
    
    /* Input arguments */
    FILE *fp = NULL; 
    
    // No inputs, so print help text
    if (argc == 1)
    {   
		printf("\nThe function renders a volume using direct volume rendering.\n\n");     
        printf("Usage:\n\n");
        printf("RenderVolume volume.nii [options]\n\n");
        printf("Options:\n\n");
        printf(" -platform                  The OpenCL platform to use (default 0) \n");
        printf(" -device                    The OpenCL device to use for the specificed platform (default 0) \n");
        printf(" -verbose                   Print extra stuff (default false) \n");
        printf(" -debug                     Get additional debug information saved as nifti files (default no). Warning: This will use a lot of extra memory! \n");
        printf("\n\n");
        
        return EXIT_SUCCESS;
    }
    // Try to open files
    else if (argc > 1)
    {        
        fp = fopen(argv[1],"r");
        if (fp == NULL)
        {
            printf("Could not open file %s !\n",argv[1]);
            return EXIT_FAILURE;
        }
        fclose(fp);        
    }
    
    // Loop over additional inputs
    int i = 2;
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
    	 
    // Read first volume 
	// -----------------------------------

    nifti_image *inputVolume = nifti_image_read(argv[1],1);
    
    if (inputVolume == NULL)
    {
        printf("Could not open volume to render!\n");
        return EXIT_FAILURE;
    }
	allNiftiImages[numberOfNiftiImages] = inputVolume;
	numberOfNiftiImages++;
  	
	// -----------------------------------

    // Get data dimensions from input data
    DATA_W = inputVolume->nx;
    DATA_H = inputVolume->ny;
    DATA_D = inputVolume->nz;    
    
    // Get voxel sizes from input data
    VOXEL_SIZE_X = inputVolume->dx;
    VOXEL_SIZE_Y = inputVolume->dy;
    VOXEL_SIZE_Z = inputVolume->dz;
                             
    int VOLUME_SIZE = DATA_W * DATA_H * DATA_D * sizeof(float);
    
    // Print some info
    printf("Authored by K.A. Eklund \n");
    printf("Volume size: %i x %i x %i \n",  DATA_W, DATA_H, DATA_D);
    printf("Volume voxel size: %f x %f x %f mm \n", VOXEL_SIZE_X, VOXEL_SIZE_Y, VOXEL_SIZE_Z);    
        
    // ------------------------------------------------
    
    // Allocate memory on the host 
	AllocateMemory(h_Volume, VOLUME_SIZE, allMemoryPointers, numberOfMemoryPointers, allNiftiImages, numberOfNiftiImages, "INPUT_VOLUME");
        
    // Convert data to floats
    if ( inputVolume->datatype == DT_SIGNED_SHORT )
    {
        short int *p = (short int*)inputVolume->data;
    
        for (int i = 0; i < DATA_W * DATA_H * DATA_D; i++)
        {
            h_Volume[i] = (float)p[i];
        }
    }
    else if ( inputVolume->datatype == DT_UINT8 )
    {
        unsigned char *p = (unsigned char*)inputVolume->data;
    
        for (int i = 0; i < DATA_W * DATA_H * DATA_D; i++)
        {
            h_Volume[i] = (float)p[i];
        }
    }
    else if ( inputVolume->datatype == DT_FLOAT )
    {
        float *p = (float*)inputVolume->data;
    
        for (int i = 0; i < DATA_W * DATA_H * DATA_D; i++)
        {
            h_Volume[i] = p[i];
        }
    }
    else
    {
        printf("Unknown data type in input volume, aborting!\n");
        FreeAllMemory(allMemoryPointers,numberOfMemoryPointers);
		FreeAllNiftiImages(allNiftiImages,numberOfNiftiImages);
        return EXIT_FAILURE;
    }
       
    //------------------------
    
    // First initialize OpenGL context, so we can properly setup the OpenGL / OpenCL interop.
    InitGL(&argc, argv); 

    // Create OpenCL context, get device info, select device, select options for image/texture and CL-GL interop
    //createCLContext(argc, (const char**)argv);
	createCLContext(OPENCL_PLATFORM, OPENCL_DEVICE);

	cl_int error;
	

    // create a command-queue
    cqCommandQueue = clCreateCommandQueue(cxGPUContext, cdDevices[uiDeviceUsed], 0, &ciErrNum);
    //oclCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);

	//clGetDeviceInfo(cdDevices[uiDeviceUsed], CL_DEVICE_IMAGE_SUPPORT, sizeof(g_bImageSupport), &g_bImageSupport, NULL);
	g_bImageSupport = true;

	// Read the kernel code from file
	std::fstream kernelFile("volumeRender.cl",std::ios::in);

	std::ostringstream oss;
	oss << kernelFile.rdbuf();
	std::string src = oss.str();
	const char *srcstr = src.c_str();


	// Create program and build the code for the selected device
	cpProgram = clCreateProgramWithSource(cxGPUContext, 1, (const char**)&srcstr, NULL, &error);
	printf("Create program with source error is %i \n",error);
    
    // build the program
    std::string buildOpts = "-cl-fast-relaxed-math";
    buildOpts += g_bImageSupport ? " -DIMAGE_SUPPORT" : "";
    ciErrNum = clBuildProgram(cpProgram, 0, NULL, buildOpts.c_str(), NULL, NULL);
	printf("Build program error is %i \n",error);
    if (ciErrNum != CL_SUCCESS)
    {
		printf("Building failed!\n");
        // write out standard error, Build Log and PTX, then cleanup and return error
        //shrlogEx(LOGBOTH | ERRORMSG, ciErrNum, STDERROR);
        //oclLogBuildInfo(cpProgram, oclGetFirstDev(cxGPUContext));
        //oclLogPtx(cpProgram, oclGetFirstDev(cxGPUContext), "oclVolumeRender.ptx");
        Cleanup(EXIT_FAILURE); 
    }

    // create the kernel
    ckKernel = clCreateKernel(cpProgram, "d_render", &error);
	printf("Create kernel error is %i \n",error);
    //oclCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);

    // Init OpenCL
    initCLVolume(h_Volume, DATA_W, DATA_H, DATA_D);

    // init timer 1 for fps measurement 
    //shrDeltaT(1);  
    
    // Create buffers and textures, 
    // and then start main GLUT rendering loop for processing and rendering, 
	// or otherwise run No-GL Q/A test sequence
    initPixelBuffer();
    glutMainLoop();

    // Normally unused return path
    Cleanup(EXIT_SUCCESS);

    // Free all memory
    FreeAllMemory(allMemoryPointers,numberOfMemoryPointers);
    FreeAllNiftiImages(allNiftiImages,numberOfNiftiImages);
       
    return EXIT_SUCCESS;
}


