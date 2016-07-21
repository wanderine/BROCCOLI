/*
    BROCCOLI: Software for Fast fMRI Analysis on Many-Core CPUs and GPUs
    Copyright (C) <2013>  Anders Eklund, andek034@gmail.com

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
    
    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, 
    INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR 
    PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE 
    FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR 
    OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER 
    DEALINGS IN THE SOFTWARE.
*/

// Help functions
int Calculate2DIndex(int x, int y, int DATA_W)
{
	return x + y * DATA_W;
}

int Calculate3DIndex(int x, int y, int z, int DATA_W, int DATA_H)
{
	return x + y * DATA_W + z * DATA_W * DATA_H;
}

int Calculate4DIndex(int x, int y, int z, int t, int DATA_W, int DATA_H, int DATA_D)
{
	return x + y * DATA_W + z * DATA_W * DATA_H + t * DATA_W * DATA_H * DATA_D;
}




void ReadSphere(__local float* Volume,
                __global const float* Volumes,
                int x,
                int y,
                int z,
                int t,
                int3 tIdx,
                int DATA_W,
                int DATA_H,
                int DATA_D)
{
    
    Volume[Calculate3DIndex(tIdx.x,tIdx.y,tIdx.z, 16, 16)] = 0.0f;
    Volume[Calculate3DIndex(tIdx.x + 8,tIdx.y,tIdx.z, 16, 16)] = 0.0f;
    Volume[Calculate3DIndex(tIdx.x,tIdx.y + 8,tIdx.z, 16, 16)] = 0.0f;
    Volume[Calculate3DIndex(tIdx.x + 8,tIdx.y + 8,tIdx.z, 16, 16)] = 0.0f;
    Volume[Calculate3DIndex(tIdx.x,tIdx.y,tIdx.z + 8, 16, 16)] = 0.0f;
    Volume[Calculate3DIndex(tIdx.x + 8,tIdx.y,tIdx.z + 8, 16, 16)] = 0.0f;
    Volume[Calculate3DIndex(tIdx.x,tIdx.y + 8,tIdx.z + 8, 16, 16)] = 0.0f;
    Volume[Calculate3DIndex(tIdx.x + 8,tIdx.y + 8,tIdx.z + 8, 16, 16)] = 0.0f;
    
    
    // X, Y, Z
    if ( ((x - 4) < DATA_W) && ((x - 4) >= 0) && ((y - 4) < DATA_H) && ((y - 4) >= 0) && ((z - 4) < DATA_D) && ((z - 4) >= 0) )
    {
        Volume[Calculate3DIndex(tIdx.x,tIdx.y,tIdx.z, 16, 16)] = Volumes[Calculate4DIndex(x - 4,y - 4,z - 4,t,DATA_W, DATA_H, DATA_D)];
    }
    
    // X + 8, Y, Z
    if ( ((x + 4) < DATA_W) && ((y - 4) < DATA_H) && ((y - 4) >= 0) && ((z - 4) < DATA_D) && ((z - 4) >= 0) )
    {
        Volume[Calculate3DIndex(tIdx.x + 8,tIdx.y,tIdx.z, 16, 16)] = Volumes[Calculate4DIndex(x + 4,y - 4,z - 4,t,DATA_W, DATA_H, DATA_D)];
    }
    
    // X, Y + 8, Z
    if ( ((x - 4) < DATA_W) && ((x - 4) >= 0) && ((y + 4) < DATA_H) && ((z - 4) < DATA_D) && ((z - 4) >= 0) )
    {
        Volume[Calculate3DIndex(tIdx.x,tIdx.y + 8,tIdx.z, 16, 16)] = Volumes[Calculate4DIndex(x - 4,y + 4,z - 4,t,DATA_W, DATA_H, DATA_D)];
    }
    
    // X + 8, Y + 8, Z
    if ( ((x + 4) < DATA_W) && ((y + 4) < DATA_H) && ((z - 4) < DATA_D) && ((z - 4) >= 0) )
    {
        Volume[Calculate3DIndex(tIdx.x + 8,tIdx.y + 8,tIdx.z, 16, 16)]= Volumes[Calculate4DIndex(x + 4,y + 4,z - 4,t,DATA_W, DATA_H, DATA_D)];
    }
    
    
    
    // X, Y, Z + 8
    if ( ((x - 4) < DATA_W) && ((x - 4) >= 0) && ((y - 4) < DATA_H) && ((y - 4) >= 0) && ((z + 4) < DATA_D) )
    {
        Volume[Calculate3DIndex(tIdx.x,tIdx.y,tIdx.z + 8, 16, 16)] = Volumes[Calculate4DIndex(x - 4,y - 4,z + 4,t,DATA_W, DATA_H, DATA_D)];
    }
    
    // X + 8, Y, Z + 8
    if ( ((x + 4) < DATA_W) && ((y - 4) < DATA_H) && ((y - 4) >= 0) && ((z + 4) < DATA_D) )
    {
        Volume[Calculate3DIndex(tIdx.x + 8,tIdx.y,tIdx.z + 8, 16, 16)] = Volumes[Calculate4DIndex(x + 4,y - 4,z + 4,t,DATA_W, DATA_H, DATA_D)];
    }
    
    
    // X, Y + 8, Z + 8
    if ( ((x - 4) < DATA_W) && ((x - 4) >= 0) && ((y + 4) < DATA_H) && ((z + 4) < DATA_D) )
    {
        Volume[Calculate3DIndex(tIdx.x,tIdx.y + 8,tIdx.z + 8, 16, 16)] = Volumes[Calculate4DIndex(x - 4,y + 4,z + 4,t,DATA_W, DATA_H, DATA_D)];
    }
    
    // X + 8, Y + 8, Z + 8
    if ( ((x + 4) < DATA_W) && ((y + 4) < DATA_H) && ((z + 4) < DATA_D) )
    {
        Volume[Calculate3DIndex(tIdx.x + 8,tIdx.y + 8,tIdx.z + 8, 16, 16)] = Volumes[Calculate4DIndex(x + 4,y + 4,z + 4,t,DATA_W, DATA_H, DATA_D)];
    }
}




__kernel void CalculateStatisticalMapSearchlight_(__global float* Classifier_Performance,
                                                 __global const float* Volumes,
                                                 __global const float* Mask,
                                                 __constant float* c_d,
                                                 __constant float* c_Correct_Classes,
                                                 __private int DATA_W,
                                                 __private int DATA_H,
                                                 __private int DATA_D,
                                                 __private int NUMBER_OF_VOLUMES,
                                                 __private float n,
                                                 __private int EPOCS)

{
    int x = get_global_id(0);
    int y = get_global_id(1);
    int z = get_global_id(2);
    
    int3 tIdx = {get_local_id(0), get_local_id(1), get_local_id(2)};
    
    if (x >= DATA_W || y >= DATA_H || z >= DATA_D)
        return;
    
    if ( Mask[Calculate3DIndex(x,y,z,DATA_W,DATA_H)] != 1.0f )
    {
        Classifier_Performance[Calculate3DIndex(x,y,z,DATA_W,DATA_H)] = 0.0f;
        return;
    }
    
    __local float l_Volume[16][16][16];    // z, y, x
    
    int classification_performance = 0;
	
    // Leave one out cross validation
    for (int validation = 0; validation < NUMBER_OF_VOLUMES; validation++)
    {
        float weights[20];
        
        weights[0]  = 0.0f;
        weights[1]  = 0.0f;
        weights[2]  = 0.0f;
        weights[3]  = 0.0f;
        weights[4]  = 0.0f;
        weights[5]  = 0.0f;
        weights[6]  = 0.0f;
        weights[7]  = 0.0f;
        weights[8]  = 0.0f;
        weights[9]  = 0.0f;
        weights[10] = 0.0f;
        weights[11] = 0.0f;
        weights[12] = 0.0f;
        weights[13] = 0.0f;
        weights[14] = 0.0f;
        weights[15] = 0.0f;
        weights[16] = 0.0f;
        weights[17] = 0.0f;
        weights[18] = 0.0f;
        weights[19] = 0.0f;
        
        // Do training for a number of iterations
        for (int epoc = 0; epoc < EPOCS; epoc++)
        {
            float gradient[20];
            
            gradient[0] = 0.0f;
            gradient[1] = 0.0f;
            gradient[2] = 0.0f;
            gradient[3] = 0.0f;
            gradient[4] = 0.0f;
            gradient[5] = 0.0f;
            gradient[6] = 0.0f;
            gradient[7] = 0.0f;
            gradient[8] = 0.0f;
            gradient[9] = 0.0f;
            gradient[10] = 0.0f;
            gradient[11] = 0.0f;
            gradient[12] = 0.0f;
            gradient[13] = 0.0f;
            gradient[14] = 0.0f;
            gradient[15] = 0.0f;
            gradient[16] = 0.0f;
            gradient[17] = 0.0f;
            gradient[18] = 0.0f;
            gradient[19] = 0.0f;
            
            for (int t = 0; t < NUMBER_OF_VOLUMES; t++)
            {
                // Skip training with validation time point
                if (t == validation)
                {
                    continue;
                }
                
                float s;
                
                // Classification for current timepoint
                ReadSphere((__local float*)l_Volume, Volumes, x, y, z, t, tIdx, DATA_W, DATA_H, DATA_D);
                
                // Make sure all threads have written to local memory
                barrier(CLK_LOCAL_MEM_FENCE);
                
                // Make classification
                s =  weights[0] * 1.0f;
                
                // z - 1
                s += weights[1] * l_Volume[tIdx.z + 4 - 1][tIdx.y + 4][tIdx.x + 4 - 1]; 		//
                s += weights[2] * l_Volume[tIdx.z + 4 - 1][tIdx.y + 4 - 1][tIdx.x + 4]; 		//
                s += weights[3] * l_Volume[tIdx.z + 4 - 1][tIdx.y + 4][tIdx.x + 4]; 			// center pixel
                s += weights[4] * l_Volume[tIdx.z + 4 - 1][tIdx.y + 4 + 1][tIdx.x + 4]; 		//
                s += weights[5] * l_Volume[tIdx.z + 4 - 1][tIdx.y + 4][tIdx.x + 4 + 1]; 		//
                
                // z
                s += weights[6] * l_Volume[tIdx.z + 4][tIdx.y + 4 - 1][tIdx.x + 4 - 1]; 	//
                s += weights[7] * l_Volume[tIdx.z + 4][tIdx.y + 4 - 1][tIdx.x + 4]; 		//
                s += weights[8] * l_Volume[tIdx.z + 4][tIdx.y + 4 - 1][tIdx.x + 4 + 1]; 	//
                s += weights[9] * l_Volume[tIdx.z + 4][tIdx.y + 4][tIdx.x + 4 - 1]; 		//
                s += weights[10] * l_Volume[tIdx.z + 4][tIdx.y + 4][tIdx.x + 4]; 			//	center pixel
                s += weights[11] * l_Volume[tIdx.z + 4][tIdx.y + 4][tIdx.x + 4 + 1]; 		//
                s += weights[12] * l_Volume[tIdx.z + 4][tIdx.y + 4 + 1][tIdx.x + 4 - 1]; 	//
                s += weights[13] * l_Volume[tIdx.z + 4][tIdx.y + 4 + 1][tIdx.x + 4]; 		//
                s += weights[14] * l_Volume[tIdx.z + 4][tIdx.y + 4 + 1][tIdx.x + 4 + 1]; 	//
                
                // z + 1
                s += weights[15] * l_Volume[tIdx.z + 4 + 1][tIdx.y + 4][tIdx.x + 4 - 1]; 		//
                s += weights[16] * l_Volume[tIdx.z + 4 + 1][tIdx.y + 4 - 1][tIdx.x + 4]; 		//
                s += weights[17] * l_Volume[tIdx.z + 4 + 1][tIdx.y + 4][tIdx.x + 4]; 			// center pixel
                s += weights[18] * l_Volume[tIdx.z + 4 + 1][tIdx.y + 4 + 1][tIdx.x + 4]; 		//
                s += weights[19] * l_Volume[tIdx.z + 4 + 1][tIdx.y + 4][tIdx.x + 4 + 1]; 		//
                
                // Calculate contribution to gradient
                gradient[0] += (s - c_d[t]) * 1.0f;
                
                // z - 1
                gradient[1]  += (s - c_d[t]) * l_Volume[tIdx.z + 4 - 1][tIdx.y + 4][tIdx.x + 4 - 1]; 		//
                gradient[2]  += (s - c_d[t]) * l_Volume[tIdx.z + 4 - 1][tIdx.y + 4 - 1][tIdx.x + 4]; 		//
                gradient[3]  += (s - c_d[t]) * l_Volume[tIdx.z + 4 - 1][tIdx.y + 4][tIdx.x + 4]; 			// center pixel
                gradient[4]  += (s - c_d[t]) * l_Volume[tIdx.z + 4 - 1][tIdx.y + 4 + 1][tIdx.x + 4]; 		//
                gradient[5]  += (s - c_d[t]) * l_Volume[tIdx.z + 4 - 1][tIdx.y + 4][tIdx.x + 4 + 1]; 		//
                
                // z
                gradient[6]  += (s - c_d[t]) * l_Volume[tIdx.z + 4][tIdx.y + 4 - 1][tIdx.x + 4 - 1]; 		//
                gradient[7]  += (s - c_d[t]) * l_Volume[tIdx.z + 4][tIdx.y + 4 - 1][tIdx.x + 4]; 			//
                gradient[8]  += (s - c_d[t]) * l_Volume[tIdx.z + 4][tIdx.y + 4 - 1][tIdx.x + 4 + 1]; 		//
                gradient[9]  += (s - c_d[t]) * l_Volume[tIdx.z + 4][tIdx.y + 4][tIdx.x + 4 - 1]; 			//
                gradient[10] += (s - c_d[t]) * l_Volume[tIdx.z + 4][tIdx.y + 4][tIdx.x + 4]; 				//	center pixel
                gradient[11] += (s - c_d[t]) * l_Volume[tIdx.z + 4][tIdx.y + 4][tIdx.x + 4 + 1]; 			//
                gradient[12] += (s - c_d[t]) * l_Volume[tIdx.z + 4][tIdx.y + 4 + 1][tIdx.x + 4 - 1]; 		//
                gradient[13] += (s - c_d[t]) * l_Volume[tIdx.z + 4][tIdx.y + 4 + 1][tIdx.x + 4]; 			//
                gradient[14] += (s - c_d[t]) * l_Volume[tIdx.z + 4][tIdx.y + 4 + 1][tIdx.x + 4 + 1]; 		//
                
                // z + 1
                gradient[15] += (s - c_d[t]) * l_Volume[tIdx.z + 4 + 1][tIdx.y + 4][tIdx.x + 4 - 1]; 		//
                gradient[16] += (s - c_d[t]) * l_Volume[tIdx.z + 4 + 1][tIdx.y + 4 - 1][tIdx.x + 4]; 		//
                gradient[17] += (s - c_d[t]) * l_Volume[tIdx.z + 4 + 1][tIdx.y + 4][tIdx.x + 4]; 			// center pixel
                gradient[18] += (s - c_d[t]) * l_Volume[tIdx.z + 4 + 1][tIdx.y + 4 + 1][tIdx.x + 4]; 		//
                gradient[19] += (s - c_d[t]) * l_Volume[tIdx.z + 4 + 1][tIdx.y + 4][tIdx.x + 4 + 1]; 		//
                
                
                // end for t
            }
            
            // Update weights
            weights[0] -= n/(float)NUMBER_OF_VOLUMES * gradient[0];
            weights[1] -= n/(float)NUMBER_OF_VOLUMES * gradient[1];
            weights[2] -= n/(float)NUMBER_OF_VOLUMES * gradient[2];
            weights[3] -= n/(float)NUMBER_OF_VOLUMES * gradient[3];
            weights[4] -= n/(float)NUMBER_OF_VOLUMES * gradient[4];
            weights[5] -= n/(float)NUMBER_OF_VOLUMES * gradient[5];
            weights[6] -= n/(float)NUMBER_OF_VOLUMES * gradient[6];
            weights[7] -= n/(float)NUMBER_OF_VOLUMES * gradient[7];
            weights[8] -= n/(float)NUMBER_OF_VOLUMES * gradient[8];
            weights[9] -= n/(float)NUMBER_OF_VOLUMES * gradient[9];
            weights[10] -= n/(float)NUMBER_OF_VOLUMES * gradient[10];
            weights[11] -= n/(float)NUMBER_OF_VOLUMES * gradient[11];
            weights[12] -= n/(float)NUMBER_OF_VOLUMES * gradient[12];
            weights[13] -= n/(float)NUMBER_OF_VOLUMES * gradient[13];
            weights[14] -= n/(float)NUMBER_OF_VOLUMES * gradient[14];
            weights[15] -= n/(float)NUMBER_OF_VOLUMES * gradient[15];
            weights[16] -= n/(float)NUMBER_OF_VOLUMES * gradient[16];
            weights[17] -= n/(float)NUMBER_OF_VOLUMES * gradient[17];
            weights[18] -= n/(float)NUMBER_OF_VOLUMES * gradient[18];
            weights[19] -= n/(float)NUMBER_OF_VOLUMES * gradient[19];
            
            // end for epocs
        }
        
        // Make classification on validation timepoint
        
        ReadSphere((__local float*)l_Volume, Volumes, x, y, z, validation, tIdx, DATA_W, DATA_H, DATA_D);
        
        // Make sure all threads have written to local memory
        barrier(CLK_LOCAL_MEM_FENCE);
        
        // Make classification
        float s;
        s =  weights[0] * 1.0f;
        
        // z - 1
        s += weights[1] * l_Volume[tIdx.z + 4 - 1][tIdx.y + 4][tIdx.x + 4 - 1]; 		//
        s += weights[2] * l_Volume[tIdx.z + 4 - 1][tIdx.y + 4 - 1][tIdx.x + 4]; 		//
        s += weights[3] * l_Volume[tIdx.z + 4 - 1][tIdx.y + 4][tIdx.x + 4]; 			// center pixel
        s += weights[4] * l_Volume[tIdx.z + 4 - 1][tIdx.y + 4 + 1][tIdx.x + 4]; 		//
        s += weights[5] * l_Volume[tIdx.z + 4 - 1][tIdx.y + 4][tIdx.x + 4 + 1]; 		//
        
        // z
        s += weights[6] * l_Volume[tIdx.z + 4][tIdx.y + 4 - 1][tIdx.x + 4 - 1]; 		//
        s += weights[7] * l_Volume[tIdx.z + 4][tIdx.y + 4 - 1][tIdx.x + 4]; 			//
        s += weights[8] * l_Volume[tIdx.z + 4][tIdx.y + 4 - 1][tIdx.x + 4 + 1]; 		//
        s += weights[9] * l_Volume[tIdx.z + 4][tIdx.y + 4][tIdx.x + 4 - 1]; 			//
        s += weights[10] * l_Volume[tIdx.z + 4][tIdx.y + 4][tIdx.x + 4]; 			//	center pixel
        s += weights[11] * l_Volume[tIdx.z + 4][tIdx.y + 4][tIdx.x + 4 + 1]; 		//
        s += weights[12] * l_Volume[tIdx.z + 4][tIdx.y + 4 + 1][tIdx.x + 4 - 1]; 	//
        s += weights[13] * l_Volume[tIdx.z + 4][tIdx.y + 4 + 1][tIdx.x + 4]; 		//
        s += weights[14] * l_Volume[tIdx.z + 4][tIdx.y + 4 + 1][tIdx.x + 4 + 1]; 	//
        
        // z + 1
        s += weights[15] * l_Volume[tIdx.z + 4 + 1][tIdx.y + 4][tIdx.x + 4 - 1]; 		//
        s += weights[16] * l_Volume[tIdx.z + 4 + 1][tIdx.y + 4 - 1][tIdx.x + 4]; 		//
        s += weights[17] * l_Volume[tIdx.z + 4 + 1][tIdx.y + 4][tIdx.x + 4]; 			// center pixel
        s += weights[18] * l_Volume[tIdx.z + 4 + 1][tIdx.y + 4 + 1][tIdx.x + 4]; 		//
        s += weights[19] * l_Volume[tIdx.z + 4 + 1][tIdx.y + 4][tIdx.x + 4 + 1]; 		//
        
        float classification;
        if (s > 0.0f)
        {
            classification = 0.0f;
        }
        else
        {
            classification = 1.0f;
        }
        
        if (classification == c_Correct_Classes[validation])
        {
            classification_performance++;
        }
        
        // end for validation
    }
    
    Classifier_Performance[Calculate3DIndex(x,y,z,DATA_W,DATA_H)] = (float)classification_performance / (float)NUMBER_OF_VOLUMES;    
}






__kernel void CalculateStatisticalMapSearchlight(__global float* Classifier_Performance,
                                                  __global const float* Volumes,
                                                  __global const float* Mask,
                                                  __constant float* c_d,
                                                  __constant float* c_Correct_Classes,
                                                  __private int DATA_W,
                                                  __private int DATA_H,
                                                  __private int DATA_D,
                                                  __private int NUMBER_OF_VOLUMES,
                                                  __private float n,
                                                  __private int EPOCS)

{
    int x = get_global_id(0);
    int y = get_global_id(1);
    int z = get_global_id(2);
    
    int3 tIdx = {get_local_id(0), get_local_id(1), get_local_id(2)};
    
    if (x >= DATA_W || y >= DATA_H || z >= DATA_D)
        return;
    
    if ( Mask[Calculate3DIndex(x,y,z,DATA_W,DATA_H)] != 1.0f )
    {
        Classifier_Performance[Calculate3DIndex(x,y,z,DATA_W,DATA_H)] = 0.0f;
        return;
    }
    
    if ( ((x + 1) >= DATA_W) || ((y + 1) >= DATA_H) || ((z + 1) >= DATA_D) )
    {
        Classifier_Performance[Calculate3DIndex(x,y,z,DATA_W,DATA_H)] = 0.0f;
        return;
    }
    
    
    if ( ((x - 1) < 0) || ((y - 1) < 0) || ((z - 1) < 0) )
    {
        Classifier_Performance[Calculate3DIndex(x,y,z,DATA_W,DATA_H)] = 0.0f;
        return;
    }
    
    
    int classification_performance = 0;
    
    float weights[20];

	int uncensoredVolumes = 0;

    // Leave one out cross validation
    for (int validation = 0; validation < NUMBER_OF_VOLUMES; validation++)
    {
		// Skip testing with censored volumes
        if (c_Correct_Classes[validation] == 9999.0f)
        {
            continue;
        } 

		uncensoredVolumes++;
       
        weights[0]  = 0.0f;
        weights[1]  = 0.0f;
        weights[2]  = 0.0f;
        weights[3]  = 0.0f;
        weights[4]  = 0.0f;
        weights[5]  = 0.0f;
        weights[6]  = 0.0f;
        weights[7]  = 0.0f;
        weights[8]  = 0.0f;
        weights[9]  = 0.0f;
        weights[10] = 0.0f;
        weights[11] = 0.0f;
        weights[12] = 0.0f;
        weights[13] = 0.0f;
        weights[14] = 0.0f;
        weights[15] = 0.0f;
        weights[16] = 0.0f;
        weights[17] = 0.0f;
        weights[18] = 0.0f;
        weights[19] = 0.0f;
        
        // Do training for a number of iterations
        for (int epoc = 0; epoc < EPOCS; epoc++)
        {
            float gradient[20];
            
            gradient[0] = 0.0f;
            gradient[1] = 0.0f;
            gradient[2] = 0.0f;
            gradient[3] = 0.0f;
            gradient[4] = 0.0f;
            gradient[5] = 0.0f;
            gradient[6] = 0.0f;
            gradient[7] = 0.0f;
            gradient[8] = 0.0f;
            gradient[9] = 0.0f;
            gradient[10] = 0.0f;
            gradient[11] = 0.0f;
            gradient[12] = 0.0f;
            gradient[13] = 0.0f;
            gradient[14] = 0.0f;
            gradient[15] = 0.0f;
            gradient[16] = 0.0f;
            gradient[17] = 0.0f;
            gradient[18] = 0.0f;
            gradient[19] = 0.0f;
            
            for (int t = 0; t < NUMBER_OF_VOLUMES; t++)
            {
                // Skip training with validation volume and censored volumes
                if ((t == validation) || (c_Correct_Classes[t] == 9999.0f))
                {
                    continue;
                }                                
                
                // Make classification
                float s;
                s =  weights[0] * 1.0f;
                
                float x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16, x17, x18, x19;
                
                x1 = Volumes[Calculate4DIndex(x-1,y,z-1,t,DATA_W,DATA_H,DATA_D)];
                x2 = Volumes[Calculate4DIndex(x,y-1,z-1,t,DATA_W,DATA_H,DATA_D)];
                x3 = Volumes[Calculate4DIndex(x,y,z-1,t,DATA_W,DATA_H,DATA_D)];
                x4 = Volumes[Calculate4DIndex(x,y+1,z-1,t,DATA_W,DATA_H,DATA_D)];
                x5 = Volumes[Calculate4DIndex(x+1,y,z-1,t,DATA_W,DATA_H,DATA_D)];
                
                x6 = Volumes[Calculate4DIndex(x-1,y-1,z,t,DATA_W,DATA_H,DATA_D)];
                x7 = Volumes[Calculate4DIndex(x-1,y,z,t,DATA_W,DATA_H,DATA_D)];
                x8 = Volumes[Calculate4DIndex(x-1,y+1,z,t,DATA_W,DATA_H,DATA_D)];
                x9 = Volumes[Calculate4DIndex(x,y-1,z,t,DATA_W,DATA_H,DATA_D)];
                x10 = Volumes[Calculate4DIndex(x,y,z,t,DATA_W,DATA_H,DATA_D)];
                x11 = Volumes[Calculate4DIndex(x,y+1,z,t,DATA_W,DATA_H,DATA_D)];
                x12 = Volumes[Calculate4DIndex(x+1,y-1,z,t,DATA_W,DATA_H,DATA_D)];
                x13 = Volumes[Calculate4DIndex(x+1,y,z,t,DATA_W,DATA_H,DATA_D)];
                x14 = Volumes[Calculate4DIndex(x+1,y+1,z,t,DATA_W,DATA_H,DATA_D)];
                
                x15 = Volumes[Calculate4DIndex(x-1,y,z+1,t,DATA_W,DATA_H,DATA_D)];
                x16 = Volumes[Calculate4DIndex(x,y-1,z+1,t,DATA_W,DATA_H,DATA_D)];
                x17 = Volumes[Calculate4DIndex(x,y,z+1,t,DATA_W,DATA_H,DATA_D)];
                x18 = Volumes[Calculate4DIndex(x,y+1,z+1,t,DATA_W,DATA_H,DATA_D)];
                x19 = Volumes[Calculate4DIndex(x+1,y,z+1,t,DATA_W,DATA_H,DATA_D)];
                
                // z - 1
                s += weights[1] * x1;
                s += weights[2] * x2;
                s += weights[3] * x3;
                s += weights[4] * x4;
                s += weights[5] * x5;
                
                // z
                s += weights[6] * x6;
                s += weights[7] * x7;
                s += weights[8] * x8;
                s += weights[9] * x9;
                s += weights[10] * x10;
                s += weights[11] * x11;
                s += weights[12] * x12;
                s += weights[13] * x13;
                s += weights[14] * x14;
                
                // z + 1
                s += weights[15] * x15;
                s += weights[16] * x16;
                s += weights[17] * x17;
                s += weights[18] * x18;
                s += weights[19] * x19;
                
                // Calculate contribution to gradient
                gradient[0] += (s - c_d[t]) * 1.0f;
                
                // z - 1
                gradient[1]  += (s - c_d[t]) * x1;
                gradient[2]  += (s - c_d[t]) * x2;
                gradient[3]  += (s - c_d[t]) * x3;
                gradient[4]  += (s - c_d[t]) * x4;
                gradient[5]  += (s - c_d[t]) * x5;
                
                // z
                gradient[6]  += (s - c_d[t]) * x6;
                gradient[7]  += (s - c_d[t]) * x7;
                gradient[8]  += (s - c_d[t]) * x8;
                gradient[9]  += (s - c_d[t]) * x9;
                gradient[10] += (s - c_d[t]) * x10;
                gradient[11] += (s - c_d[t]) * x11;
                gradient[12] += (s - c_d[t]) * x12;
                gradient[13] += (s - c_d[t]) * x13;
                gradient[14] += (s - c_d[t]) * x14;
                
                // z + 1
                gradient[15] += (s - c_d[t]) * x15;
                gradient[16] += (s - c_d[t]) * x16;
                gradient[17] += (s - c_d[t]) * x17;
                gradient[18] += (s - c_d[t]) * x18;
                gradient[19] += (s - c_d[t]) * x19;
                
                // end for t
            }
            
            // Update weights
            weights[0] -= n/(float)NUMBER_OF_VOLUMES * gradient[0];
            weights[1] -= n/(float)NUMBER_OF_VOLUMES * gradient[1];
            weights[2] -= n/(float)NUMBER_OF_VOLUMES * gradient[2];
            weights[3] -= n/(float)NUMBER_OF_VOLUMES * gradient[3];
            weights[4] -= n/(float)NUMBER_OF_VOLUMES * gradient[4];
            weights[5] -= n/(float)NUMBER_OF_VOLUMES * gradient[5];
            weights[6] -= n/(float)NUMBER_OF_VOLUMES * gradient[6];
            weights[7] -= n/(float)NUMBER_OF_VOLUMES * gradient[7];
            weights[8] -= n/(float)NUMBER_OF_VOLUMES * gradient[8];
            weights[9] -= n/(float)NUMBER_OF_VOLUMES * gradient[9];
            weights[10] -= n/(float)NUMBER_OF_VOLUMES * gradient[10];
            weights[11] -= n/(float)NUMBER_OF_VOLUMES * gradient[11];
            weights[12] -= n/(float)NUMBER_OF_VOLUMES * gradient[12];
            weights[13] -= n/(float)NUMBER_OF_VOLUMES * gradient[13];
            weights[14] -= n/(float)NUMBER_OF_VOLUMES * gradient[14];
            weights[15] -= n/(float)NUMBER_OF_VOLUMES * gradient[15];
            weights[16] -= n/(float)NUMBER_OF_VOLUMES * gradient[16];
            weights[17] -= n/(float)NUMBER_OF_VOLUMES * gradient[17];
            weights[18] -= n/(float)NUMBER_OF_VOLUMES * gradient[18];
            weights[19] -= n/(float)NUMBER_OF_VOLUMES * gradient[19];
        
            // end for epocs
        }
        
        // Make classification
        float s;
        s =  weights[0] * 1.0f;
        
        float x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16, x17, x18, x19;
        
        x1 = Volumes[Calculate4DIndex(x-1,y,z-1,validation,DATA_W,DATA_H,DATA_D)];
        x2 = Volumes[Calculate4DIndex(x,y-1,z-1,validation,DATA_W,DATA_H,DATA_D)];
        x3 = Volumes[Calculate4DIndex(x,y,z-1,validation,DATA_W,DATA_H,DATA_D)];
        x4 = Volumes[Calculate4DIndex(x,y+1,z-1,validation,DATA_W,DATA_H,DATA_D)];
        x5 = Volumes[Calculate4DIndex(x+1,y,z-1,validation,DATA_W,DATA_H,DATA_D)];
        
        x6 = Volumes[Calculate4DIndex(x-1,y-1,z,validation,DATA_W,DATA_H,DATA_D)];
        x7 = Volumes[Calculate4DIndex(x-1,y,z,validation,DATA_W,DATA_H,DATA_D)];
        x8 = Volumes[Calculate4DIndex(x-1,y+1,z,validation,DATA_W,DATA_H,DATA_D)];
        x9 = Volumes[Calculate4DIndex(x,y-1,z,validation,DATA_W,DATA_H,DATA_D)];
        x10 = Volumes[Calculate4DIndex(x,y,z,validation,DATA_W,DATA_H,DATA_D)];
        x11 = Volumes[Calculate4DIndex(x,y+1,z,validation,DATA_W,DATA_H,DATA_D)];
        x12 = Volumes[Calculate4DIndex(x+1,y-1,z,validation,DATA_W,DATA_H,DATA_D)];
        x13 = Volumes[Calculate4DIndex(x+1,y,z,validation,DATA_W,DATA_H,DATA_D)];
        x14 = Volumes[Calculate4DIndex(x+1,y+1,z,validation,DATA_W,DATA_H,DATA_D)];
        
        x15 = Volumes[Calculate4DIndex(x-1,y,z+1,validation,DATA_W,DATA_H,DATA_D)];
        x16 = Volumes[Calculate4DIndex(x,y-1,z+1,validation,DATA_W,DATA_H,DATA_D)];
        x17 = Volumes[Calculate4DIndex(x,y,z+1,validation,DATA_W,DATA_H,DATA_D)];
        x18 = Volumes[Calculate4DIndex(x,y+1,z+1,validation,DATA_W,DATA_H,DATA_D)];
        x19 = Volumes[Calculate4DIndex(x+1,y,z+1,validation,DATA_W,DATA_H,DATA_D)];
        
        // z - 1
        s += weights[1] * x1;
        s += weights[2] * x2;
        s += weights[3] * x3;
        s += weights[4] * x4;
        s += weights[5] * x5;
        
        // z
        s += weights[6] * x6;
        s += weights[7] * x7;
        s += weights[8] * x8;
        s += weights[9] * x9;
        s += weights[10] * x10;
        s += weights[11] * x11;
        s += weights[12] * x12;
        s += weights[13] * x13;
        s += weights[14] * x14;
        
        // z + 1
        s += weights[15] * x15;
        s += weights[16] * x16;
        s += weights[17] * x17;
        s += weights[18] * x18;
        s += weights[19] * x19;
        
        float classification;
        if (s > 0.0f)
        {
            classification = 0.0f;
        }
        else
        {
            classification = 1.0f;
        }
        
        if (classification == c_Correct_Classes[validation])
        {
            classification_performance++;
        }
        
        // end for validation
    }
    
    Classifier_Performance[Calculate3DIndex(x,y,z,DATA_W,DATA_H)] = (float)classification_performance / (float)uncensoredVolumes;
}






__kernel void CalculateStatisticalMapSearchlight___(__global float* Classifier_Performance,
                                                  __global const float* Volumes,
                                                  __global const float* Mask,
                                                  __constant float* c_d,
                                                  __constant float* c_Correct_Classes,
                                                  __private int DATA_W,
                                                  __private int DATA_H,
                                                  __private int DATA_D,
                                                  __private int NUMBER_OF_VOLUMES,

                                                  __private float n,
                                                  __private int EPOCS)

{
    int x = get_global_id(0);
    int y = get_global_id(1);
    int z = get_global_id(2);
    
    int3 tIdx = {get_local_id(0), get_local_id(1), get_local_id(2)};
    
    if (x >= DATA_W || y >= DATA_H || z >= DATA_D)
        return;
    
    if ( Mask[Calculate3DIndex(x,y,z,DATA_W,DATA_H)] != 1.0f )
    {
        Classifier_Performance[Calculate3DIndex(x,y,z,DATA_W,DATA_H)] = 0.0f;
        return;
    }
    
    if ( ((x + 1) >= DATA_W) || ((y + 1) >= DATA_H) || ((z + 1) >= DATA_D) )
    {
        Classifier_Performance[Calculate3DIndex(x,y,z,DATA_W,DATA_H)] = 0.0f;
        return;
    }
    
    
    if ( ((x - 1) < 0) || ((y - 1) < 0) || ((z - 1) < 0) )
    {
        Classifier_Performance[Calculate3DIndex(x,y,z,DATA_W,DATA_H)] = 0.0f;
        return;
    }
    
    
    int classification_performance = 0;
    

	// 
	// Training
	//

    float weights[20];
       
    weights[0]  = 0.0f;
    weights[1]  = 0.0f;
    weights[2]  = 0.0f;
    weights[3]  = 0.0f;
    weights[4]  = 0.0f;
    weights[5]  = 0.0f;
    weights[6]  = 0.0f;
    weights[7]  = 0.0f;
    weights[8]  = 0.0f;
    weights[9]  = 0.0f;
    weights[10] = 0.0f;
    weights[11] = 0.0f;
    weights[12] = 0.0f;
    weights[13] = 0.0f;
    weights[14] = 0.0f;
    weights[15] = 0.0f;
    weights[16] = 0.0f;
    weights[17] = 0.0f;
    weights[18] = 0.0f;
    weights[19] = 0.0f;
        
    // Do training for a number of iterations
    for (int epoc = 0; epoc < EPOCS; epoc++)
    {
        float gradient[20];
            
        gradient[0] = 0.0f;
        gradient[1] = 0.0f;
        gradient[2] = 0.0f;
        gradient[3] = 0.0f;
        gradient[4] = 0.0f;
        gradient[5] = 0.0f;
        gradient[6] = 0.0f;
        gradient[7] = 0.0f;
        gradient[8] = 0.0f;
        gradient[9] = 0.0f;
        gradient[10] = 0.0f;
        gradient[11] = 0.0f;
        gradient[12] = 0.0f;
        gradient[13] = 0.0f;
        gradient[14] = 0.0f;
        gradient[15] = 0.0f;
        gradient[16] = 0.0f;
        gradient[17] = 0.0f;
        gradient[18] = 0.0f;
        gradient[19] = 0.0f;
            
        for (int t = 0; t < NUMBER_OF_VOLUMES / 2; t++)
        {
   			// Ignore censored volumes
			if (c_Correct_Classes[t] == 9999.0f)
			{
				continue;
			}
             
            // Make classification
            float s;
            s =  weights[0] * 1.0f;
                
            float x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16, x17, x18, x19;
                
            x1 = Volumes[Calculate4DIndex(x-1,y,z-1,t,DATA_W,DATA_H,DATA_D)];
            x2 = Volumes[Calculate4DIndex(x,y-1,z-1,t,DATA_W,DATA_H,DATA_D)];
            x3 = Volumes[Calculate4DIndex(x,y,z-1,t,DATA_W,DATA_H,DATA_D)];
            x4 = Volumes[Calculate4DIndex(x,y+1,z-1,t,DATA_W,DATA_H,DATA_D)];
            x5 = Volumes[Calculate4DIndex(x+1,y,z-1,t,DATA_W,DATA_H,DATA_D)];
                
            x6 = Volumes[Calculate4DIndex(x-1,y-1,z,t,DATA_W,DATA_H,DATA_D)];
            x7 = Volumes[Calculate4DIndex(x-1,y,z,t,DATA_W,DATA_H,DATA_D)];
            x8 = Volumes[Calculate4DIndex(x-1,y+1,z,t,DATA_W,DATA_H,DATA_D)];
            x9 = Volumes[Calculate4DIndex(x,y-1,z,t,DATA_W,DATA_H,DATA_D)];
            x10 = Volumes[Calculate4DIndex(x,y,z,t,DATA_W,DATA_H,DATA_D)];
            x11 = Volumes[Calculate4DIndex(x,y+1,z,t,DATA_W,DATA_H,DATA_D)];
            x12 = Volumes[Calculate4DIndex(x+1,y-1,z,t,DATA_W,DATA_H,DATA_D)];
            x13 = Volumes[Calculate4DIndex(x+1,y,z,t,DATA_W,DATA_H,DATA_D)];
            x14 = Volumes[Calculate4DIndex(x+1,y+1,z,t,DATA_W,DATA_H,DATA_D)];
                
            x15 = Volumes[Calculate4DIndex(x-1,y,z+1,t,DATA_W,DATA_H,DATA_D)];
            x16 = Volumes[Calculate4DIndex(x,y-1,z+1,t,DATA_W,DATA_H,DATA_D)];
            x17 = Volumes[Calculate4DIndex(x,y,z+1,t,DATA_W,DATA_H,DATA_D)];
            x18 = Volumes[Calculate4DIndex(x,y+1,z+1,t,DATA_W,DATA_H,DATA_D)];
            x19 = Volumes[Calculate4DIndex(x+1,y,z+1,t,DATA_W,DATA_H,DATA_D)];
                
            // z - 1
            s += weights[1] * x1;
            s += weights[2] * x2;
            s += weights[3] * x3;
            s += weights[4] * x4;
            s += weights[5] * x5;
                
            // z
            s += weights[6] * x6;
            s += weights[7] * x7;
            s += weights[8] * x8;
            s += weights[9] * x9;
            s += weights[10] * x10;
            s += weights[11] * x11;
            s += weights[12] * x12;
            s += weights[13] * x13;
            s += weights[14] * x14;
                
            // z + 1
            s += weights[15] * x15;
            s += weights[16] * x16;
            s += weights[17] * x17;
            s += weights[18] * x18;
            s += weights[19] * x19;
                
            // Calculate contribution to gradient
            gradient[0] += (s - c_d[t]) * 1.0f;
                
            // z - 1
            gradient[1]  += (s - c_d[t]) * x1;
            gradient[2]  += (s - c_d[t]) * x2;
            gradient[3]  += (s - c_d[t]) * x3;
            gradient[4]  += (s - c_d[t]) * x4;
            gradient[5]  += (s - c_d[t]) * x5;
                
            // z
            gradient[6]  += (s - c_d[t]) * x6;
            gradient[7]  += (s - c_d[t]) * x7;
            gradient[8]  += (s - c_d[t]) * x8;
            gradient[9]  += (s - c_d[t]) * x9;
            gradient[10] += (s - c_d[t]) * x10;
            gradient[11] += (s - c_d[t]) * x11;
            gradient[12] += (s - c_d[t]) * x12;
            gradient[13] += (s - c_d[t]) * x13;
            gradient[14] += (s - c_d[t]) * x14;
                
            // z + 1
            gradient[15] += (s - c_d[t]) * x15;
            gradient[16] += (s - c_d[t]) * x16;
            gradient[17] += (s - c_d[t]) * x17;
            gradient[18] += (s - c_d[t]) * x18;
            gradient[19] += (s - c_d[t]) * x19;
                
            // end for t
        }
            
        // Update weights
        weights[0] -= n/(float)NUMBER_OF_VOLUMES * gradient[0];
        weights[1] -= n/(float)NUMBER_OF_VOLUMES * gradient[1];
        weights[2] -= n/(float)NUMBER_OF_VOLUMES * gradient[2];
        weights[3] -= n/(float)NUMBER_OF_VOLUMES * gradient[3];
        weights[4] -= n/(float)NUMBER_OF_VOLUMES * gradient[4];
        weights[5] -= n/(float)NUMBER_OF_VOLUMES * gradient[5];
        weights[6] -= n/(float)NUMBER_OF_VOLUMES * gradient[6];
        weights[7] -= n/(float)NUMBER_OF_VOLUMES * gradient[7];
        weights[8] -= n/(float)NUMBER_OF_VOLUMES * gradient[8];
        weights[9] -= n/(float)NUMBER_OF_VOLUMES * gradient[9];
        weights[10] -= n/(float)NUMBER_OF_VOLUMES * gradient[10];
        weights[11] -= n/(float)NUMBER_OF_VOLUMES * gradient[11];
        weights[12] -= n/(float)NUMBER_OF_VOLUMES * gradient[12];
        weights[13] -= n/(float)NUMBER_OF_VOLUMES * gradient[13];
        weights[14] -= n/(float)NUMBER_OF_VOLUMES * gradient[14];
        weights[15] -= n/(float)NUMBER_OF_VOLUMES * gradient[15];
        weights[16] -= n/(float)NUMBER_OF_VOLUMES * gradient[16];
        weights[17] -= n/(float)NUMBER_OF_VOLUMES * gradient[17];
        weights[18] -= n/(float)NUMBER_OF_VOLUMES * gradient[18];
        weights[19] -= n/(float)NUMBER_OF_VOLUMES * gradient[19];
        
        // end for epocs
    }


	//
	// Testing
	//
        
	float s;

	int uncensoredVolumes = 0;

    // Make classifications
    for (int t = NUMBER_OF_VOLUMES / 2 + 1; t < NUMBER_OF_VOLUMES; t++)
    {
		// Ignore censored volumes
		if (c_Correct_Classes[t] == 9999.0f)
		{
			continue;
		}

		uncensoredVolumes++;

	    s =  weights[0] * 1.0f;
        
	    float x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16, x17, x18, x19;
        
	    x1 = Volumes[Calculate4DIndex(x-1,y,z-1,t,DATA_W,DATA_H,DATA_D)];
	    x2 = Volumes[Calculate4DIndex(x,y-1,z-1,t,DATA_W,DATA_H,DATA_D)];
	    x3 = Volumes[Calculate4DIndex(x,y,z-1,t,DATA_W,DATA_H,DATA_D)];
	    x4 = Volumes[Calculate4DIndex(x,y+1,z-1,t,DATA_W,DATA_H,DATA_D)];
		x5 = Volumes[Calculate4DIndex(x+1,y,z-1,t,DATA_W,DATA_H,DATA_D)];
        
	    x6 = Volumes[Calculate4DIndex(x-1,y-1,z,t,DATA_W,DATA_H,DATA_D)];
	    x7 = Volumes[Calculate4DIndex(x-1,y,z,t,DATA_W,DATA_H,DATA_D)];
	    x8 = Volumes[Calculate4DIndex(x-1,y+1,z,t,DATA_W,DATA_H,DATA_D)];
	    x9 = Volumes[Calculate4DIndex(x,y-1,z,t,DATA_W,DATA_H,DATA_D)];
	    x10 = Volumes[Calculate4DIndex(x,y,z,t,DATA_W,DATA_H,DATA_D)];

    	x11 = Volumes[Calculate4DIndex(x,y+1,z,t,DATA_W,DATA_H,DATA_D)];
    	x12 = Volumes[Calculate4DIndex(x+1,y-1,z,t,DATA_W,DATA_H,DATA_D)];
    	x13 = Volumes[Calculate4DIndex(x+1,y,z,t,DATA_W,DATA_H,DATA_D)];
    	x14 = Volumes[Calculate4DIndex(x+1,y+1,z,t,DATA_W,DATA_H,DATA_D)];
    	    
    	x15 = Volumes[Calculate4DIndex(x-1,y,z+1,t,DATA_W,DATA_H,DATA_D)];
    	x16 = Volumes[Calculate4DIndex(x,y-1,z+1,t,DATA_W,DATA_H,DATA_D)];
    	x17 = Volumes[Calculate4DIndex(x,y,z+1,t,DATA_W,DATA_H,DATA_D)];
    	x18 = Volumes[Calculate4DIndex(x,y+1,z+1,t,DATA_W,DATA_H,DATA_D)];
    	x19 = Volumes[Calculate4DIndex(x+1,y,z+1,t,DATA_W,DATA_H,DATA_D)];
        
    	// z - 1
    	s += weights[1] * x1;
    	s += weights[2] * x2;
    	s += weights[3] * x3;
    	s += weights[4] * x4;
    	s += weights[5] * x5;
    	    
	    // z
 	    s += weights[6] * x6;
   	    s += weights[7] * x7;
   	    s += weights[8] * x8;
   	    s += weights[9] * x9;
   	    s += weights[10] * x10;
   	    s += weights[11] * x11;
   	    s += weights[12] * x12;
   	    s += weights[13] * x13;
   	    s += weights[14] * x14;
    	    
   	    // z + 1
   	    s += weights[15] * x15;
   	    s += weights[16] * x16;
   	    s += weights[17] * x17;
   	    s += weights[18] * x18;
   	    s += weights[19] * x19;
        
   	    float classification;
   	    if (s > 0.0f)
   	    {
   	        classification = 0.0f;
   	    }
   	    else
   	    {
   	        classification = 1.0f;
   	    }
       
   	    if (classification == c_Correct_Classes[t])
        {
   	        classification_performance++;
   	    }           
    }
  
    Classifier_Performance[Calculate3DIndex(x,y,z,DATA_W,DATA_H)] = (float)classification_performance / (float)uncensoredVolumes;
}

