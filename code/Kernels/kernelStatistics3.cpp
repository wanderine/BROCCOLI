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





// Functions for permutation test

int LoadBetaWeights(__private float* beta,
                    __global const float* Beta_Volumes,
                    int x,
                    int y,
                    int z,
                    int DATA_W,
                    int DATA_H,
                    int DATA_D,
                    int NUMBER_OF_REGRESSORS)
{
    switch(NUMBER_OF_REGRESSORS)
    {
        case 1:
            
            beta[0] = Beta_Volumes[Calculate4DIndex(x,y,z,0,DATA_W,DATA_H,DATA_D)];
            
            break;
            
        case 2:
            
            beta[0] = Beta_Volumes[Calculate4DIndex(x,y,z,0,DATA_W,DATA_H,DATA_D)];
            beta[1] = Beta_Volumes[Calculate4DIndex(x,y,z,1,DATA_W,DATA_H,DATA_D)];
            
            break;
            
        case 3:
            
            beta[0] = Beta_Volumes[Calculate4DIndex(x,y,z,0,DATA_W,DATA_H,DATA_D)];
            beta[1] = Beta_Volumes[Calculate4DIndex(x,y,z,1,DATA_W,DATA_H,DATA_D)];
            beta[2] = Beta_Volumes[Calculate4DIndex(x,y,z,2,DATA_W,DATA_H,DATA_D)];
            
            break;
            
        case 4:
            
            beta[0] = Beta_Volumes[Calculate4DIndex(x,y,z,0,DATA_W,DATA_H,DATA_D)];
            beta[1] = Beta_Volumes[Calculate4DIndex(x,y,z,1,DATA_W,DATA_H,DATA_D)];
            beta[2] = Beta_Volumes[Calculate4DIndex(x,y,z,2,DATA_W,DATA_H,DATA_D)];
            beta[3] = Beta_Volumes[Calculate4DIndex(x,y,z,3,DATA_W,DATA_H,DATA_D)];
            
            break;
            
        case 5:
            
            beta[0] = Beta_Volumes[Calculate4DIndex(x,y,z,0,DATA_W,DATA_H,DATA_D)];
            beta[1] = Beta_Volumes[Calculate4DIndex(x,y,z,1,DATA_W,DATA_H,DATA_D)];
            beta[2] = Beta_Volumes[Calculate4DIndex(x,y,z,2,DATA_W,DATA_H,DATA_D)];
            beta[3] = Beta_Volumes[Calculate4DIndex(x,y,z,3,DATA_W,DATA_H,DATA_D)];
            beta[4] = Beta_Volumes[Calculate4DIndex(x,y,z,4,DATA_W,DATA_H,DATA_D)];
            
            break;
            
        case 6:
            
            beta[0] = Beta_Volumes[Calculate4DIndex(x,y,z,0,DATA_W,DATA_H,DATA_D)];
            beta[1] = Beta_Volumes[Calculate4DIndex(x,y,z,1,DATA_W,DATA_H,DATA_D)];
            beta[2] = Beta_Volumes[Calculate4DIndex(x,y,z,2,DATA_W,DATA_H,DATA_D)];
            beta[3] = Beta_Volumes[Calculate4DIndex(x,y,z,3,DATA_W,DATA_H,DATA_D)];
            beta[4] = Beta_Volumes[Calculate4DIndex(x,y,z,4,DATA_W,DATA_H,DATA_D)];
            beta[5] = Beta_Volumes[Calculate4DIndex(x,y,z,5,DATA_W,DATA_H,DATA_D)];
            
            break;
            
        case 7:
            
            beta[0] = Beta_Volumes[Calculate4DIndex(x,y,z,0,DATA_W,DATA_H,DATA_D)];
            beta[1] = Beta_Volumes[Calculate4DIndex(x,y,z,1,DATA_W,DATA_H,DATA_D)];
            beta[2] = Beta_Volumes[Calculate4DIndex(x,y,z,2,DATA_W,DATA_H,DATA_D)];
            beta[3] = Beta_Volumes[Calculate4DIndex(x,y,z,3,DATA_W,DATA_H,DATA_D)];
            beta[4] = Beta_Volumes[Calculate4DIndex(x,y,z,4,DATA_W,DATA_H,DATA_D)];
            beta[5] = Beta_Volumes[Calculate4DIndex(x,y,z,5,DATA_W,DATA_H,DATA_D)];
            beta[6] = Beta_Volumes[Calculate4DIndex(x,y,z,6,DATA_W,DATA_H,DATA_D)];
            
            break;
            
        case 8:
            
            beta[0] = Beta_Volumes[Calculate4DIndex(x,y,z,0,DATA_W,DATA_H,DATA_D)];
            beta[1] = Beta_Volumes[Calculate4DIndex(x,y,z,1,DATA_W,DATA_H,DATA_D)];
            beta[2] = Beta_Volumes[Calculate4DIndex(x,y,z,2,DATA_W,DATA_H,DATA_D)];
            beta[3] = Beta_Volumes[Calculate4DIndex(x,y,z,3,DATA_W,DATA_H,DATA_D)];
            beta[4] = Beta_Volumes[Calculate4DIndex(x,y,z,4,DATA_W,DATA_H,DATA_D)];
            beta[5] = Beta_Volumes[Calculate4DIndex(x,y,z,5,DATA_W,DATA_H,DATA_D)];
            beta[6] = Beta_Volumes[Calculate4DIndex(x,y,z,6,DATA_W,DATA_H,DATA_D)];
            beta[7] = Beta_Volumes[Calculate4DIndex(x,y,z,7,DATA_W,DATA_H,DATA_D)];
            
            break;
            
        case 9:
            
            beta[0] = Beta_Volumes[Calculate4DIndex(x,y,z,0,DATA_W,DATA_H,DATA_D)];
            beta[1] = Beta_Volumes[Calculate4DIndex(x,y,z,1,DATA_W,DATA_H,DATA_D)];
            beta[2] = Beta_Volumes[Calculate4DIndex(x,y,z,2,DATA_W,DATA_H,DATA_D)];
            beta[3] = Beta_Volumes[Calculate4DIndex(x,y,z,3,DATA_W,DATA_H,DATA_D)];
            beta[4] = Beta_Volumes[Calculate4DIndex(x,y,z,4,DATA_W,DATA_H,DATA_D)];
            beta[5] = Beta_Volumes[Calculate4DIndex(x,y,z,5,DATA_W,DATA_H,DATA_D)];
            beta[6] = Beta_Volumes[Calculate4DIndex(x,y,z,6,DATA_W,DATA_H,DATA_D)];
            beta[7] = Beta_Volumes[Calculate4DIndex(x,y,z,7,DATA_W,DATA_H,DATA_D)];
            beta[8] = Beta_Volumes[Calculate4DIndex(x,y,z,8,DATA_W,DATA_H,DATA_D)];
            
            break;
            
        case 10:
            
            beta[0] = Beta_Volumes[Calculate4DIndex(x,y,z,0,DATA_W,DATA_H,DATA_D)];
            beta[1] = Beta_Volumes[Calculate4DIndex(x,y,z,1,DATA_W,DATA_H,DATA_D)];
            beta[2] = Beta_Volumes[Calculate4DIndex(x,y,z,2,DATA_W,DATA_H,DATA_D)];
            beta[3] = Beta_Volumes[Calculate4DIndex(x,y,z,3,DATA_W,DATA_H,DATA_D)];
            beta[4] = Beta_Volumes[Calculate4DIndex(x,y,z,4,DATA_W,DATA_H,DATA_D)];
            beta[5] = Beta_Volumes[Calculate4DIndex(x,y,z,5,DATA_W,DATA_H,DATA_D)];
            beta[6] = Beta_Volumes[Calculate4DIndex(x,y,z,6,DATA_W,DATA_H,DATA_D)];
            beta[7] = Beta_Volumes[Calculate4DIndex(x,y,z,7,DATA_W,DATA_H,DATA_D)];
            beta[8] = Beta_Volumes[Calculate4DIndex(x,y,z,8,DATA_W,DATA_H,DATA_D)];
            beta[9] = Beta_Volumes[Calculate4DIndex(x,y,z,9,DATA_W,DATA_H,DATA_D)];
            
            break;
            
        case 11:
            
            beta[0] = Beta_Volumes[Calculate4DIndex(x,y,z,0,DATA_W,DATA_H,DATA_D)];
            beta[1] = Beta_Volumes[Calculate4DIndex(x,y,z,1,DATA_W,DATA_H,DATA_D)];
            beta[2] = Beta_Volumes[Calculate4DIndex(x,y,z,2,DATA_W,DATA_H,DATA_D)];
            beta[3] = Beta_Volumes[Calculate4DIndex(x,y,z,3,DATA_W,DATA_H,DATA_D)];
            beta[4] = Beta_Volumes[Calculate4DIndex(x,y,z,4,DATA_W,DATA_H,DATA_D)];
            beta[5] = Beta_Volumes[Calculate4DIndex(x,y,z,5,DATA_W,DATA_H,DATA_D)];
            beta[6] = Beta_Volumes[Calculate4DIndex(x,y,z,6,DATA_W,DATA_H,DATA_D)];
            beta[7] = Beta_Volumes[Calculate4DIndex(x,y,z,7,DATA_W,DATA_H,DATA_D)];
            beta[8] = Beta_Volumes[Calculate4DIndex(x,y,z,8,DATA_W,DATA_H,DATA_D)];
            beta[9] = Beta_Volumes[Calculate4DIndex(x,y,z,9,DATA_W,DATA_H,DATA_D)];
            beta[10] = Beta_Volumes[Calculate4DIndex(x,y,z,10,DATA_W,DATA_H,DATA_D)];
            
            break;
            
        case 12:
            
            beta[0] = Beta_Volumes[Calculate4DIndex(x,y,z,0,DATA_W,DATA_H,DATA_D)];
            beta[1] = Beta_Volumes[Calculate4DIndex(x,y,z,1,DATA_W,DATA_H,DATA_D)];
            beta[2] = Beta_Volumes[Calculate4DIndex(x,y,z,2,DATA_W,DATA_H,DATA_D)];
            beta[3] = Beta_Volumes[Calculate4DIndex(x,y,z,3,DATA_W,DATA_H,DATA_D)];
            beta[4] = Beta_Volumes[Calculate4DIndex(x,y,z,4,DATA_W,DATA_H,DATA_D)];
            beta[5] = Beta_Volumes[Calculate4DIndex(x,y,z,5,DATA_W,DATA_H,DATA_D)];
            beta[6] = Beta_Volumes[Calculate4DIndex(x,y,z,6,DATA_W,DATA_H,DATA_D)];
            beta[7] = Beta_Volumes[Calculate4DIndex(x,y,z,7,DATA_W,DATA_H,DATA_D)];
            beta[8] = Beta_Volumes[Calculate4DIndex(x,y,z,8,DATA_W,DATA_H,DATA_D)];
            beta[9] = Beta_Volumes[Calculate4DIndex(x,y,z,9,DATA_W,DATA_H,DATA_D)];
            beta[10] = Beta_Volumes[Calculate4DIndex(x,y,z,10,DATA_W,DATA_H,DATA_D)];
            beta[11] = Beta_Volumes[Calculate4DIndex(x,y,z,11,DATA_W,DATA_H,DATA_D)];
            
            break;
            
        case 13:
            
            beta[0] = Beta_Volumes[Calculate4DIndex(x,y,z,0,DATA_W,DATA_H,DATA_D)];
            beta[1] = Beta_Volumes[Calculate4DIndex(x,y,z,1,DATA_W,DATA_H,DATA_D)];
            beta[2] = Beta_Volumes[Calculate4DIndex(x,y,z,2,DATA_W,DATA_H,DATA_D)];
            beta[3] = Beta_Volumes[Calculate4DIndex(x,y,z,3,DATA_W,DATA_H,DATA_D)];
            beta[4] = Beta_Volumes[Calculate4DIndex(x,y,z,4,DATA_W,DATA_H,DATA_D)];
            beta[5] = Beta_Volumes[Calculate4DIndex(x,y,z,5,DATA_W,DATA_H,DATA_D)];
            beta[6] = Beta_Volumes[Calculate4DIndex(x,y,z,6,DATA_W,DATA_H,DATA_D)];
            beta[7] = Beta_Volumes[Calculate4DIndex(x,y,z,7,DATA_W,DATA_H,DATA_D)];
            beta[8] = Beta_Volumes[Calculate4DIndex(x,y,z,8,DATA_W,DATA_H,DATA_D)];
            beta[9] = Beta_Volumes[Calculate4DIndex(x,y,z,9,DATA_W,DATA_H,DATA_D)];
            beta[10] = Beta_Volumes[Calculate4DIndex(x,y,z,10,DATA_W,DATA_H,DATA_D)];
            beta[11] = Beta_Volumes[Calculate4DIndex(x,y,z,11,DATA_W,DATA_H,DATA_D)];
            beta[12] = Beta_Volumes[Calculate4DIndex(x,y,z,12,DATA_W,DATA_H,DATA_D)];
            
            break;
            
        case 14:
            
            beta[0] = Beta_Volumes[Calculate4DIndex(x,y,z,0,DATA_W,DATA_H,DATA_D)];
            beta[1] = Beta_Volumes[Calculate4DIndex(x,y,z,1,DATA_W,DATA_H,DATA_D)];
            beta[2] = Beta_Volumes[Calculate4DIndex(x,y,z,2,DATA_W,DATA_H,DATA_D)];
            beta[3] = Beta_Volumes[Calculate4DIndex(x,y,z,3,DATA_W,DATA_H,DATA_D)];
            beta[4] = Beta_Volumes[Calculate4DIndex(x,y,z,4,DATA_W,DATA_H,DATA_D)];
            beta[5] = Beta_Volumes[Calculate4DIndex(x,y,z,5,DATA_W,DATA_H,DATA_D)];
            beta[6] = Beta_Volumes[Calculate4DIndex(x,y,z,6,DATA_W,DATA_H,DATA_D)];
            beta[7] = Beta_Volumes[Calculate4DIndex(x,y,z,7,DATA_W,DATA_H,DATA_D)];
            beta[8] = Beta_Volumes[Calculate4DIndex(x,y,z,8,DATA_W,DATA_H,DATA_D)];
            beta[9] = Beta_Volumes[Calculate4DIndex(x,y,z,9,DATA_W,DATA_H,DATA_D)];
            beta[10] = Beta_Volumes[Calculate4DIndex(x,y,z,10,DATA_W,DATA_H,DATA_D)];
            beta[11] = Beta_Volumes[Calculate4DIndex(x,y,z,11,DATA_W,DATA_H,DATA_D)];
            beta[12] = Beta_Volumes[Calculate4DIndex(x,y,z,12,DATA_W,DATA_H,DATA_D)];
            beta[13] = Beta_Volumes[Calculate4DIndex(x,y,z,13,DATA_W,DATA_H,DATA_D)];
            
            break;
            
        case 15:
            
            beta[0] = Beta_Volumes[Calculate4DIndex(x,y,z,0,DATA_W,DATA_H,DATA_D)];
            beta[1] = Beta_Volumes[Calculate4DIndex(x,y,z,1,DATA_W,DATA_H,DATA_D)];
            beta[2] = Beta_Volumes[Calculate4DIndex(x,y,z,2,DATA_W,DATA_H,DATA_D)];
            beta[3] = Beta_Volumes[Calculate4DIndex(x,y,z,3,DATA_W,DATA_H,DATA_D)];
            beta[4] = Beta_Volumes[Calculate4DIndex(x,y,z,4,DATA_W,DATA_H,DATA_D)];
            beta[5] = Beta_Volumes[Calculate4DIndex(x,y,z,5,DATA_W,DATA_H,DATA_D)];
            beta[6] = Beta_Volumes[Calculate4DIndex(x,y,z,6,DATA_W,DATA_H,DATA_D)];
            beta[7] = Beta_Volumes[Calculate4DIndex(x,y,z,7,DATA_W,DATA_H,DATA_D)];
            beta[8] = Beta_Volumes[Calculate4DIndex(x,y,z,8,DATA_W,DATA_H,DATA_D)];
            beta[9] = Beta_Volumes[Calculate4DIndex(x,y,z,9,DATA_W,DATA_H,DATA_D)];
            beta[10] = Beta_Volumes[Calculate4DIndex(x,y,z,10,DATA_W,DATA_H,DATA_D)];
            beta[11] = Beta_Volumes[Calculate4DIndex(x,y,z,11,DATA_W,DATA_H,DATA_D)];
            beta[12] = Beta_Volumes[Calculate4DIndex(x,y,z,12,DATA_W,DATA_H,DATA_D)];
            beta[13] = Beta_Volumes[Calculate4DIndex(x,y,z,13,DATA_W,DATA_H,DATA_D)];
            beta[14] = Beta_Volumes[Calculate4DIndex(x,y,z,14,DATA_W,DATA_H,DATA_D)];
            
            break;
            
        case 16:
            
            beta[0] = Beta_Volumes[Calculate4DIndex(x,y,z,0,DATA_W,DATA_H,DATA_D)];
            beta[1] = Beta_Volumes[Calculate4DIndex(x,y,z,1,DATA_W,DATA_H,DATA_D)];
            beta[2] = Beta_Volumes[Calculate4DIndex(x,y,z,2,DATA_W,DATA_H,DATA_D)];
            beta[3] = Beta_Volumes[Calculate4DIndex(x,y,z,3,DATA_W,DATA_H,DATA_D)];
            beta[4] = Beta_Volumes[Calculate4DIndex(x,y,z,4,DATA_W,DATA_H,DATA_D)];
            beta[5] = Beta_Volumes[Calculate4DIndex(x,y,z,5,DATA_W,DATA_H,DATA_D)];
            beta[6] = Beta_Volumes[Calculate4DIndex(x,y,z,6,DATA_W,DATA_H,DATA_D)];
            beta[7] = Beta_Volumes[Calculate4DIndex(x,y,z,7,DATA_W,DATA_H,DATA_D)];
            beta[8] = Beta_Volumes[Calculate4DIndex(x,y,z,8,DATA_W,DATA_H,DATA_D)];
            beta[9] = Beta_Volumes[Calculate4DIndex(x,y,z,9,DATA_W,DATA_H,DATA_D)];
            beta[10] = Beta_Volumes[Calculate4DIndex(x,y,z,10,DATA_W,DATA_H,DATA_D)];
            beta[11] = Beta_Volumes[Calculate4DIndex(x,y,z,11,DATA_W,DATA_H,DATA_D)];
            beta[12] = Beta_Volumes[Calculate4DIndex(x,y,z,12,DATA_W,DATA_H,DATA_D)];
            beta[13] = Beta_Volumes[Calculate4DIndex(x,y,z,13,DATA_W,DATA_H,DATA_D)];
            beta[14] = Beta_Volumes[Calculate4DIndex(x,y,z,14,DATA_W,DATA_H,DATA_D)];
            beta[15] = Beta_Volumes[Calculate4DIndex(x,y,z,15,DATA_W,DATA_H,DATA_D)];
            
            break;
            
        case 17:
            
            beta[0] = Beta_Volumes[Calculate4DIndex(x,y,z,0,DATA_W,DATA_H,DATA_D)];
            beta[1] = Beta_Volumes[Calculate4DIndex(x,y,z,1,DATA_W,DATA_H,DATA_D)];
            beta[2] = Beta_Volumes[Calculate4DIndex(x,y,z,2,DATA_W,DATA_H,DATA_D)];
            beta[3] = Beta_Volumes[Calculate4DIndex(x,y,z,3,DATA_W,DATA_H,DATA_D)];
            beta[4] = Beta_Volumes[Calculate4DIndex(x,y,z,4,DATA_W,DATA_H,DATA_D)];
            beta[5] = Beta_Volumes[Calculate4DIndex(x,y,z,5,DATA_W,DATA_H,DATA_D)];
            beta[6] = Beta_Volumes[Calculate4DIndex(x,y,z,6,DATA_W,DATA_H,DATA_D)];
            beta[7] = Beta_Volumes[Calculate4DIndex(x,y,z,7,DATA_W,DATA_H,DATA_D)];
            beta[8] = Beta_Volumes[Calculate4DIndex(x,y,z,8,DATA_W,DATA_H,DATA_D)];
            beta[9] = Beta_Volumes[Calculate4DIndex(x,y,z,9,DATA_W,DATA_H,DATA_D)];
            beta[10] = Beta_Volumes[Calculate4DIndex(x,y,z,10,DATA_W,DATA_H,DATA_D)];
            beta[11] = Beta_Volumes[Calculate4DIndex(x,y,z,11,DATA_W,DATA_H,DATA_D)];
            beta[12] = Beta_Volumes[Calculate4DIndex(x,y,z,12,DATA_W,DATA_H,DATA_D)];
            beta[13] = Beta_Volumes[Calculate4DIndex(x,y,z,13,DATA_W,DATA_H,DATA_D)];
            beta[14] = Beta_Volumes[Calculate4DIndex(x,y,z,14,DATA_W,DATA_H,DATA_D)];
            beta[15] = Beta_Volumes[Calculate4DIndex(x,y,z,15,DATA_W,DATA_H,DATA_D)];
            beta[16] = Beta_Volumes[Calculate4DIndex(x,y,z,16,DATA_W,DATA_H,DATA_D)];
            
            break;
            
        case 18:
            
            beta[0] = Beta_Volumes[Calculate4DIndex(x,y,z,0,DATA_W,DATA_H,DATA_D)];
            beta[1] = Beta_Volumes[Calculate4DIndex(x,y,z,1,DATA_W,DATA_H,DATA_D)];
            beta[2] = Beta_Volumes[Calculate4DIndex(x,y,z,2,DATA_W,DATA_H,DATA_D)];
            beta[3] = Beta_Volumes[Calculate4DIndex(x,y,z,3,DATA_W,DATA_H,DATA_D)];
            beta[4] = Beta_Volumes[Calculate4DIndex(x,y,z,4,DATA_W,DATA_H,DATA_D)];
            beta[5] = Beta_Volumes[Calculate4DIndex(x,y,z,5,DATA_W,DATA_H,DATA_D)];
            beta[6] = Beta_Volumes[Calculate4DIndex(x,y,z,6,DATA_W,DATA_H,DATA_D)];
            beta[7] = Beta_Volumes[Calculate4DIndex(x,y,z,7,DATA_W,DATA_H,DATA_D)];
            beta[8] = Beta_Volumes[Calculate4DIndex(x,y,z,8,DATA_W,DATA_H,DATA_D)];
            beta[9] = Beta_Volumes[Calculate4DIndex(x,y,z,9,DATA_W,DATA_H,DATA_D)];
            beta[10] = Beta_Volumes[Calculate4DIndex(x,y,z,10,DATA_W,DATA_H,DATA_D)];
            beta[11] = Beta_Volumes[Calculate4DIndex(x,y,z,11,DATA_W,DATA_H,DATA_D)];
            beta[12] = Beta_Volumes[Calculate4DIndex(x,y,z,12,DATA_W,DATA_H,DATA_D)];
            beta[13] = Beta_Volumes[Calculate4DIndex(x,y,z,13,DATA_W,DATA_H,DATA_D)];
            beta[14] = Beta_Volumes[Calculate4DIndex(x,y,z,14,DATA_W,DATA_H,DATA_D)];
            beta[15] = Beta_Volumes[Calculate4DIndex(x,y,z,15,DATA_W,DATA_H,DATA_D)];
            beta[16] = Beta_Volumes[Calculate4DIndex(x,y,z,16,DATA_W,DATA_H,DATA_D)];
            beta[17] = Beta_Volumes[Calculate4DIndex(x,y,z,17,DATA_W,DATA_H,DATA_D)];
            
            break;
            
        case 19:
            
            beta[0] = Beta_Volumes[Calculate4DIndex(x,y,z,0,DATA_W,DATA_H,DATA_D)];
            beta[1] = Beta_Volumes[Calculate4DIndex(x,y,z,1,DATA_W,DATA_H,DATA_D)];
            beta[2] = Beta_Volumes[Calculate4DIndex(x,y,z,2,DATA_W,DATA_H,DATA_D)];
            beta[3] = Beta_Volumes[Calculate4DIndex(x,y,z,3,DATA_W,DATA_H,DATA_D)];
            beta[4] = Beta_Volumes[Calculate4DIndex(x,y,z,4,DATA_W,DATA_H,DATA_D)];
            beta[5] = Beta_Volumes[Calculate4DIndex(x,y,z,5,DATA_W,DATA_H,DATA_D)];
            beta[6] = Beta_Volumes[Calculate4DIndex(x,y,z,6,DATA_W,DATA_H,DATA_D)];
            beta[7] = Beta_Volumes[Calculate4DIndex(x,y,z,7,DATA_W,DATA_H,DATA_D)];
            beta[8] = Beta_Volumes[Calculate4DIndex(x,y,z,8,DATA_W,DATA_H,DATA_D)];
            beta[9] = Beta_Volumes[Calculate4DIndex(x,y,z,9,DATA_W,DATA_H,DATA_D)];
            beta[10] = Beta_Volumes[Calculate4DIndex(x,y,z,10,DATA_W,DATA_H,DATA_D)];
            beta[11] = Beta_Volumes[Calculate4DIndex(x,y,z,11,DATA_W,DATA_H,DATA_D)];
            beta[12] = Beta_Volumes[Calculate4DIndex(x,y,z,12,DATA_W,DATA_H,DATA_D)];
            beta[13] = Beta_Volumes[Calculate4DIndex(x,y,z,13,DATA_W,DATA_H,DATA_D)];
            beta[14] = Beta_Volumes[Calculate4DIndex(x,y,z,14,DATA_W,DATA_H,DATA_D)];
            beta[15] = Beta_Volumes[Calculate4DIndex(x,y,z,15,DATA_W,DATA_H,DATA_D)];
            beta[16] = Beta_Volumes[Calculate4DIndex(x,y,z,16,DATA_W,DATA_H,DATA_D)];
            beta[17] = Beta_Volumes[Calculate4DIndex(x,y,z,17,DATA_W,DATA_H,DATA_D)];
            beta[18] = Beta_Volumes[Calculate4DIndex(x,y,z,18,DATA_W,DATA_H,DATA_D)];
            
            break;
            
        case 20:
            
            beta[0] = Beta_Volumes[Calculate4DIndex(x,y,z,0,DATA_W,DATA_H,DATA_D)];
            beta[1] = Beta_Volumes[Calculate4DIndex(x,y,z,1,DATA_W,DATA_H,DATA_D)];
            beta[2] = Beta_Volumes[Calculate4DIndex(x,y,z,2,DATA_W,DATA_H,DATA_D)];
            beta[3] = Beta_Volumes[Calculate4DIndex(x,y,z,3,DATA_W,DATA_H,DATA_D)];
            beta[4] = Beta_Volumes[Calculate4DIndex(x,y,z,4,DATA_W,DATA_H,DATA_D)];
            beta[5] = Beta_Volumes[Calculate4DIndex(x,y,z,5,DATA_W,DATA_H,DATA_D)];
            beta[6] = Beta_Volumes[Calculate4DIndex(x,y,z,6,DATA_W,DATA_H,DATA_D)];
            beta[7] = Beta_Volumes[Calculate4DIndex(x,y,z,7,DATA_W,DATA_H,DATA_D)];
            beta[8] = Beta_Volumes[Calculate4DIndex(x,y,z,8,DATA_W,DATA_H,DATA_D)];
            beta[9] = Beta_Volumes[Calculate4DIndex(x,y,z,9,DATA_W,DATA_H,DATA_D)];
            beta[10] = Beta_Volumes[Calculate4DIndex(x,y,z,10,DATA_W,DATA_H,DATA_D)];
            beta[11] = Beta_Volumes[Calculate4DIndex(x,y,z,11,DATA_W,DATA_H,DATA_D)];
            beta[12] = Beta_Volumes[Calculate4DIndex(x,y,z,12,DATA_W,DATA_H,DATA_D)];
            beta[13] = Beta_Volumes[Calculate4DIndex(x,y,z,13,DATA_W,DATA_H,DATA_D)];
            beta[14] = Beta_Volumes[Calculate4DIndex(x,y,z,14,DATA_W,DATA_H,DATA_D)];
            beta[15] = Beta_Volumes[Calculate4DIndex(x,y,z,15,DATA_W,DATA_H,DATA_D)];
            beta[16] = Beta_Volumes[Calculate4DIndex(x,y,z,16,DATA_W,DATA_H,DATA_D)];
            beta[17] = Beta_Volumes[Calculate4DIndex(x,y,z,17,DATA_W,DATA_H,DATA_D)];
            beta[18] = Beta_Volumes[Calculate4DIndex(x,y,z,18,DATA_W,DATA_H,DATA_D)];
            beta[19] = Beta_Volumes[Calculate4DIndex(x,y,z,19,DATA_W,DATA_H,DATA_D)];
            
            break;
            
        case 21:
            
            beta[0] = Beta_Volumes[Calculate4DIndex(x,y,z,0,DATA_W,DATA_H,DATA_D)];
            beta[1] = Beta_Volumes[Calculate4DIndex(x,y,z,1,DATA_W,DATA_H,DATA_D)];
            beta[2] = Beta_Volumes[Calculate4DIndex(x,y,z,2,DATA_W,DATA_H,DATA_D)];
            beta[3] = Beta_Volumes[Calculate4DIndex(x,y,z,3,DATA_W,DATA_H,DATA_D)];
            beta[4] = Beta_Volumes[Calculate4DIndex(x,y,z,4,DATA_W,DATA_H,DATA_D)];
            beta[5] = Beta_Volumes[Calculate4DIndex(x,y,z,5,DATA_W,DATA_H,DATA_D)];
            beta[6] = Beta_Volumes[Calculate4DIndex(x,y,z,6,DATA_W,DATA_H,DATA_D)];
            beta[7] = Beta_Volumes[Calculate4DIndex(x,y,z,7,DATA_W,DATA_H,DATA_D)];
            beta[8] = Beta_Volumes[Calculate4DIndex(x,y,z,8,DATA_W,DATA_H,DATA_D)];
            beta[9] = Beta_Volumes[Calculate4DIndex(x,y,z,9,DATA_W,DATA_H,DATA_D)];
            beta[10] = Beta_Volumes[Calculate4DIndex(x,y,z,10,DATA_W,DATA_H,DATA_D)];
            beta[11] = Beta_Volumes[Calculate4DIndex(x,y,z,11,DATA_W,DATA_H,DATA_D)];
            beta[12] = Beta_Volumes[Calculate4DIndex(x,y,z,12,DATA_W,DATA_H,DATA_D)];
            beta[13] = Beta_Volumes[Calculate4DIndex(x,y,z,13,DATA_W,DATA_H,DATA_D)];
            beta[14] = Beta_Volumes[Calculate4DIndex(x,y,z,14,DATA_W,DATA_H,DATA_D)];
            beta[15] = Beta_Volumes[Calculate4DIndex(x,y,z,15,DATA_W,DATA_H,DATA_D)];
            beta[16] = Beta_Volumes[Calculate4DIndex(x,y,z,16,DATA_W,DATA_H,DATA_D)];
            beta[17] = Beta_Volumes[Calculate4DIndex(x,y,z,17,DATA_W,DATA_H,DATA_D)];
            beta[18] = Beta_Volumes[Calculate4DIndex(x,y,z,18,DATA_W,DATA_H,DATA_D)];
            beta[19] = Beta_Volumes[Calculate4DIndex(x,y,z,19,DATA_W,DATA_H,DATA_D)];
            beta[20] = Beta_Volumes[Calculate4DIndex(x,y,z,20,DATA_W,DATA_H,DATA_D)];
            
            break;
            
        case 22:
            
            beta[0] = Beta_Volumes[Calculate4DIndex(x,y,z,0,DATA_W,DATA_H,DATA_D)];
            beta[1] = Beta_Volumes[Calculate4DIndex(x,y,z,1,DATA_W,DATA_H,DATA_D)];
            beta[2] = Beta_Volumes[Calculate4DIndex(x,y,z,2,DATA_W,DATA_H,DATA_D)];
            beta[3] = Beta_Volumes[Calculate4DIndex(x,y,z,3,DATA_W,DATA_H,DATA_D)];
            beta[4] = Beta_Volumes[Calculate4DIndex(x,y,z,4,DATA_W,DATA_H,DATA_D)];
            beta[5] = Beta_Volumes[Calculate4DIndex(x,y,z,5,DATA_W,DATA_H,DATA_D)];
            beta[6] = Beta_Volumes[Calculate4DIndex(x,y,z,6,DATA_W,DATA_H,DATA_D)];
            beta[7] = Beta_Volumes[Calculate4DIndex(x,y,z,7,DATA_W,DATA_H,DATA_D)];
            beta[8] = Beta_Volumes[Calculate4DIndex(x,y,z,8,DATA_W,DATA_H,DATA_D)];
            beta[9] = Beta_Volumes[Calculate4DIndex(x,y,z,9,DATA_W,DATA_H,DATA_D)];
            beta[10] = Beta_Volumes[Calculate4DIndex(x,y,z,10,DATA_W,DATA_H,DATA_D)];
            beta[11] = Beta_Volumes[Calculate4DIndex(x,y,z,11,DATA_W,DATA_H,DATA_D)];
            beta[12] = Beta_Volumes[Calculate4DIndex(x,y,z,12,DATA_W,DATA_H,DATA_D)];
            beta[13] = Beta_Volumes[Calculate4DIndex(x,y,z,13,DATA_W,DATA_H,DATA_D)];
            beta[14] = Beta_Volumes[Calculate4DIndex(x,y,z,14,DATA_W,DATA_H,DATA_D)];
            beta[15] = Beta_Volumes[Calculate4DIndex(x,y,z,15,DATA_W,DATA_H,DATA_D)];
            beta[16] = Beta_Volumes[Calculate4DIndex(x,y,z,16,DATA_W,DATA_H,DATA_D)];
            beta[17] = Beta_Volumes[Calculate4DIndex(x,y,z,17,DATA_W,DATA_H,DATA_D)];
            beta[18] = Beta_Volumes[Calculate4DIndex(x,y,z,18,DATA_W,DATA_H,DATA_D)];
            beta[19] = Beta_Volumes[Calculate4DIndex(x,y,z,19,DATA_W,DATA_H,DATA_D)];
            beta[20] = Beta_Volumes[Calculate4DIndex(x,y,z,20,DATA_W,DATA_H,DATA_D)];
            beta[21] = Beta_Volumes[Calculate4DIndex(x,y,z,21,DATA_W,DATA_H,DATA_D)];
            
            break;
            
        case 23:
            
            beta[0] = Beta_Volumes[Calculate4DIndex(x,y,z,0,DATA_W,DATA_H,DATA_D)];
            beta[1] = Beta_Volumes[Calculate4DIndex(x,y,z,1,DATA_W,DATA_H,DATA_D)];
            beta[2] = Beta_Volumes[Calculate4DIndex(x,y,z,2,DATA_W,DATA_H,DATA_D)];
            beta[3] = Beta_Volumes[Calculate4DIndex(x,y,z,3,DATA_W,DATA_H,DATA_D)];
            beta[4] = Beta_Volumes[Calculate4DIndex(x,y,z,4,DATA_W,DATA_H,DATA_D)];
            beta[5] = Beta_Volumes[Calculate4DIndex(x,y,z,5,DATA_W,DATA_H,DATA_D)];
            beta[6] = Beta_Volumes[Calculate4DIndex(x,y,z,6,DATA_W,DATA_H,DATA_D)];
            beta[7] = Beta_Volumes[Calculate4DIndex(x,y,z,7,DATA_W,DATA_H,DATA_D)];
            beta[8] = Beta_Volumes[Calculate4DIndex(x,y,z,8,DATA_W,DATA_H,DATA_D)];
            beta[9] = Beta_Volumes[Calculate4DIndex(x,y,z,9,DATA_W,DATA_H,DATA_D)];
            beta[10] = Beta_Volumes[Calculate4DIndex(x,y,z,10,DATA_W,DATA_H,DATA_D)];
            beta[11] = Beta_Volumes[Calculate4DIndex(x,y,z,11,DATA_W,DATA_H,DATA_D)];
            beta[12] = Beta_Volumes[Calculate4DIndex(x,y,z,12,DATA_W,DATA_H,DATA_D)];
            beta[13] = Beta_Volumes[Calculate4DIndex(x,y,z,13,DATA_W,DATA_H,DATA_D)];
            beta[14] = Beta_Volumes[Calculate4DIndex(x,y,z,14,DATA_W,DATA_H,DATA_D)];
            beta[15] = Beta_Volumes[Calculate4DIndex(x,y,z,15,DATA_W,DATA_H,DATA_D)];
            beta[16] = Beta_Volumes[Calculate4DIndex(x,y,z,16,DATA_W,DATA_H,DATA_D)];
            beta[17] = Beta_Volumes[Calculate4DIndex(x,y,z,17,DATA_W,DATA_H,DATA_D)];
            beta[18] = Beta_Volumes[Calculate4DIndex(x,y,z,18,DATA_W,DATA_H,DATA_D)];
            beta[19] = Beta_Volumes[Calculate4DIndex(x,y,z,19,DATA_W,DATA_H,DATA_D)];
            beta[20] = Beta_Volumes[Calculate4DIndex(x,y,z,20,DATA_W,DATA_H,DATA_D)];
            beta[21] = Beta_Volumes[Calculate4DIndex(x,y,z,21,DATA_W,DATA_H,DATA_D)];
            beta[22] = Beta_Volumes[Calculate4DIndex(x,y,z,22,DATA_W,DATA_H,DATA_D)];
            
            break;
            
        case 24:
            
            beta[0] = Beta_Volumes[Calculate4DIndex(x,y,z,0,DATA_W,DATA_H,DATA_D)];
            beta[1] = Beta_Volumes[Calculate4DIndex(x,y,z,1,DATA_W,DATA_H,DATA_D)];
            beta[2] = Beta_Volumes[Calculate4DIndex(x,y,z,2,DATA_W,DATA_H,DATA_D)];
            beta[3] = Beta_Volumes[Calculate4DIndex(x,y,z,3,DATA_W,DATA_H,DATA_D)];
            beta[4] = Beta_Volumes[Calculate4DIndex(x,y,z,4,DATA_W,DATA_H,DATA_D)];
            beta[5] = Beta_Volumes[Calculate4DIndex(x,y,z,5,DATA_W,DATA_H,DATA_D)];
            beta[6] = Beta_Volumes[Calculate4DIndex(x,y,z,6,DATA_W,DATA_H,DATA_D)];
            beta[7] = Beta_Volumes[Calculate4DIndex(x,y,z,7,DATA_W,DATA_H,DATA_D)];
            beta[8] = Beta_Volumes[Calculate4DIndex(x,y,z,8,DATA_W,DATA_H,DATA_D)];
            beta[9] = Beta_Volumes[Calculate4DIndex(x,y,z,9,DATA_W,DATA_H,DATA_D)];
            beta[10] = Beta_Volumes[Calculate4DIndex(x,y,z,10,DATA_W,DATA_H,DATA_D)];
            beta[11] = Beta_Volumes[Calculate4DIndex(x,y,z,11,DATA_W,DATA_H,DATA_D)];
            beta[12] = Beta_Volumes[Calculate4DIndex(x,y,z,12,DATA_W,DATA_H,DATA_D)];
            beta[13] = Beta_Volumes[Calculate4DIndex(x,y,z,13,DATA_W,DATA_H,DATA_D)];
            beta[14] = Beta_Volumes[Calculate4DIndex(x,y,z,14,DATA_W,DATA_H,DATA_D)];
            beta[15] = Beta_Volumes[Calculate4DIndex(x,y,z,15,DATA_W,DATA_H,DATA_D)];
            beta[16] = Beta_Volumes[Calculate4DIndex(x,y,z,16,DATA_W,DATA_H,DATA_D)];
            beta[17] = Beta_Volumes[Calculate4DIndex(x,y,z,17,DATA_W,DATA_H,DATA_D)];
            beta[18] = Beta_Volumes[Calculate4DIndex(x,y,z,18,DATA_W,DATA_H,DATA_D)];
            beta[19] = Beta_Volumes[Calculate4DIndex(x,y,z,19,DATA_W,DATA_H,DATA_D)];
            beta[20] = Beta_Volumes[Calculate4DIndex(x,y,z,20,DATA_W,DATA_H,DATA_D)];
            beta[21] = Beta_Volumes[Calculate4DIndex(x,y,z,21,DATA_W,DATA_H,DATA_D)];
            beta[22] = Beta_Volumes[Calculate4DIndex(x,y,z,22,DATA_W,DATA_H,DATA_D)];
            beta[23] = Beta_Volumes[Calculate4DIndex(x,y,z,23,DATA_W,DATA_H,DATA_D)];
            
            break;
            
        case 25:
            
            beta[0] = Beta_Volumes[Calculate4DIndex(x,y,z,0,DATA_W,DATA_H,DATA_D)];
            beta[1] = Beta_Volumes[Calculate4DIndex(x,y,z,1,DATA_W,DATA_H,DATA_D)];
            beta[2] = Beta_Volumes[Calculate4DIndex(x,y,z,2,DATA_W,DATA_H,DATA_D)];
            beta[3] = Beta_Volumes[Calculate4DIndex(x,y,z,3,DATA_W,DATA_H,DATA_D)];
            beta[4] = Beta_Volumes[Calculate4DIndex(x,y,z,4,DATA_W,DATA_H,DATA_D)];
            beta[5] = Beta_Volumes[Calculate4DIndex(x,y,z,5,DATA_W,DATA_H,DATA_D)];
            beta[6] = Beta_Volumes[Calculate4DIndex(x,y,z,6,DATA_W,DATA_H,DATA_D)];
            beta[7] = Beta_Volumes[Calculate4DIndex(x,y,z,7,DATA_W,DATA_H,DATA_D)];
            beta[8] = Beta_Volumes[Calculate4DIndex(x,y,z,8,DATA_W,DATA_H,DATA_D)];
            beta[9] = Beta_Volumes[Calculate4DIndex(x,y,z,9,DATA_W,DATA_H,DATA_D)];
            beta[10] = Beta_Volumes[Calculate4DIndex(x,y,z,10,DATA_W,DATA_H,DATA_D)];
            beta[11] = Beta_Volumes[Calculate4DIndex(x,y,z,11,DATA_W,DATA_H,DATA_D)];
            beta[12] = Beta_Volumes[Calculate4DIndex(x,y,z,12,DATA_W,DATA_H,DATA_D)];
            beta[13] = Beta_Volumes[Calculate4DIndex(x,y,z,13,DATA_W,DATA_H,DATA_D)];
            beta[14] = Beta_Volumes[Calculate4DIndex(x,y,z,14,DATA_W,DATA_H,DATA_D)];
            beta[15] = Beta_Volumes[Calculate4DIndex(x,y,z,15,DATA_W,DATA_H,DATA_D)];
            beta[16] = Beta_Volumes[Calculate4DIndex(x,y,z,16,DATA_W,DATA_H,DATA_D)];
            beta[17] = Beta_Volumes[Calculate4DIndex(x,y,z,17,DATA_W,DATA_H,DATA_D)];
            beta[18] = Beta_Volumes[Calculate4DIndex(x,y,z,18,DATA_W,DATA_H,DATA_D)];
            beta[19] = Beta_Volumes[Calculate4DIndex(x,y,z,19,DATA_W,DATA_H,DATA_D)];
            beta[20] = Beta_Volumes[Calculate4DIndex(x,y,z,20,DATA_W,DATA_H,DATA_D)];
            beta[21] = Beta_Volumes[Calculate4DIndex(x,y,z,21,DATA_W,DATA_H,DATA_D)];
            beta[22] = Beta_Volumes[Calculate4DIndex(x,y,z,22,DATA_W,DATA_H,DATA_D)];
            beta[23] = Beta_Volumes[Calculate4DIndex(x,y,z,23,DATA_W,DATA_H,DATA_D)];
            beta[24] = Beta_Volumes[Calculate4DIndex(x,y,z,24,DATA_W,DATA_H,DATA_D)];
            
            break;
            
            
        default:
            1;
            break;
    }
    
    return 0;
}

/*
 int CalculateCovarianceMatricesFirstLevel(__private float* Cxy,
 __private float* Cyy,
 __private float value1,
 __private float value2,
 __constant float* c_X_GLM,
 int v,
 int NUMBER_OF_VOLUMES,
 int NUMBER_OF_REGRESSORS)
 {
	switch(NUMBER_OF_REGRESSORS)
	{
 case 1:
 
 Cxy[0][0] += c_X_GLM[v + 0 * NUMBER_OF_VOLUMES] * value1;
 Cxy[0][1] += c_X_GLM[v + 0 * NUMBER_OF_VOLUMES] * value2;
 
 break;
 
 case 2:
 
 Cxy[0][0] += c_X_GLM[v + 0 * NUMBER_OF_VOLUMES] * value1;
 Cxy[0][1] += c_X_GLM[v + 0 * NUMBER_OF_VOLUMES] * value2;
 
 Cxy[1][0] += c_X_GLM[v + 1 * NUMBER_OF_VOLUMES] * value1;
 Cxy[1][1] += c_X_GLM[v + 1 * NUMBER_OF_VOLUMES] * value2;
 
 break;
 
 case 3:
 
 Cxy[0][0] += c_X_GLM[v + 0 * NUMBER_OF_VOLUMES] * value1;
 Cxy[0][1] += c_X_GLM[v + 0 * NUMBER_OF_VOLUMES] * value2;
 
 Cxy[1][0] += c_X_GLM[v + 1 * NUMBER_OF_VOLUMES] * value1;
 Cxy[1][1] += c_X_GLM[v + 1 * NUMBER_OF_VOLUMES] * value2;
 
 Cxy[2][0] += c_X_GLM[v + 2 * NUMBER_OF_VOLUMES] * value1;
 Cxy[2][1] += c_X_GLM[v + 2 * NUMBER_OF_VOLUMES] * value2;
 
 break;
 
 case 4:
 
 Cxy[0][0] += c_X_GLM[v + 0 * NUMBER_OF_VOLUMES] * value1;
 Cxy[0][1] += c_X_GLM[v + 0 * NUMBER_OF_VOLUMES] * value2;
 
 Cxy[1][0] += c_X_GLM[v + 1 * NUMBER_OF_VOLUMES] * value1;
 Cxy[1][1] += c_X_GLM[v + 1 * NUMBER_OF_VOLUMES] * value2;
 
 Cxy[2][0] += c_X_GLM[v + 2 * NUMBER_OF_VOLUMES] * value1;
 Cxy[2][1] += c_X_GLM[v + 2 * NUMBER_OF_VOLUMES] * value2;
 
 Cxy[3][0] += c_X_GLM[v + 3 * NUMBER_OF_VOLUMES] * value1;
 Cxy[3][1] += c_X_GLM[v + 3 * NUMBER_OF_VOLUMES] * value2;
 
 break;
 
 case 5:
 
 Cxy[0][0] += c_X_GLM[v + 0 * NUMBER_OF_VOLUMES] * value1;
 Cxy[0][1] += c_X_GLM[v + 0 * NUMBER_OF_VOLUMES] * value2;
 
 Cxy[1][0] += c_X_GLM[v + 1 * NUMBER_OF_VOLUMES] * value1;
 Cxy[1][1] += c_X_GLM[v + 1 * NUMBER_OF_VOLUMES] * value2;
 
 Cxy[2][0] += c_X_GLM[v + 2 * NUMBER_OF_VOLUMES] * value1;
 Cxy[2][1] += c_X_GLM[v + 2 * NUMBER_OF_VOLUMES] * value2;
 
 Cxy[3][0] += c_X_GLM[v + 3 * NUMBER_OF_VOLUMES] * value1;
 Cxy[3][1] += c_X_GLM[v + 3 * NUMBER_OF_VOLUMES] * value2;
 
 Cxy[4][0] += c_X_GLM[v + 4 * NUMBER_OF_VOLUMES] * value1;
 Cxy[4][1] += c_X_GLM[v + 4 * NUMBER_OF_VOLUMES] * value2;
 
 break;
 
 default:
 1;
 break;
	}
 
	Cyy[0][0] += value1 * value1;
	Cyy[0][1] += value1 * value2;
	Cyy[1][0] += value2 * value1;
	Cyy[1][1] += value2 * value2;
 
	return 0;
 }
 
 
 int NormalizeCovarianceMatrices(__private float* Cxy, __private float* Cyy, int NUMBER_OF_VOLUMES, int NUMBER_OF_REGRESSORS)
 {
	float div = ((float)NUMBER_OF_VOLUMES - 1);
 
	switch(NUMBER_OF_REGRESSORS)
	{
 case 1:
 
 Cxy[0][0] /= div;
 Cxy[0][1] /= div;
 
 break;
 
 case 2:
 
 Cxy[0][0] /= div;
 Cxy[0][1] /= div;
 
 Cxy[1][0] /= div;
 Cxy[1][1] /= div;
 
 break;
 
 case 3:
 
 Cxy[0][0] /= div;
 Cxy[0][1] /= div;
 
 Cxy[1][0] /= div;
 Cxy[1][1] /= div;
 
 Cxy[2][0] /= div;
 Cxy[2][1] /= div;
 
 break;
 
 case 4:
 
 Cxy[0][0] /= div;
 Cxy[0][1] /= div;
 
 Cxy[1][0] /= div;
 Cxy[1][1] /= div;
 
 Cxy[2][0] /= div;
 Cxy[2][1] /= div;
 
 Cxy[3][0] /= div;
 Cxy[3][1] /= div;
 
 break;
 
 case 5:
 
 Cxy[0][0] /= div;
 Cxy[0][1] /= div;
 
 Cxy[1][0] /= div;
 Cxy[1][1] /= div;
 
 Cxy[2][0] /= div;
 Cxy[2][1] /= div;
 
 Cxy[3][0] /= div;
 Cxy[3][1] /= div;
 
 Cxy[4][0] /= div;
 Cxy[4][1] /= div;
 
 break;
 
 default:
 1;
 break;
	}
 
	Cyy[0][0] /= div;
	Cyy[0][1] /= div;
	Cyy[1][0] /= div;
	Cyy[1][1] /= div;
 
	return 0;
 }
 */

int CalculateBetaWeightsFirstLevel(__private float* beta,
                                   __private float value,
                                   __constant float* c_xtxxt_GLM,
                                   int v,
                                   int NUMBER_OF_VOLUMES,
                                   int NUMBER_OF_REGRESSORS)
{
    switch(NUMBER_OF_REGRESSORS)
    {
        case 1:
            
            beta[0] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 0 + v];
            
            break;
            
        case 2:
            
            beta[0] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 0 + v];
            beta[1] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 1 + v];
            
            break;
            
        case 3:
            
            beta[0] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 0 + v];
            beta[1] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 1 + v];
            beta[2] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 2 + v];
            
            break;
            
        case 4:
            
            beta[0] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 0 + v];
            beta[1] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 1 + v];
            beta[2] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 2 + v];
            beta[3] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 3 + v];
            
            break;
            
        case 5:
            
            beta[0] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 0 + v];
            beta[1] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 1 + v];
            beta[2] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 2 + v];
            beta[3] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 3 + v];
            beta[4] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 4 + v];
            
            break;
            
        case 6:
            
            beta[0] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 0 + v];
            beta[1] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 1 + v];
            beta[2] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 2 + v];
            beta[3] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 3 + v];
            beta[4] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 4 + v];
            beta[5] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 5 + v];
            
            break;
            
        case 7:
            
            beta[0] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 0 + v];
            beta[1] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 1 + v];
            beta[2] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 2 + v];
            beta[3] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 3 + v];
            beta[4] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 4 + v];
            beta[5] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 5 + v];
            beta[6] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 6 + v];
            
            break;
            
        case 8:
            
            beta[0] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 0 + v];
            beta[1] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 1 + v];
            beta[2] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 2 + v];
            beta[3] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 3 + v];
            beta[4] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 4 + v];
            beta[5] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 5 + v];
            beta[6] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 6 + v];
            beta[7] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 7 + v];
            
            break;
            
        case 9:
            
            beta[0] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 0 + v];
            beta[1] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 1 + v];
            beta[2] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 2 + v];
            beta[3] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 3 + v];
            beta[4] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 4 + v];
            beta[5] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 5 + v];
            beta[6] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 6 + v];
            beta[7] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 7 + v];
            beta[8] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 8 + v];
            
            break;
            
        case 10:
            
            beta[0] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 0 + v];
            beta[1] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 1 + v];
            beta[2] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 2 + v];
            beta[3] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 3 + v];
            beta[4] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 4 + v];
            beta[5] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 5 + v];
            beta[6] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 6 + v];
            beta[7] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 7 + v];
            beta[8] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 8 + v];
            beta[9] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 9 + v];
            
            break;
            
        case 11:
            
            beta[0] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 0 + v];
            beta[1] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 1 + v];
            beta[2] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 2 + v];
            beta[3] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 3 + v];
            beta[4] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 4 + v];
            beta[5] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 5 + v];
            beta[6] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 6 + v];
            beta[7] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 7 + v];
            beta[8] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 8 + v];
            beta[9] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 9 + v];
            beta[10] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 10 + v];
            
            break;
            
        case 12:
            
            beta[0] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 0 + v];
            beta[1] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 1 + v];
            beta[2] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 2 + v];
            beta[3] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 3 + v];
            beta[4] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 4 + v];
            beta[5] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 5 + v];
            beta[6] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 6 + v];
            beta[7] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 7 + v];
            beta[8] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 8 + v];
            beta[9] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 9 + v];
            beta[10] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 10 + v];
            beta[11] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 11 + v];
            
            break;
            
        case 13:
            
            beta[0] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 0 + v];
            beta[1] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 1 + v];
            beta[2] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 2 + v];
            beta[3] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 3 + v];
            beta[4] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 4 + v];
            beta[5] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 5 + v];
            beta[6] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 6 + v];
            beta[7] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 7 + v];
            beta[8] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 8 + v];
            beta[9] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 9 + v];
            beta[10] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 10 + v];
            beta[11] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 11 + v];
            beta[12] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 12 + v];
            
            break;
            
        case 14:
            
            beta[0] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 0 + v];
            beta[1] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 1 + v];
            beta[2] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 2 + v];
            beta[3] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 3 + v];
            beta[4] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 4 + v];
            beta[5] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 5 + v];
            beta[6] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 6 + v];
            beta[7] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 7 + v];
            beta[8] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 8 + v];
            beta[9] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 9 + v];
            beta[10] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 10 + v];
            beta[11] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 11 + v];
            beta[12] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 12 + v];
            beta[13] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 13 + v];
            
            break;
            
        case 15:
            
            beta[0] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 0 + v];
            beta[1] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 1 + v];
            beta[2] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 2 + v];
            beta[3] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 3 + v];
            beta[4] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 4 + v];
            beta[5] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 5 + v];
            beta[6] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 6 + v];
            beta[7] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 7 + v];
            beta[8] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 8 + v];
            beta[9] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 9 + v];
            beta[10] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 10 + v];
            beta[11] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 11 + v];
            beta[12] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 12 + v];
            beta[13] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 13 + v];
            beta[14] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 14 + v];
            
            break;
            
        case 16:
            
            beta[0] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 0 + v];
            beta[1] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 1 + v];
            beta[2] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 2 + v];
            beta[3] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 3 + v];
            beta[4] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 4 + v];
            beta[5] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 5 + v];
            beta[6] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 6 + v];
            beta[7] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 7 + v];
            beta[8] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 8 + v];
            beta[9] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 9 + v];
            beta[10] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 10 + v];
            beta[11] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 11 + v];
            beta[12] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 12 + v];
            beta[13] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 13 + v];
            beta[14] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 14 + v];
            beta[15] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 15 + v];
            
            break;
            
        case 17:
            
            beta[0] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 0 + v];
            beta[1] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 1 + v];
            beta[2] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 2 + v];
            beta[3] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 3 + v];
            beta[4] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 4 + v];
            beta[5] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 5 + v];
            beta[6] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 6 + v];
            beta[7] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 7 + v];
            beta[8] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 8 + v];
            beta[9] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 9 + v];
            beta[10] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 10 + v];
            beta[11] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 11 + v];
            beta[12] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 12 + v];
            beta[13] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 13 + v];
            beta[14] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 14 + v];
            beta[15] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 15 + v];
            beta[16] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 16 + v];
            
            break;
            
        case 18:
            
            beta[0] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 0 + v];
            beta[1] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 1 + v];
            beta[2] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 2 + v];
            beta[3] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 3 + v];
            beta[4] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 4 + v];
            beta[5] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 5 + v];
            beta[6] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 6 + v];
            beta[7] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 7 + v];
            beta[8] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 8 + v];
            beta[9] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 9 + v];
            beta[10] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 10 + v];
            beta[11] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 11 + v];
            beta[12] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 12 + v];
            beta[13] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 13 + v];
            beta[14] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 14 + v];
            beta[15] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 15 + v];
            beta[16] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 16 + v];
            beta[17] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 17 + v];
            
            break;
            
        case 19:
            
            beta[0] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 0 + v];
            beta[1] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 1 + v];
            beta[2] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 2 + v];
            beta[3] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 3 + v];
            beta[4] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 4 + v];
            beta[5] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 5 + v];
            beta[6] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 6 + v];
            beta[7] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 7 + v];
            beta[8] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 8 + v];
            beta[9] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 9 + v];
            beta[10] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 10 + v];
            beta[11] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 11 + v];
            beta[12] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 12 + v];
            beta[13] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 13 + v];
            beta[14] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 14 + v];
            beta[15] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 15 + v];
            beta[16] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 16 + v];
            beta[17] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 17 + v];
            beta[18] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 18 + v];
            
            break;
            
        case 20:
            
            beta[0] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 0 + v];
            beta[1] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 1 + v];
            beta[2] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 2 + v];
            beta[3] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 3 + v];
            beta[4] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 4 + v];
            beta[5] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 5 + v];
            beta[6] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 6 + v];
            beta[7] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 7 + v];
            beta[8] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 8 + v];
            beta[9] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 9 + v];
            beta[10] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 10 + v];
            beta[11] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 11 + v];
            beta[12] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 12 + v];
            beta[13] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 13 + v];
            beta[14] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 14 + v];
            beta[15] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 15 + v];
            beta[16] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 16 + v];
            beta[17] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 17 + v];
            beta[18] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 18 + v];
            beta[19] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 19 + v];
            
            break;
            
        case 21:
            
            beta[0] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 0 + v];
            beta[1] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 1 + v];
            beta[2] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 2 + v];
            beta[3] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 3 + v];
            beta[4] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 4 + v];
            beta[5] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 5 + v];
            beta[6] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 6 + v];
            beta[7] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 7 + v];
            beta[8] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 8 + v];
            beta[9] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 9 + v];
            beta[10] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 10 + v];
            beta[11] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 11 + v];
            beta[12] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 12 + v];
            beta[13] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 13 + v];
            beta[14] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 14 + v];
            beta[15] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 15 + v];
            beta[16] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 16 + v];
            beta[17] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 17 + v];
            beta[18] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 18 + v];
            beta[19] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 19 + v];
            beta[20] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 20 + v];
            
            break;
            
        case 22:
            
            beta[0] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 0 + v];
            beta[1] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 1 + v];
            beta[2] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 2 + v];
            beta[3] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 3 + v];
            beta[4] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 4 + v];
            beta[5] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 5 + v];
            beta[6] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 6 + v];
            beta[7] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 7 + v];
            beta[8] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 8 + v];
            beta[9] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 9 + v];
            beta[10] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 10 + v];
            beta[11] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 11 + v];
            beta[12] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 12 + v];
            beta[13] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 13 + v];
            beta[14] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 14 + v];
            beta[15] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 15 + v];
            beta[16] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 16 + v];
            beta[17] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 17 + v];
            beta[18] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 18 + v];
            beta[19] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 19 + v];
            beta[20] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 20 + v];
            beta[21] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 21 + v];
            
            break;
            
        case 23:
            
            beta[0] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 0 + v];
            beta[1] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 1 + v];
            beta[2] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 2 + v];
            beta[3] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 3 + v];
            beta[4] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 4 + v];
            beta[5] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 5 + v];
            beta[6] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 6 + v];
            beta[7] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 7 + v];
            beta[8] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 8 + v];
            beta[9] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 9 + v];
            beta[10] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 10 + v];
            beta[11] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 11 + v];
            beta[12] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 12 + v];
            beta[13] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 13 + v];
            beta[14] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 14 + v];
            beta[15] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 15 + v];
            beta[16] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 16 + v];
            beta[17] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 17 + v];
            beta[18] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 18 + v];
            beta[19] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 19 + v];
            beta[20] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 20 + v];
            beta[21] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 21 + v];
            beta[22] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 22 + v];
            
            break;
            
        case 24:
            
            beta[0] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 0 + v];
            beta[1] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 1 + v];
            beta[2] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 2 + v];
            beta[3] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 3 + v];
            beta[4] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 4 + v];
            beta[5] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 5 + v];
            beta[6] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 6 + v];
            beta[7] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 7 + v];
            beta[8] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 8 + v];
            beta[9] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 9 + v];
            beta[10] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 10 + v];
            beta[11] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 11 + v];
            beta[12] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 12 + v];
            beta[13] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 13 + v];
            beta[14] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 14 + v];
            beta[15] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 15 + v];
            beta[16] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 16 + v];
            beta[17] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 17 + v];
            beta[18] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 18 + v];
            beta[19] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 19 + v];
            beta[20] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 20 + v];
            beta[21] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 21 + v];
            beta[22] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 22 + v];
            beta[23] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 23 + v];
            
            break;
            
        case 25:
            
            beta[0] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 0 + v];
            beta[1] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 1 + v];
            beta[2] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 2 + v];
            beta[3] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 3 + v];
            beta[4] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 4 + v];
            beta[5] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 5 + v];
            beta[6] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 6 + v];
            beta[7] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 7 + v];
            beta[8] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 8 + v];
            beta[9] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 9 + v];
            beta[10] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 10 + v];
            beta[11] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 11 + v];
            beta[12] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 12 + v];
            beta[13] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 13 + v];
            beta[14] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 14 + v];
            beta[15] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 15 + v];
            beta[16] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 16 + v];
            beta[17] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 17 + v];
            beta[18] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 18 + v];
            beta[19] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 19 + v];
            beta[20] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 20 + v];
            beta[21] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 21 + v];
            beta[22] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 22 + v];
            beta[23] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 23 + v];
            beta[24] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 24 + v];
            
            break;
            
            
        default:
            1;
            break;
    }
    
    return 0;
}






// For first level, volumes already permuted
float CalculateEpsFirstLevel(__private float eps,
                             __private float* beta,
                             __constant float* c_X_GLM,
                             int v,
                             int NUMBER_OF_VOLUMES,
                             int NUMBER_OF_REGRESSORS)
{
    switch(NUMBER_OF_REGRESSORS)
    {
        case 1:
            
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 0 + v] * beta[0];
            
            break;
            
        case 2:
            
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 0 + v] * beta[0];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 1 + v] * beta[1];
            
            break;
            
        case 3:
            
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 0 + v] * beta[0];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 1 + v] * beta[1];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 2 + v] * beta[2];
            
            break;
            
        case 4:
            
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 0 + v] * beta[0];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 1 + v] * beta[1];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 2 + v] * beta[2];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 3 + v] * beta[3];
            
            break;
            
        case 5:
            
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 0 + v] * beta[0];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 1 + v] * beta[1];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 2 + v] * beta[2];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 3 + v] * beta[3];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 4 + v] * beta[4];
            
            break;
            
        case 6:
            
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 0 + v] * beta[0];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 1 + v] * beta[1];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 2 + v] * beta[2];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 3 + v] * beta[3];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 4 + v] * beta[4];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 5 + v] * beta[5];
            
            break;
            
        case 7:
            
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 0 + v] * beta[0];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 1 + v] * beta[1];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 2 + v] * beta[2];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 3 + v] * beta[3];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 4 + v] * beta[4];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 5 + v] * beta[5];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 6 + v] * beta[6];
            
            break;
            
        case 8:
            
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 0 + v] * beta[0];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 1 + v] * beta[1];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 2 + v] * beta[2];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 3 + v] * beta[3];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 4 + v] * beta[4];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 5 + v] * beta[5];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 6 + v] * beta[6];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 7 + v] * beta[7];
            
            break;
            
        case 9:
            
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 0 + v] * beta[0];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 1 + v] * beta[1];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 2 + v] * beta[2];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 3 + v] * beta[3];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 4 + v] * beta[4];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 5 + v] * beta[5];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 6 + v] * beta[6];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 7 + v] * beta[7];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 8 + v] * beta[8];
            
            break;
            
        case 10:
            
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 0 + v] * beta[0];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 1 + v] * beta[1];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 2 + v] * beta[2];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 3 + v] * beta[3];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 4 + v] * beta[4];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 5 + v] * beta[5];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 6 + v] * beta[6];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 7 + v] * beta[7];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 8 + v] * beta[8];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 9 + v] * beta[9];
            
            break;
            
        case 11:
            
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 0 + v] * beta[0];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 1 + v] * beta[1];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 2 + v] * beta[2];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 3 + v] * beta[3];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 4 + v] * beta[4];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 5 + v] * beta[5];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 6 + v] * beta[6];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 7 + v] * beta[7];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 8 + v] * beta[8];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 9 + v] * beta[9];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 10 + v] * beta[10];
            
            break;
            
        case 12:
            
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 0 + v] * beta[0];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 1 + v] * beta[1];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 2 + v] * beta[2];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 3 + v] * beta[3];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 4 + v] * beta[4];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 5 + v] * beta[5];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 6 + v] * beta[6];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 7 + v] * beta[7];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 8 + v] * beta[8];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 9 + v] * beta[9];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 10 + v] * beta[10];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 11 + v] * beta[11];
            
            break;
            
        case 13:
            
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 0 + v] * beta[0];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 1 + v] * beta[1];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 2 + v] * beta[2];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 3 + v] * beta[3];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 4 + v] * beta[4];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 5 + v] * beta[5];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 6 + v] * beta[6];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 7 + v] * beta[7];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 8 + v] * beta[8];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 9 + v] * beta[9];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 10 + v] * beta[10];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 11 + v] * beta[11];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 12 + v] * beta[12];
            
            break;
            
        case 14:
            
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 0 + v] * beta[0];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 1 + v] * beta[1];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 2 + v] * beta[2];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 3 + v] * beta[3];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 4 + v] * beta[4];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 5 + v] * beta[5];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 6 + v] * beta[6];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 7 + v] * beta[7];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 8 + v] * beta[8];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 9 + v] * beta[9];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 10 + v] * beta[10];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 11 + v] * beta[11];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 12 + v] * beta[12];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 13 + v] * beta[13];
            
            break;
            
        case 15:
            
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 0 + v] * beta[0];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 1 + v] * beta[1];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 2 + v] * beta[2];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 3 + v] * beta[3];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 4 + v] * beta[4];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 5 + v] * beta[5];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 6 + v] * beta[6];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 7 + v] * beta[7];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 8 + v] * beta[8];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 9 + v] * beta[9];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 10 + v] * beta[10];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 11 + v] * beta[11];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 12 + v] * beta[12];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 13 + v] * beta[13];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 14 + v] * beta[14];
            
            break;
            
        case 16:
            
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 0 + v] * beta[0];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 1 + v] * beta[1];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 2 + v] * beta[2];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 3 + v] * beta[3];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 4 + v] * beta[4];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 5 + v] * beta[5];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 6 + v] * beta[6];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 7 + v] * beta[7];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 8 + v] * beta[8];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 9 + v] * beta[9];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 10 + v] * beta[10];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 11 + v] * beta[11];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 12 + v] * beta[12];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 13 + v] * beta[13];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 14 + v] * beta[14];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 15 + v] * beta[15];
            
            break;
            
        case 17:
            
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 0 + v] * beta[0];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 1 + v] * beta[1];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 2 + v] * beta[2];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 3 + v] * beta[3];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 4 + v] * beta[4];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 5 + v] * beta[5];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 6 + v] * beta[6];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 7 + v] * beta[7];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 8 + v] * beta[8];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 9 + v] * beta[9];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 10 + v] * beta[10];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 11 + v] * beta[11];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 12 + v] * beta[12];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 13 + v] * beta[13];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 14 + v] * beta[14];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 15 + v] * beta[15];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 16 + v] * beta[16];
            
            break;
            
        case 18:
            
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 0 + v] * beta[0];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 1 + v] * beta[1];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 2 + v] * beta[2];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 3 + v] * beta[3];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 4 + v] * beta[4];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 5 + v] * beta[5];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 6 + v] * beta[6];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 7 + v] * beta[7];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 8 + v] * beta[8];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 9 + v] * beta[9];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 10 + v] * beta[10];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 11 + v] * beta[11];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 12 + v] * beta[12];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 13 + v] * beta[13];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 14 + v] * beta[14];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 15 + v] * beta[15];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 16 + v] * beta[16];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 17 + v] * beta[17];
            
            break;
            
        case 19:
            
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 0 + v] * beta[0];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 1 + v] * beta[1];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 2 + v] * beta[2];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 3 + v] * beta[3];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 4 + v] * beta[4];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 5 + v] * beta[5];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 6 + v] * beta[6];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 7 + v] * beta[7];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 8 + v] * beta[8];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 9 + v] * beta[9];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 10 + v] * beta[10];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 11 + v] * beta[11];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 12 + v] * beta[12];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 13 + v] * beta[13];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 14 + v] * beta[14];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 15 + v] * beta[15];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 16 + v] * beta[16];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 17 + v] * beta[17];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 18 + v] * beta[18];
            
            break;
            
        case 20:
            
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 0 + v] * beta[0];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 1 + v] * beta[1];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 2 + v] * beta[2];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 3 + v] * beta[3];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 4 + v] * beta[4];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 5 + v] * beta[5];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 6 + v] * beta[6];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 7 + v] * beta[7];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 8 + v] * beta[8];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 9 + v] * beta[9];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 10 + v] * beta[10];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 11 + v] * beta[11];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 12 + v] * beta[12];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 13 + v] * beta[13];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 14 + v] * beta[14];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 15 + v] * beta[15];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 16 + v] * beta[16];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 17 + v] * beta[17];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 18 + v] * beta[18];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 19 + v] * beta[19];
            
            break;
            
        case 21:
            
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 0 + v] * beta[0];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 1 + v] * beta[1];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 2 + v] * beta[2];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 3 + v] * beta[3];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 4 + v] * beta[4];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 5 + v] * beta[5];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 6 + v] * beta[6];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 7 + v] * beta[7];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 8 + v] * beta[8];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 9 + v] * beta[9];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 10 + v] * beta[10];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 11 + v] * beta[11];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 12 + v] * beta[12];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 13 + v] * beta[13];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 14 + v] * beta[14];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 15 + v] * beta[15];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 16 + v] * beta[16];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 17 + v] * beta[17];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 18 + v] * beta[18];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 19 + v] * beta[19];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 20 + v] * beta[20];
            
            break;
            
        case 22:
            
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 0 + v] * beta[0];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 1 + v] * beta[1];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 2 + v] * beta[2];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 3 + v] * beta[3];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 4 + v] * beta[4];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 5 + v] * beta[5];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 6 + v] * beta[6];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 7 + v] * beta[7];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 8 + v] * beta[8];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 9 + v] * beta[9];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 10 + v] * beta[10];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 11 + v] * beta[11];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 12 + v] * beta[12];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 13 + v] * beta[13];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 14 + v] * beta[14];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 15 + v] * beta[15];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 16 + v] * beta[16];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 17 + v] * beta[17];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 18 + v] * beta[18];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 19 + v] * beta[19];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 20 + v] * beta[20];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 21 + v] * beta[21];
            
            break;
            
        case 23:
            
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 0 + v] * beta[0];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 1 + v] * beta[1];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 2 + v] * beta[2];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 3 + v] * beta[3];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 4 + v] * beta[4];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 5 + v] * beta[5];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 6 + v] * beta[6];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 7 + v] * beta[7];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 8 + v] * beta[8];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 9 + v] * beta[9];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 10 + v] * beta[10];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 11 + v] * beta[11];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 12 + v] * beta[12];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 13 + v] * beta[13];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 14 + v] * beta[14];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 15 + v] * beta[15];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 16 + v] * beta[16];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 17 + v] * beta[17];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 18 + v] * beta[18];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 19 + v] * beta[19];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 20 + v] * beta[20];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 21 + v] * beta[21];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 22 + v] * beta[22];
            
            break;
            
        case 24:
            
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 0 + v] * beta[0];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 1 + v] * beta[1];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 2 + v] * beta[2];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 3 + v] * beta[3];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 4 + v] * beta[4];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 5 + v] * beta[5];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 6 + v] * beta[6];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 7 + v] * beta[7];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 8 + v] * beta[8];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 9 + v] * beta[9];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 10 + v] * beta[10];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 11 + v] * beta[11];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 12 + v] * beta[12];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 13 + v] * beta[13];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 14 + v] * beta[14];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 15 + v] * beta[15];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 16 + v] * beta[16];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 17 + v] * beta[17];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 18 + v] * beta[18];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 19 + v] * beta[19];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 20 + v] * beta[20];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 21 + v] * beta[21];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 22 + v] * beta[22];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 23 + v] * beta[23];
            
            break;
            
        case 25:
            
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 0 + v] * beta[0];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 1 + v] * beta[1];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 2 + v] * beta[2];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 3 + v] * beta[3];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 4 + v] * beta[4];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 5 + v] * beta[5];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 6 + v] * beta[6];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 7 + v] * beta[7];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 8 + v] * beta[8];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 9 + v] * beta[9];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 10 + v] * beta[10];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 11 + v] * beta[11];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 12 + v] * beta[12];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 13 + v] * beta[13];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 14 + v] * beta[14];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 15 + v] * beta[15];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 16 + v] * beta[16];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 17 + v] * beta[17];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 18 + v] * beta[18];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 19 + v] * beta[19];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 20 + v] * beta[20];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 21 + v] * beta[21];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 22 + v] * beta[22];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 23 + v] * beta[23];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 24 + v] * beta[24];
            
            break;
            
        default:
            1;
            break;
    }
    
    return eps;
}




float CalculateContrastValue(__private float* beta, __constant float* c_Contrasts, int c, int NUMBER_OF_REGRESSORS)
{
    float contrast_value = 0.0f;
    
    switch(NUMBER_OF_REGRESSORS)
    {
        case 1:
            
            contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 0] * beta[0];
            
            break;
            
        case 2:
            
            contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 0] * beta[0];
            contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 1] * beta[1];
            
            break;
            
        case 3:
            
            contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 0] * beta[0];
            contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 1] * beta[1];
            contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 2] * beta[2];
            
            break;
            
        case 4:
            
            contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 0] * beta[0];
            contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 1] * beta[1];
            contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 2] * beta[2];
            contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 3] * beta[3];
            
            break;
            
        case 5:
            
            contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 0] * beta[0];
            contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 1] * beta[1];
            contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 2] * beta[2];
            contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 3] * beta[3];
            contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 4] * beta[4];
            
            break;
            
        case 6:
            
            contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 0] * beta[0];
            contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 1] * beta[1];
            contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 2] * beta[2];
            contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 3] * beta[3];
            contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 4] * beta[4];
            contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 5] * beta[5];
            
            break;
            
        case 7:
            
            contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 0] * beta[0];
            contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 1] * beta[1];
            contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 2] * beta[2];
            contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 3] * beta[3];
            contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 4] * beta[4];
            contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 5] * beta[5];
            contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 6] * beta[6];
            
            break;
            
        case 8:
            
            contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 0] * beta[0];
            contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 1] * beta[1];
            contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 2] * beta[2];
            contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 3] * beta[3];
            contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 4] * beta[4];
            contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 5] * beta[5];
            contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 6] * beta[6];
            contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 7] * beta[7];
            
            break;
            
        case 9:
            
            contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 0] * beta[0];
            contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 1] * beta[1];
            contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 2] * beta[2];
            contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 3] * beta[3];
            contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 4] * beta[4];
            contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 5] * beta[5];
            contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 6] * beta[6];
            contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 7] * beta[7];
            contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 8] * beta[8];
            
            break;
            
        case 10:
            
            contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 0] * beta[0];
            contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 1] * beta[1];
            contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 2] * beta[2];
            contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 3] * beta[3];
            contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 4] * beta[4];
            contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 5] * beta[5];
            contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 6] * beta[6];
            contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 7] * beta[7];
            contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 8] * beta[8];
            contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 9] * beta[9];
            
            break;
            
        case 11:
            
            contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 0] * beta[0];
            contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 1] * beta[1];
            contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 2] * beta[2];
            contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 3] * beta[3];
            contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 4] * beta[4];
            contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 5] * beta[5];
            contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 6] * beta[6];
            contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 7] * beta[7];
            contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 8] * beta[8];
            contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 9] * beta[9];
            contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 10] * beta[10];
            
            break;
            
        case 12:
            
            contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 0] * beta[0];
            contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 1] * beta[1];
            contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 2] * beta[2];
            contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 3] * beta[3];
            contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 4] * beta[4];
            contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 5] * beta[5];
            contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 6] * beta[6];
            contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 7] * beta[7];
            contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 8] * beta[8];
            contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 9] * beta[9];
            contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 10] * beta[10];
            contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 11] * beta[11];
            
            break;
            
        case 13:
            
            contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 0] * beta[0];
            contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 1] * beta[1];
            contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 2] * beta[2];
            contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 3] * beta[3];
            contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 4] * beta[4];
            contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 5] * beta[5];
            contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 6] * beta[6];
            contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 7] * beta[7];
            contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 8] * beta[8];
            contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 9] * beta[9];
            contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 10] * beta[10];
            contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 11] * beta[11];
            contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 12] * beta[12];
            
            break;
            
        case 14:
            
            contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 0] * beta[0];
            contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 1] * beta[1];
            contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 2] * beta[2];
            contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 3] * beta[3];
            contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 4] * beta[4];
            contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 5] * beta[5];
            contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 6] * beta[6];
            contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 7] * beta[7];
            contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 8] * beta[8];
            contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 9] * beta[9];
            contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 10] * beta[10];
            contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 11] * beta[11];
            contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 12] * beta[12];
            contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 13] * beta[13];
            
            break;
            
        case 15:
            
            contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 0] * beta[0];
            contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 1] * beta[1];
            contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 2] * beta[2];
            contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 3] * beta[3];
            contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 4] * beta[4];
            contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 5] * beta[5];
            contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 6] * beta[6];
            contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 7] * beta[7];
            contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 8] * beta[8];
            contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 9] * beta[9];
            contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 10] * beta[10];
            contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 11] * beta[11];
            contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 12] * beta[12];
            contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 13] * beta[13];
            contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 14] * beta[14];
            
            break;
            
        case 16:
            
            contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 0] * beta[0];
            contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 1] * beta[1];
            contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 2] * beta[2];
            contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 3] * beta[3];
            contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 4] * beta[4];
            contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 5] * beta[5];
            contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 6] * beta[6];
            contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 7] * beta[7];
            contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 8] * beta[8];
            contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 9] * beta[9];
            contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 10] * beta[10];
            contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 11] * beta[11];
            contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 12] * beta[12];
            contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 13] * beta[13];
            contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 14] * beta[14];
            contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 15] * beta[15];
            
            break;
            
        case 17:
            
            contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 0] * beta[0];
            contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 1] * beta[1];
            contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 2] * beta[2];
            contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 3] * beta[3];
            contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 4] * beta[4];
            contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 5] * beta[5];
            contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 6] * beta[6];
            contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 7] * beta[7];
            contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 8] * beta[8];
            contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 9] * beta[9];
            contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 10] * beta[10];
            contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 11] * beta[11];
            contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 12] * beta[12];
            contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 13] * beta[13];
            contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 14] * beta[14];
            contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 15] * beta[15];
            contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 16] * beta[16];
            
            break;
            
        case 18:
            
            contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 0] * beta[0];
            contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 1] * beta[1];
            contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 2] * beta[2];
            contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 3] * beta[3];
            contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 4] * beta[4];
            contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 5] * beta[5];
            contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 6] * beta[6];
            contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 7] * beta[7];
            contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 8] * beta[8];
            contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 9] * beta[9];
            contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 10] * beta[10];
            contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 11] * beta[11];
            contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 12] * beta[12];
            contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 13] * beta[13];
            contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 14] * beta[14];
            contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 15] * beta[15];
            contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 16] * beta[16];
            contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 17] * beta[17];
            
            break;
            
        case 19:
            
            contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 0] * beta[0];
            contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 1] * beta[1];
            contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 2] * beta[2];
            contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 3] * beta[3];
            contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 4] * beta[4];
            contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 5] * beta[5];
            contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 6] * beta[6];
            contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 7] * beta[7];
            contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 8] * beta[8];
            contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 9] * beta[9];
            contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 10] * beta[10];
            contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 11] * beta[11];
            contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 12] * beta[12];
            contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 13] * beta[13];
            contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 14] * beta[14];
            contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 15] * beta[15];
            contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 16] * beta[16];
            contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 17] * beta[17];
            contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 18] * beta[18];
            
            break;
            
        case 20:
            
            contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 0] * beta[0];
            contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 1] * beta[1];
            contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 2] * beta[2];
            contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 3] * beta[3];
            contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 4] * beta[4];
            contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 5] * beta[5];
            contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 6] * beta[6];
            contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 7] * beta[7];
            contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 8] * beta[8];
            contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 9] * beta[9];
            contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 10] * beta[10];
            contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 11] * beta[11];
            contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 12] * beta[12];
            contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 13] * beta[13];
            contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 14] * beta[14];
            contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 15] * beta[15];
            contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 16] * beta[16];
            contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 17] * beta[17];
            contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 18] * beta[18];
            contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 19] * beta[19];
            
            break;
            
        case 21:
            
            contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 0] * beta[0];
            contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 1] * beta[1];
            contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 2] * beta[2];
            contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 3] * beta[3];
            contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 4] * beta[4];
            contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 5] * beta[5];
            contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 6] * beta[6];
            contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 7] * beta[7];
            contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 8] * beta[8];
            contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 9] * beta[9];
            contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 10] * beta[10];
            contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 11] * beta[11];
            contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 12] * beta[12];
            contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 13] * beta[13];
            contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 14] * beta[14];
            contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 15] * beta[15];
            contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 16] * beta[16];
            contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 17] * beta[17];
            contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 18] * beta[18];
            contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 19] * beta[19];
            contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 20] * beta[20];
            
            break;
            
        case 22:
            
            contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 0] * beta[0];
            contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 1] * beta[1];
            contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 2] * beta[2];
            contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 3] * beta[3];
            contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 4] * beta[4];
            contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 5] * beta[5];
            contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 6] * beta[6];
            contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 7] * beta[7];
            contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 8] * beta[8];
            contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 9] * beta[9];
            contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 10] * beta[10];
            contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 11] * beta[11];
            contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 12] * beta[12];
            contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 13] * beta[13];
            contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 14] * beta[14];
            contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 15] * beta[15];
            contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 16] * beta[16];
            contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 17] * beta[17];
            contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 18] * beta[18];
            contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 19] * beta[19];
            contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 20] * beta[20];
            contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 21] * beta[21];
            
            break;
            
        case 23:
            
            contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 0] * beta[0];
            contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 1] * beta[1];
            contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 2] * beta[2];
            contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 3] * beta[3];
            contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 4] * beta[4];
            contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 5] * beta[5];
            contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 6] * beta[6];
            contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 7] * beta[7];
            contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 8] * beta[8];
            contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 9] * beta[9];
            contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 10] * beta[10];
            contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 11] * beta[11];
            contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 12] * beta[12];
            contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 13] * beta[13];
            contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 14] * beta[14];
            contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 15] * beta[15];
            contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 16] * beta[16];
            contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 17] * beta[17];
            contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 18] * beta[18];
            contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 19] * beta[19];
            contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 20] * beta[20];
            contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 21] * beta[21];
            contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 22] * beta[22];
            
            break;
            
        case 24:
            
            contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 0] * beta[0];
            contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 1] * beta[1];
            contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 2] * beta[2];
            contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 3] * beta[3];
            contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 4] * beta[4];
            contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 5] * beta[5];
            contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 6] * beta[6];
            contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 7] * beta[7];
            contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 8] * beta[8];
            contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 9] * beta[9];
            contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 10] * beta[10];
            contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 11] * beta[11];
            contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 12] * beta[12];
            contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 13] * beta[13];
            contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 14] * beta[14];
            contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 15] * beta[15];
            contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 16] * beta[16];
            contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 17] * beta[17];
            contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 18] * beta[18];
            contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 19] * beta[19];
            contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 20] * beta[20];
            contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 21] * beta[21];
            contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 22] * beta[22];
            contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 23] * beta[23];
            
            break;
            
        case 25:
            
            contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 0] * beta[0];
            contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 1] * beta[1];
            contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 2] * beta[2];
            contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 3] * beta[3];
            contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 4] * beta[4];
            contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 5] * beta[5];
            contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 6] * beta[6];
            contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 7] * beta[7];
            contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 8] * beta[8];
            contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 9] * beta[9];
            contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 10] * beta[10];
            contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 11] * beta[11];
            contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 12] * beta[12];
            contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 13] * beta[13];
            contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 14] * beta[14];
            contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 15] * beta[15];
            contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 16] * beta[16];
            contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 17] * beta[17];
            contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 18] * beta[18];
            contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 19] * beta[19];
            contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 20] * beta[20];
            contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 21] * beta[21];
            contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 22] * beta[22];
            contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 23] * beta[23];
            contrast_value += c_Contrasts[NUMBER_OF_REGRESSORS * c + 24] * beta[24];
            
            break;
            
            
        default:
            1;
            break;
    }
    
    return contrast_value;
}




__kernel void CalculateStatisticalMapsGLMTTestFirstLevelPermutation(__global float* Statistical_Maps,
                                                                    __global const float* Volumes,
                                                                    __global const float* Mask,
                                                                    __constant float* c_X_GLM,
                                                                    __constant float* c_xtxxt_GLM,
                                                                    __constant float* c_Contrasts,	
                                                                    __constant float* c_ctxtxc_GLM,
                                                                    __private int DATA_W,
                                                                    __private int DATA_H,
                                                                    __private int DATA_D,
                                                                    __private int NUMBER_OF_VOLUMES,
                                                                    __private int NUMBER_OF_REGRESSORS,
                                                                    __private int NUMBER_OF_CONTRASTS,
                                                                    __private int contrast)
{	
    int x = get_global_id(0);
    int y = get_global_id(1);
    int z = get_global_id(2);
    
    int3 tIdx = {get_local_id(0), get_local_id(1), get_local_id(2)};
    
    if (x >= DATA_W || y >= DATA_H || z >= DATA_D)
        return;
    
    if ( Mask[Calculate3DIndex(x,y,z,DATA_W,DATA_H)] != 1.0f )
        return;
    
    int t = 0;
    float eps, meaneps, vareps;
    float beta[25];
    
    // Reset beta weights
    beta[0] = 0.0f;
    beta[1] = 0.0f;
    beta[2] = 0.0f;
    beta[3] = 0.0f;
    beta[4] = 0.0f;
    beta[5] = 0.0f;
    beta[6] = 0.0f;
    beta[7] = 0.0f;
    beta[8] = 0.0f;
    beta[9] = 0.0f;
    beta[10] = 0.0f;
    beta[11] = 0.0f;
    beta[12] = 0.0f;
    beta[13] = 0.0f;
    beta[14] = 0.0f;
    beta[15] = 0.0f;
    beta[16] = 0.0f;
    beta[17] = 0.0f;
    beta[18] = 0.0f;
    beta[19] = 0.0f;
    beta[20] = 0.0f;
    beta[21] = 0.0f;
    beta[22] = 0.0f;
    beta[23] = 0.0f;
    beta[24] = 0.0f;
    
    // Calculate betahat, i.e. multiply (x^T x)^(-1) x^T with Y
    // Loop over volumes
    for (int v = 0; v < NUMBER_OF_VOLUMES; v++)
    {
        float value = Volumes[Calculate4DIndex(x,y,z,v,DATA_W,DATA_H,DATA_D)];
        
        // Loop over regressors using unrolled code for performance
        CalculateBetaWeightsFirstLevel(beta, value, c_xtxxt_GLM, v, NUMBER_OF_VOLUMES, NUMBER_OF_REGRESSORS);
    }
    
    // Calculate the mean and variance of the error eps
    meaneps = 0.0f;
    vareps = 0.0f;
    float n = 0.0f;
    for (int v = 0; v < NUMBER_OF_VOLUMES; v++)
    {
        eps = Volumes[Calculate4DIndex(x,y,z,v,DATA_W,DATA_H,DATA_D)];
        eps = CalculateEpsFirstLevel(eps, beta, c_X_GLM, v, NUMBER_OF_VOLUMES, NUMBER_OF_REGRESSORS);
        
        n += 1.0f;
        float delta = eps - meaneps;
        meaneps += delta/n;
        vareps += delta * (eps - meaneps);
    }
    vareps = vareps / (n - 1.0f);
    
    // Calculate t-values
    float contrast_value = CalculateContrastValue(beta, c_Contrasts, contrast, NUMBER_OF_REGRESSORS);
    Statistical_Maps[Calculate3DIndex(x,y,z,DATA_W,DATA_H)] = contrast_value * rsqrt(vareps * c_ctxtxc_GLM[contrast]);	
}

