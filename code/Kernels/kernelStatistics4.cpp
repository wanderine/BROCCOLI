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



int CalculateBetaWeightsSecondLevel(__private float* beta,
                                    __private float value,
                                    __constant float* c_xtxxt_GLM,
                                    int v,
                                    __constant unsigned short int* c_Permutation_Vector,
                                    int NUMBER_OF_VOLUMES,
                                    int NUMBER_OF_REGRESSORS)
{
    switch(NUMBER_OF_REGRESSORS)
    {
        case 1:
            
            beta[0] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 0 + c_Permutation_Vector[v]];
            
            break;
            
        case 2:
            
            beta[0] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 0 + c_Permutation_Vector[v]];
            beta[1] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 1 + c_Permutation_Vector[v]];
            
            break;
            
        case 3:
            
            beta[0] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 0 + c_Permutation_Vector[v]];
            beta[1] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 1 + c_Permutation_Vector[v]];
            beta[2] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 2 + c_Permutation_Vector[v]];
            
            break;
            
        case 4:
            
            beta[0] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 0 + c_Permutation_Vector[v]];
            beta[1] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 1 + c_Permutation_Vector[v]];
            beta[2] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 2 + c_Permutation_Vector[v]];
            beta[3] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 3 + c_Permutation_Vector[v]];
            
            break;
            
        case 5:
            
            beta[0] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 0 + c_Permutation_Vector[v]];
            beta[1] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 1 + c_Permutation_Vector[v]];
            beta[2] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 2 + c_Permutation_Vector[v]];
            beta[3] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 3 + c_Permutation_Vector[v]];
            beta[4] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 4 + c_Permutation_Vector[v]];
            
            break;
            
        case 6:
            
            beta[0] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 0 + c_Permutation_Vector[v]];
            beta[1] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 1 + c_Permutation_Vector[v]];
            beta[2] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 2 + c_Permutation_Vector[v]];
            beta[3] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 3 + c_Permutation_Vector[v]];
            beta[4] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 4 + c_Permutation_Vector[v]];
            beta[5] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 5 + c_Permutation_Vector[v]];
            
            break;
            
        case 7:
            
            beta[0] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 0 + c_Permutation_Vector[v]];
            beta[1] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 1 + c_Permutation_Vector[v]];
            beta[2] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 2 + c_Permutation_Vector[v]];
            beta[3] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 3 + c_Permutation_Vector[v]];
            beta[4] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 4 + c_Permutation_Vector[v]];
            beta[5] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 5 + c_Permutation_Vector[v]];
            beta[6] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 6 + c_Permutation_Vector[v]];
            
            break;
            
        case 8:
            
            beta[0] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 0 + c_Permutation_Vector[v]];
            beta[1] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 1 + c_Permutation_Vector[v]];
            beta[2] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 2 + c_Permutation_Vector[v]];
            beta[3] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 3 + c_Permutation_Vector[v]];
            beta[4] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 4 + c_Permutation_Vector[v]];
            beta[5] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 5 + c_Permutation_Vector[v]];
            beta[6] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 6 + c_Permutation_Vector[v]];
            beta[7] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 7 + c_Permutation_Vector[v]];
            
            break;
            
        case 9:
            
            beta[0] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 0 + c_Permutation_Vector[v]];
            beta[1] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 1 + c_Permutation_Vector[v]];
            beta[2] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 2 + c_Permutation_Vector[v]];
            beta[3] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 3 + c_Permutation_Vector[v]];
            beta[4] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 4 + c_Permutation_Vector[v]];
            beta[5] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 5 + c_Permutation_Vector[v]];
            beta[6] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 6 + c_Permutation_Vector[v]];
            beta[7] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 7 + c_Permutation_Vector[v]];
            beta[8] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 8 + c_Permutation_Vector[v]];
            
            break;
            
        case 10:
            
            beta[0] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 0 + c_Permutation_Vector[v]];
            beta[1] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 1 + c_Permutation_Vector[v]];
            beta[2] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 2 + c_Permutation_Vector[v]];
            beta[3] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 3 + c_Permutation_Vector[v]];
            beta[4] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 4 + c_Permutation_Vector[v]];
            beta[5] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 5 + c_Permutation_Vector[v]];
            beta[6] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 6 + c_Permutation_Vector[v]];
            beta[7] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 7 + c_Permutation_Vector[v]];
            beta[8] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 8 + c_Permutation_Vector[v]];
            beta[9] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 9 + c_Permutation_Vector[v]];
            
            break;
            
        case 11:
            
            beta[0] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 0 + c_Permutation_Vector[v]];
            beta[1] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 1 + c_Permutation_Vector[v]];
            beta[2] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 2 + c_Permutation_Vector[v]];
            beta[3] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 3 + c_Permutation_Vector[v]];
            beta[4] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 4 + c_Permutation_Vector[v]];
            beta[5] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 5 + c_Permutation_Vector[v]];
            beta[6] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 6 + c_Permutation_Vector[v]];
            beta[7] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 7 + c_Permutation_Vector[v]];
            beta[8] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 8 + c_Permutation_Vector[v]];
            beta[9] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 9 + c_Permutation_Vector[v]];
            beta[10] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 10 + c_Permutation_Vector[v]];
            
            break;
            
        case 12:
            
            beta[0] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 0 + c_Permutation_Vector[v]];
            beta[1] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 1 + c_Permutation_Vector[v]];
            beta[2] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 2 + c_Permutation_Vector[v]];
            beta[3] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 3 + c_Permutation_Vector[v]];
            beta[4] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 4 + c_Permutation_Vector[v]];
            beta[5] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 5 + c_Permutation_Vector[v]];
            beta[6] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 6 + c_Permutation_Vector[v]];
            beta[7] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 7 + c_Permutation_Vector[v]];
            beta[8] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 8 + c_Permutation_Vector[v]];
            beta[9] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 9 + c_Permutation_Vector[v]];
            beta[10] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 10 + c_Permutation_Vector[v]];
            beta[11] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 11 + c_Permutation_Vector[v]];
            
            break;
            
        case 13:
            
            beta[0] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 0 + c_Permutation_Vector[v]];
            beta[1] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 1 + c_Permutation_Vector[v]];
            beta[2] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 2 + c_Permutation_Vector[v]];
            beta[3] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 3 + c_Permutation_Vector[v]];
            beta[4] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 4 + c_Permutation_Vector[v]];
            beta[5] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 5 + c_Permutation_Vector[v]];
            beta[6] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 6 + c_Permutation_Vector[v]];
            beta[7] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 7 + c_Permutation_Vector[v]];
            beta[8] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 8 + c_Permutation_Vector[v]];
            beta[9] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 9 + c_Permutation_Vector[v]];
            beta[10] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 10 + c_Permutation_Vector[v]];
            beta[11] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 11 + c_Permutation_Vector[v]];
            beta[12] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 12 + c_Permutation_Vector[v]];
            
            break;
            
        case 14:
            
            beta[0] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 0 + c_Permutation_Vector[v]];
            beta[1] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 1 + c_Permutation_Vector[v]];
            beta[2] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 2 + c_Permutation_Vector[v]];
            beta[3] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 3 + c_Permutation_Vector[v]];
            beta[4] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 4 + c_Permutation_Vector[v]];
            beta[5] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 5 + c_Permutation_Vector[v]];
            beta[6] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 6 + c_Permutation_Vector[v]];
            beta[7] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 7 + c_Permutation_Vector[v]];
            beta[8] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 8 + c_Permutation_Vector[v]];
            beta[9] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 9 + c_Permutation_Vector[v]];
            beta[10] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 10 + c_Permutation_Vector[v]];
            beta[11] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 11 + c_Permutation_Vector[v]];
            beta[12] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 12 + c_Permutation_Vector[v]];
            beta[13] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 13 + c_Permutation_Vector[v]];
            
            break;
            
        case 15:
            
            beta[0] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 0 + c_Permutation_Vector[v]];
            beta[1] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 1 + c_Permutation_Vector[v]];
            beta[2] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 2 + c_Permutation_Vector[v]];
            beta[3] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 3 + c_Permutation_Vector[v]];
            beta[4] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 4 + c_Permutation_Vector[v]];
            beta[5] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 5 + c_Permutation_Vector[v]];
            beta[6] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 6 + c_Permutation_Vector[v]];
            beta[7] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 7 + c_Permutation_Vector[v]];
            beta[8] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 8 + c_Permutation_Vector[v]];
            beta[9] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 9 + c_Permutation_Vector[v]];
            beta[10] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 10 + c_Permutation_Vector[v]];
            beta[11] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 11 + c_Permutation_Vector[v]];
            beta[12] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 12 + c_Permutation_Vector[v]];
            beta[13] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 13 + c_Permutation_Vector[v]];
            beta[14] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 14 + c_Permutation_Vector[v]];
            
            break;
            
        case 16:
            
            beta[0] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 0 + c_Permutation_Vector[v]];
            beta[1] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 1 + c_Permutation_Vector[v]];
            beta[2] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 2 + c_Permutation_Vector[v]];
            beta[3] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 3 + c_Permutation_Vector[v]];
            beta[4] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 4 + c_Permutation_Vector[v]];
            beta[5] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 5 + c_Permutation_Vector[v]];
            beta[6] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 6 + c_Permutation_Vector[v]];
            beta[7] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 7 + c_Permutation_Vector[v]];
            beta[8] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 8 + c_Permutation_Vector[v]];
            beta[9] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 9 + c_Permutation_Vector[v]];
            beta[10] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 10 + c_Permutation_Vector[v]];
            beta[11] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 11 + c_Permutation_Vector[v]];
            beta[12] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 12 + c_Permutation_Vector[v]];
            beta[13] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 13 + c_Permutation_Vector[v]];
            beta[14] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 14 + c_Permutation_Vector[v]];
            beta[15] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 15 + c_Permutation_Vector[v]];
            
            break;
            
        case 17:
            
            beta[0] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 0 + c_Permutation_Vector[v]];
            beta[1] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 1 + c_Permutation_Vector[v]];
            beta[2] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 2 + c_Permutation_Vector[v]];
            beta[3] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 3 + c_Permutation_Vector[v]];
            beta[4] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 4 + c_Permutation_Vector[v]];
            beta[5] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 5 + c_Permutation_Vector[v]];
            beta[6] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 6 + c_Permutation_Vector[v]];
            beta[7] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 7 + c_Permutation_Vector[v]];
            beta[8] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 8 + c_Permutation_Vector[v]];
            beta[9] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 9 + c_Permutation_Vector[v]];
            beta[10] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 10 + c_Permutation_Vector[v]];
            beta[11] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 11 + c_Permutation_Vector[v]];
            beta[12] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 12 + c_Permutation_Vector[v]];
            beta[13] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 13 + c_Permutation_Vector[v]];
            beta[14] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 14 + c_Permutation_Vector[v]];
            beta[15] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 15 + c_Permutation_Vector[v]];
            beta[16] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 16 + c_Permutation_Vector[v]];
            
            break;
            
        case 18:
            
            beta[0] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 0 + c_Permutation_Vector[v]];
            beta[1] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 1 + c_Permutation_Vector[v]];
            beta[2] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 2 + c_Permutation_Vector[v]];
            beta[3] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 3 + c_Permutation_Vector[v]];
            beta[4] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 4 + c_Permutation_Vector[v]];
            beta[5] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 5 + c_Permutation_Vector[v]];
            beta[6] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 6 + c_Permutation_Vector[v]];
            beta[7] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 7 + c_Permutation_Vector[v]];
            beta[8] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 8 + c_Permutation_Vector[v]];
            beta[9] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 9 + c_Permutation_Vector[v]];
            beta[10] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 10 + c_Permutation_Vector[v]];
            beta[11] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 11 + c_Permutation_Vector[v]];
            beta[12] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 12 + c_Permutation_Vector[v]];
            beta[13] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 13 + c_Permutation_Vector[v]];
            beta[14] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 14 + c_Permutation_Vector[v]];
            beta[15] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 15 + c_Permutation_Vector[v]];
            beta[16] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 16 + c_Permutation_Vector[v]];
            beta[17] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 17 + c_Permutation_Vector[v]];
            
            break;
            
        case 19:
            
            beta[0] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 0 + c_Permutation_Vector[v]];
            beta[1] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 1 + c_Permutation_Vector[v]];
            beta[2] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 2 + c_Permutation_Vector[v]];
            beta[3] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 3 + c_Permutation_Vector[v]];
            beta[4] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 4 + c_Permutation_Vector[v]];
            beta[5] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 5 + c_Permutation_Vector[v]];
            beta[6] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 6 + c_Permutation_Vector[v]];
            beta[7] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 7 + c_Permutation_Vector[v]];
            beta[8] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 8 + c_Permutation_Vector[v]];
            beta[9] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 9 + c_Permutation_Vector[v]];
            beta[10] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 10 + c_Permutation_Vector[v]];
            beta[11] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 11 + c_Permutation_Vector[v]];
            beta[12] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 12 + c_Permutation_Vector[v]];
            beta[13] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 13 + c_Permutation_Vector[v]];
            beta[14] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 14 + c_Permutation_Vector[v]];
            beta[15] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 15 + c_Permutation_Vector[v]];
            beta[16] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 16 + c_Permutation_Vector[v]];
            beta[17] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 17 + c_Permutation_Vector[v]];
            beta[18] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 18 + c_Permutation_Vector[v]];
            
            break;
            
        case 20:
            
            beta[0] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 0 + c_Permutation_Vector[v]];
            beta[1] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 1 + c_Permutation_Vector[v]];
            beta[2] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 2 + c_Permutation_Vector[v]];
            beta[3] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 3 + c_Permutation_Vector[v]];
            beta[4] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 4 + c_Permutation_Vector[v]];
            beta[5] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 5 + c_Permutation_Vector[v]];
            beta[6] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 6 + c_Permutation_Vector[v]];
            beta[7] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 7 + c_Permutation_Vector[v]];
            beta[8] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 8 + c_Permutation_Vector[v]];
            beta[9] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 9 + c_Permutation_Vector[v]];
            beta[10] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 10 + c_Permutation_Vector[v]];
            beta[11] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 11 + c_Permutation_Vector[v]];
            beta[12] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 12 + c_Permutation_Vector[v]];
            beta[13] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 13 + c_Permutation_Vector[v]];
            beta[14] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 14 + c_Permutation_Vector[v]];
            beta[15] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 15 + c_Permutation_Vector[v]];
            beta[16] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 16 + c_Permutation_Vector[v]];
            beta[17] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 17 + c_Permutation_Vector[v]];
            beta[18] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 18 + c_Permutation_Vector[v]];
            beta[19] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 19 + c_Permutation_Vector[v]];
            
            break;
            
        case 21:
            
            beta[0] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 0 + c_Permutation_Vector[v]];
            beta[1] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 1 + c_Permutation_Vector[v]];
            beta[2] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 2 + c_Permutation_Vector[v]];
            beta[3] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 3 + c_Permutation_Vector[v]];
            beta[4] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 4 + c_Permutation_Vector[v]];
            beta[5] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 5 + c_Permutation_Vector[v]];
            beta[6] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 6 + c_Permutation_Vector[v]];
            beta[7] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 7 + c_Permutation_Vector[v]];
            beta[8] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 8 + c_Permutation_Vector[v]];
            beta[9] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 9 + c_Permutation_Vector[v]];
            beta[10] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 10 + c_Permutation_Vector[v]];
            beta[11] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 11 + c_Permutation_Vector[v]];
            beta[12] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 12 + c_Permutation_Vector[v]];
            beta[13] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 13 + c_Permutation_Vector[v]];
            beta[14] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 14 + c_Permutation_Vector[v]];
            beta[15] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 15 + c_Permutation_Vector[v]];
            beta[16] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 16 + c_Permutation_Vector[v]];
            beta[17] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 17 + c_Permutation_Vector[v]];
            beta[18] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 18 + c_Permutation_Vector[v]];
            beta[19] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 19 + c_Permutation_Vector[v]];
            beta[20] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 20 + c_Permutation_Vector[v]];
            
            break;
            
        case 22:
            
            beta[0] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 0 + c_Permutation_Vector[v]];
            beta[1] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 1 + c_Permutation_Vector[v]];
            beta[2] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 2 + c_Permutation_Vector[v]];
            beta[3] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 3 + c_Permutation_Vector[v]];
            beta[4] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 4 + c_Permutation_Vector[v]];
            beta[5] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 5 + c_Permutation_Vector[v]];
            beta[6] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 6 + c_Permutation_Vector[v]];
            beta[7] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 7 + c_Permutation_Vector[v]];
            beta[8] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 8 + c_Permutation_Vector[v]];
            beta[9] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 9 + c_Permutation_Vector[v]];
            beta[10] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 10 + c_Permutation_Vector[v]];
            beta[11] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 11 + c_Permutation_Vector[v]];
            beta[12] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 12 + c_Permutation_Vector[v]];
            beta[13] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 13 + c_Permutation_Vector[v]];
            beta[14] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 14 + c_Permutation_Vector[v]];
            beta[15] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 15 + c_Permutation_Vector[v]];
            beta[16] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 16 + c_Permutation_Vector[v]];
            beta[17] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 17 + c_Permutation_Vector[v]];
            beta[18] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 18 + c_Permutation_Vector[v]];
            beta[19] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 19 + c_Permutation_Vector[v]];
            beta[20] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 20 + c_Permutation_Vector[v]];
            beta[21] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 21 + c_Permutation_Vector[v]];
            
            break;
            
        case 23:
            
            beta[0] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 0 + c_Permutation_Vector[v]];
            beta[1] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 1 + c_Permutation_Vector[v]];
            beta[2] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 2 + c_Permutation_Vector[v]];
            beta[3] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 3 + c_Permutation_Vector[v]];
            beta[4] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 4 + c_Permutation_Vector[v]];
            beta[5] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 5 + c_Permutation_Vector[v]];
            beta[6] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 6 + c_Permutation_Vector[v]];
            beta[7] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 7 + c_Permutation_Vector[v]];
            beta[8] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 8 + c_Permutation_Vector[v]];
            beta[9] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 9 + c_Permutation_Vector[v]];
            beta[10] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 10 + c_Permutation_Vector[v]];
            beta[11] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 11 + c_Permutation_Vector[v]];
            beta[12] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 12 + c_Permutation_Vector[v]];
            beta[13] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 13 + c_Permutation_Vector[v]];
            beta[14] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 14 + c_Permutation_Vector[v]];
            beta[15] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 15 + c_Permutation_Vector[v]];
            beta[16] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 16 + c_Permutation_Vector[v]];
            beta[17] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 17 + c_Permutation_Vector[v]];
            beta[18] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 18 + c_Permutation_Vector[v]];
            beta[19] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 19 + c_Permutation_Vector[v]];
            beta[20] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 20 + c_Permutation_Vector[v]];
            beta[21] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 21 + c_Permutation_Vector[v]];
            beta[22] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 22 + c_Permutation_Vector[v]];
            
            break;
            
        case 24:
            
            beta[0] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 0 + c_Permutation_Vector[v]];
            beta[1] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 1 + c_Permutation_Vector[v]];
            beta[2] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 2 + c_Permutation_Vector[v]];
            beta[3] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 3 + c_Permutation_Vector[v]];
            beta[4] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 4 + c_Permutation_Vector[v]];
            beta[5] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 5 + c_Permutation_Vector[v]];
            beta[6] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 6 + c_Permutation_Vector[v]];
            beta[7] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 7 + c_Permutation_Vector[v]];
            beta[8] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 8 + c_Permutation_Vector[v]];
            beta[9] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 9 + c_Permutation_Vector[v]];
            beta[10] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 10 + c_Permutation_Vector[v]];
            beta[11] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 11 + c_Permutation_Vector[v]];
            beta[12] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 12 + c_Permutation_Vector[v]];
            beta[13] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 13 + c_Permutation_Vector[v]];
            beta[14] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 14 + c_Permutation_Vector[v]];
            beta[15] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 15 + c_Permutation_Vector[v]];
            beta[16] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 16 + c_Permutation_Vector[v]];
            beta[17] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 17 + c_Permutation_Vector[v]];
            beta[18] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 18 + c_Permutation_Vector[v]];
            beta[19] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 19 + c_Permutation_Vector[v]];
            beta[20] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 20 + c_Permutation_Vector[v]];
            beta[21] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 21 + c_Permutation_Vector[v]];
            beta[22] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 22 + c_Permutation_Vector[v]];
            beta[23] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 23 + c_Permutation_Vector[v]];
            
            break;
            
        case 25:
            
            beta[0] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 0 + c_Permutation_Vector[v]];
            beta[1] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 1 + c_Permutation_Vector[v]];
            beta[2] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 2 + c_Permutation_Vector[v]];
            beta[3] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 3 + c_Permutation_Vector[v]];
            beta[4] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 4 + c_Permutation_Vector[v]];
            beta[5] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 5 + c_Permutation_Vector[v]];
            beta[6] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 6 + c_Permutation_Vector[v]];
            beta[7] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 7 + c_Permutation_Vector[v]];
            beta[8] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 8 + c_Permutation_Vector[v]];
            beta[9] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 9 + c_Permutation_Vector[v]];
            beta[10] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 10 + c_Permutation_Vector[v]];
            beta[11] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 11 + c_Permutation_Vector[v]];
            beta[12] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 12 + c_Permutation_Vector[v]];
            beta[13] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 13 + c_Permutation_Vector[v]];
            beta[14] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 14 + c_Permutation_Vector[v]];
            beta[15] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 15 + c_Permutation_Vector[v]];
            beta[16] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 16 + c_Permutation_Vector[v]];
            beta[17] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 17 + c_Permutation_Vector[v]];
            beta[18] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 18 + c_Permutation_Vector[v]];
            beta[19] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 19 + c_Permutation_Vector[v]];
            beta[20] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 20 + c_Permutation_Vector[v]];
            beta[21] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 21 + c_Permutation_Vector[v]];
            beta[22] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 22 + c_Permutation_Vector[v]];
            beta[23] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 23 + c_Permutation_Vector[v]];
            beta[24] += value * c_xtxxt_GLM[NUMBER_OF_VOLUMES * 24 + c_Permutation_Vector[v]];
            
            break;
            
            
        default:
            1;
            break;
    }
    
    return 0;
}





// For second level, permutation of rows in design matrix (as in FSL)
float CalculateEpsSecondLevel(__private float eps,
                              __private float* beta,
                              __constant float* c_X_GLM,
                              int v,
                              __constant unsigned short int* c_Permutation_Vector,
                              int NUMBER_OF_VOLUMES,
                              int NUMBER_OF_REGRESSORS)
{
    switch(NUMBER_OF_REGRESSORS)
    {
        case 1:
            
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 0 + c_Permutation_Vector[v]] * beta[0];
            
            break;
            
        case 2:
            
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 0 + c_Permutation_Vector[v]] * beta[0];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 1 + c_Permutation_Vector[v]] * beta[1];
            
            break;
            
        case 3:
            
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 0 + c_Permutation_Vector[v]] * beta[0];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 1 + c_Permutation_Vector[v]] * beta[1];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 2 + c_Permutation_Vector[v]] * beta[2];
            
            break;
            
        case 4:
            
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 0 + c_Permutation_Vector[v]] * beta[0];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 1 + c_Permutation_Vector[v]] * beta[1];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 2 + c_Permutation_Vector[v]] * beta[2];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 3 + c_Permutation_Vector[v]] * beta[3];
            
            break;
            
        case 5:
            
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 0 + c_Permutation_Vector[v]] * beta[0];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 1 + c_Permutation_Vector[v]] * beta[1];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 2 + c_Permutation_Vector[v]] * beta[2];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 3 + c_Permutation_Vector[v]] * beta[3];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 4 + c_Permutation_Vector[v]] * beta[4];
            
            break;
            
        case 6:
            
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 0 + c_Permutation_Vector[v]] * beta[0];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 1 + c_Permutation_Vector[v]] * beta[1];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 2 + c_Permutation_Vector[v]] * beta[2];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 3 + c_Permutation_Vector[v]] * beta[3];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 4 + c_Permutation_Vector[v]] * beta[4];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 5 + c_Permutation_Vector[v]] * beta[5];
            
            break;
            
        case 7:
            
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 0 + c_Permutation_Vector[v]] * beta[0];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 1 + c_Permutation_Vector[v]] * beta[1];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 2 + c_Permutation_Vector[v]] * beta[2];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 3 + c_Permutation_Vector[v]] * beta[3];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 4 + c_Permutation_Vector[v]] * beta[4];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 5 + c_Permutation_Vector[v]] * beta[5];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 6 + c_Permutation_Vector[v]] * beta[6];
            
            break;
            
        case 8:
            
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 0 + c_Permutation_Vector[v]] * beta[0];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 1 + c_Permutation_Vector[v]] * beta[1];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 2 + c_Permutation_Vector[v]] * beta[2];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 3 + c_Permutation_Vector[v]] * beta[3];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 4 + c_Permutation_Vector[v]] * beta[4];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 5 + c_Permutation_Vector[v]] * beta[5];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 6 + c_Permutation_Vector[v]] * beta[6];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 7 + c_Permutation_Vector[v]] * beta[7];
            
            break;
            
        case 9:
            
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 0 + c_Permutation_Vector[v]] * beta[0];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 1 + c_Permutation_Vector[v]] * beta[1];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 2 + c_Permutation_Vector[v]] * beta[2];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 3 + c_Permutation_Vector[v]] * beta[3];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 4 + c_Permutation_Vector[v]] * beta[4];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 5 + c_Permutation_Vector[v]] * beta[5];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 6 + c_Permutation_Vector[v]] * beta[6];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 7 + c_Permutation_Vector[v]] * beta[7];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 8 + c_Permutation_Vector[v]] * beta[8];
            
            break;
            
        case 10:
            
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 0 + c_Permutation_Vector[v]] * beta[0];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 1 + c_Permutation_Vector[v]] * beta[1];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 2 + c_Permutation_Vector[v]] * beta[2];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 3 + c_Permutation_Vector[v]] * beta[3];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 4 + c_Permutation_Vector[v]] * beta[4];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 5 + c_Permutation_Vector[v]] * beta[5];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 6 + c_Permutation_Vector[v]] * beta[6];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 7 + c_Permutation_Vector[v]] * beta[7];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 8 + c_Permutation_Vector[v]] * beta[8];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 9 + c_Permutation_Vector[v]] * beta[9];
            
            break;
            
        case 11:
            
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 0 + c_Permutation_Vector[v]] * beta[0];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 1 + c_Permutation_Vector[v]] * beta[1];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 2 + c_Permutation_Vector[v]] * beta[2];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 3 + c_Permutation_Vector[v]] * beta[3];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 4 + c_Permutation_Vector[v]] * beta[4];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 5 + c_Permutation_Vector[v]] * beta[5];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 6 + c_Permutation_Vector[v]] * beta[6];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 7 + c_Permutation_Vector[v]] * beta[7];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 8 + c_Permutation_Vector[v]] * beta[8];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 9 + c_Permutation_Vector[v]] * beta[9];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 10 + c_Permutation_Vector[v]] * beta[10];
            
            break;
            
        case 12:
            
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 0 + c_Permutation_Vector[v]] * beta[0];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 1 + c_Permutation_Vector[v]] * beta[1];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 2 + c_Permutation_Vector[v]] * beta[2];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 3 + c_Permutation_Vector[v]] * beta[3];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 4 + c_Permutation_Vector[v]] * beta[4];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 5 + c_Permutation_Vector[v]] * beta[5];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 6 + c_Permutation_Vector[v]] * beta[6];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 7 + c_Permutation_Vector[v]] * beta[7];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 8 + c_Permutation_Vector[v]] * beta[8];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 9 + c_Permutation_Vector[v]] * beta[9];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 10 + c_Permutation_Vector[v]] * beta[10];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 11 + c_Permutation_Vector[v]] * beta[11];
            
            break;
            
        case 13:
            
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 0 + c_Permutation_Vector[v]] * beta[0];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 1 + c_Permutation_Vector[v]] * beta[1];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 2 + c_Permutation_Vector[v]] * beta[2];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 3 + c_Permutation_Vector[v]] * beta[3];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 4 + c_Permutation_Vector[v]] * beta[4];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 5 + c_Permutation_Vector[v]] * beta[5];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 6 + c_Permutation_Vector[v]] * beta[6];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 7 + c_Permutation_Vector[v]] * beta[7];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 8 + c_Permutation_Vector[v]] * beta[8];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 9 + c_Permutation_Vector[v]] * beta[9];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 10 + c_Permutation_Vector[v]] * beta[10];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 11 + c_Permutation_Vector[v]] * beta[11];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 12 + c_Permutation_Vector[v]] * beta[12];
            
            break;
            
        case 14:
            
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 0 + c_Permutation_Vector[v]] * beta[0];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 1 + c_Permutation_Vector[v]] * beta[1];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 2 + c_Permutation_Vector[v]] * beta[2];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 3 + c_Permutation_Vector[v]] * beta[3];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 4 + c_Permutation_Vector[v]] * beta[4];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 5 + c_Permutation_Vector[v]] * beta[5];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 6 + c_Permutation_Vector[v]] * beta[6];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 7 + c_Permutation_Vector[v]] * beta[7];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 8 + c_Permutation_Vector[v]] * beta[8];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 9 + c_Permutation_Vector[v]] * beta[9];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 10 + c_Permutation_Vector[v]] * beta[10];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 11 + c_Permutation_Vector[v]] * beta[11];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 12 + c_Permutation_Vector[v]] * beta[12];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 13 + c_Permutation_Vector[v]] * beta[13];
            
            break;
            
        case 15:
            
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 0 + c_Permutation_Vector[v]] * beta[0];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 1 + c_Permutation_Vector[v]] * beta[1];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 2 + c_Permutation_Vector[v]] * beta[2];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 3 + c_Permutation_Vector[v]] * beta[3];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 4 + c_Permutation_Vector[v]] * beta[4];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 5 + c_Permutation_Vector[v]] * beta[5];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 6 + c_Permutation_Vector[v]] * beta[6];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 7 + c_Permutation_Vector[v]] * beta[7];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 8 + c_Permutation_Vector[v]] * beta[8];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 9 + c_Permutation_Vector[v]] * beta[9];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 10 + c_Permutation_Vector[v]] * beta[10];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 11 + c_Permutation_Vector[v]] * beta[11];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 12 + c_Permutation_Vector[v]] * beta[12];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 13 + c_Permutation_Vector[v]] * beta[13];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 14 + c_Permutation_Vector[v]] * beta[14];
            
            break;
            
        case 16:
            
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 0 + c_Permutation_Vector[v]] * beta[0];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 1 + c_Permutation_Vector[v]] * beta[1];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 2 + c_Permutation_Vector[v]] * beta[2];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 3 + c_Permutation_Vector[v]] * beta[3];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 4 + c_Permutation_Vector[v]] * beta[4];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 5 + c_Permutation_Vector[v]] * beta[5];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 6 + c_Permutation_Vector[v]] * beta[6];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 7 + c_Permutation_Vector[v]] * beta[7];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 8 + c_Permutation_Vector[v]] * beta[8];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 9 + c_Permutation_Vector[v]] * beta[9];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 10 + c_Permutation_Vector[v]] * beta[10];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 11 + c_Permutation_Vector[v]] * beta[11];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 12 + c_Permutation_Vector[v]] * beta[12];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 13 + c_Permutation_Vector[v]] * beta[13];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 14 + c_Permutation_Vector[v]] * beta[14];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 15 + c_Permutation_Vector[v]] * beta[15];
            
            break;
            
        case 17:
            
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 0 + c_Permutation_Vector[v]] * beta[0];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 1 + c_Permutation_Vector[v]] * beta[1];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 2 + c_Permutation_Vector[v]] * beta[2];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 3 + c_Permutation_Vector[v]] * beta[3];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 4 + c_Permutation_Vector[v]] * beta[4];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 5 + c_Permutation_Vector[v]] * beta[5];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 6 + c_Permutation_Vector[v]] * beta[6];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 7 + c_Permutation_Vector[v]] * beta[7];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 8 + c_Permutation_Vector[v]] * beta[8];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 9 + c_Permutation_Vector[v]] * beta[9];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 10 + c_Permutation_Vector[v]] * beta[10];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 11 + c_Permutation_Vector[v]] * beta[11];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 12 + c_Permutation_Vector[v]] * beta[12];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 13 + c_Permutation_Vector[v]] * beta[13];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 14 + c_Permutation_Vector[v]] * beta[14];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 15 + c_Permutation_Vector[v]] * beta[15];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 16 + c_Permutation_Vector[v]] * beta[16];
            
            break;
            
        case 18:
            
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 0 + c_Permutation_Vector[v]] * beta[0];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 1 + c_Permutation_Vector[v]] * beta[1];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 2 + c_Permutation_Vector[v]] * beta[2];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 3 + c_Permutation_Vector[v]] * beta[3];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 4 + c_Permutation_Vector[v]] * beta[4];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 5 + c_Permutation_Vector[v]] * beta[5];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 6 + c_Permutation_Vector[v]] * beta[6];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 7 + c_Permutation_Vector[v]] * beta[7];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 8 + c_Permutation_Vector[v]] * beta[8];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 9 + c_Permutation_Vector[v]] * beta[9];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 10 + c_Permutation_Vector[v]] * beta[10];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 11 + c_Permutation_Vector[v]] * beta[11];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 12 + c_Permutation_Vector[v]] * beta[12];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 13 + c_Permutation_Vector[v]] * beta[13];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 14 + c_Permutation_Vector[v]] * beta[14];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 15 + c_Permutation_Vector[v]] * beta[15];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 16 + c_Permutation_Vector[v]] * beta[16];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 17 + c_Permutation_Vector[v]] * beta[17];
            
            break;
            
        case 19:
            
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 0 + c_Permutation_Vector[v]] * beta[0];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 1 + c_Permutation_Vector[v]] * beta[1];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 2 + c_Permutation_Vector[v]] * beta[2];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 3 + c_Permutation_Vector[v]] * beta[3];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 4 + c_Permutation_Vector[v]] * beta[4];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 5 + c_Permutation_Vector[v]] * beta[5];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 6 + c_Permutation_Vector[v]] * beta[6];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 7 + c_Permutation_Vector[v]] * beta[7];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 8 + c_Permutation_Vector[v]] * beta[8];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 9 + c_Permutation_Vector[v]] * beta[9];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 10 + c_Permutation_Vector[v]] * beta[10];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 11 + c_Permutation_Vector[v]] * beta[11];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 12 + c_Permutation_Vector[v]] * beta[12];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 13 + c_Permutation_Vector[v]] * beta[13];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 14 + c_Permutation_Vector[v]] * beta[14];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 15 + c_Permutation_Vector[v]] * beta[15];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 16 + c_Permutation_Vector[v]] * beta[16];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 17 + c_Permutation_Vector[v]] * beta[17];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 18 + c_Permutation_Vector[v]] * beta[18];
            
            break;
            
        case 20:
            
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 0 + c_Permutation_Vector[v]] * beta[0];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 1 + c_Permutation_Vector[v]] * beta[1];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 2 + c_Permutation_Vector[v]] * beta[2];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 3 + c_Permutation_Vector[v]] * beta[3];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 4 + c_Permutation_Vector[v]] * beta[4];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 5 + c_Permutation_Vector[v]] * beta[5];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 6 + c_Permutation_Vector[v]] * beta[6];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 7 + c_Permutation_Vector[v]] * beta[7];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 8 + c_Permutation_Vector[v]] * beta[8];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 9 + c_Permutation_Vector[v]] * beta[9];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 10 + c_Permutation_Vector[v]] * beta[10];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 11 + c_Permutation_Vector[v]] * beta[11];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 12 + c_Permutation_Vector[v]] * beta[12];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 13 + c_Permutation_Vector[v]] * beta[13];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 14 + c_Permutation_Vector[v]] * beta[14];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 15 + c_Permutation_Vector[v]] * beta[15];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 16 + c_Permutation_Vector[v]] * beta[16];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 17 + c_Permutation_Vector[v]] * beta[17];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 18 + c_Permutation_Vector[v]] * beta[18];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 19 + c_Permutation_Vector[v]] * beta[19];
            
            break;
            
        case 21:
            
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 0 + c_Permutation_Vector[v]] * beta[0];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 1 + c_Permutation_Vector[v]] * beta[1];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 2 + c_Permutation_Vector[v]] * beta[2];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 3 + c_Permutation_Vector[v]] * beta[3];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 4 + c_Permutation_Vector[v]] * beta[4];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 5 + c_Permutation_Vector[v]] * beta[5];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 6 + c_Permutation_Vector[v]] * beta[6];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 7 + c_Permutation_Vector[v]] * beta[7];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 8 + c_Permutation_Vector[v]] * beta[8];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 9 + c_Permutation_Vector[v]] * beta[9];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 10 + c_Permutation_Vector[v]] * beta[10];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 11 + c_Permutation_Vector[v]] * beta[11];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 12 + c_Permutation_Vector[v]] * beta[12];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 13 + c_Permutation_Vector[v]] * beta[13];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 14 + c_Permutation_Vector[v]] * beta[14];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 15 + c_Permutation_Vector[v]] * beta[15];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 16 + c_Permutation_Vector[v]] * beta[16];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 17 + c_Permutation_Vector[v]] * beta[17];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 18 + c_Permutation_Vector[v]] * beta[18];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 19 + c_Permutation_Vector[v]] * beta[19];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 20 + c_Permutation_Vector[v]] * beta[20];
            
            break;
            
        case 22:
            
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 0 + c_Permutation_Vector[v]] * beta[0];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 1 + c_Permutation_Vector[v]] * beta[1];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 2 + c_Permutation_Vector[v]] * beta[2];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 3 + c_Permutation_Vector[v]] * beta[3];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 4 + c_Permutation_Vector[v]] * beta[4];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 5 + c_Permutation_Vector[v]] * beta[5];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 6 + c_Permutation_Vector[v]] * beta[6];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 7 + c_Permutation_Vector[v]] * beta[7];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 8 + c_Permutation_Vector[v]] * beta[8];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 9 + c_Permutation_Vector[v]] * beta[9];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 10 + c_Permutation_Vector[v]] * beta[10];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 11 + c_Permutation_Vector[v]] * beta[11];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 12 + c_Permutation_Vector[v]] * beta[12];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 13 + c_Permutation_Vector[v]] * beta[13];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 14 + c_Permutation_Vector[v]] * beta[14];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 15 + c_Permutation_Vector[v]] * beta[15];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 16 + c_Permutation_Vector[v]] * beta[16];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 17 + c_Permutation_Vector[v]] * beta[17];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 18 + c_Permutation_Vector[v]] * beta[18];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 19 + c_Permutation_Vector[v]] * beta[19];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 20 + c_Permutation_Vector[v]] * beta[20];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 21 + c_Permutation_Vector[v]] * beta[21];
            
            break;
            
        case 23:
            
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 0 + c_Permutation_Vector[v]] * beta[0];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 1 + c_Permutation_Vector[v]] * beta[1];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 2 + c_Permutation_Vector[v]] * beta[2];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 3 + c_Permutation_Vector[v]] * beta[3];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 4 + c_Permutation_Vector[v]] * beta[4];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 5 + c_Permutation_Vector[v]] * beta[5];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 6 + c_Permutation_Vector[v]] * beta[6];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 7 + c_Permutation_Vector[v]] * beta[7];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 8 + c_Permutation_Vector[v]] * beta[8];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 9 + c_Permutation_Vector[v]] * beta[9];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 10 + c_Permutation_Vector[v]] * beta[10];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 11 + c_Permutation_Vector[v]] * beta[11];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 12 + c_Permutation_Vector[v]] * beta[12];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 13 + c_Permutation_Vector[v]] * beta[13];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 14 + c_Permutation_Vector[v]] * beta[14];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 15 + c_Permutation_Vector[v]] * beta[15];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 16 + c_Permutation_Vector[v]] * beta[16];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 17 + c_Permutation_Vector[v]] * beta[17];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 18 + c_Permutation_Vector[v]] * beta[18];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 19 + c_Permutation_Vector[v]] * beta[19];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 20 + c_Permutation_Vector[v]] * beta[20];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 21 + c_Permutation_Vector[v]] * beta[21];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 22 + c_Permutation_Vector[v]] * beta[22];
            
            break;
            
        case 24:
            
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 0 + c_Permutation_Vector[v]] * beta[0];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 1 + c_Permutation_Vector[v]] * beta[1];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 2 + c_Permutation_Vector[v]] * beta[2];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 3 + c_Permutation_Vector[v]] * beta[3];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 4 + c_Permutation_Vector[v]] * beta[4];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 5 + c_Permutation_Vector[v]] * beta[5];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 6 + c_Permutation_Vector[v]] * beta[6];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 7 + c_Permutation_Vector[v]] * beta[7];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 8 + c_Permutation_Vector[v]] * beta[8];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 9 + c_Permutation_Vector[v]] * beta[9];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 10 + c_Permutation_Vector[v]] * beta[10];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 11 + c_Permutation_Vector[v]] * beta[11];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 12 + c_Permutation_Vector[v]] * beta[12];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 13 + c_Permutation_Vector[v]] * beta[13];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 14 + c_Permutation_Vector[v]] * beta[14];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 15 + c_Permutation_Vector[v]] * beta[15];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 16 + c_Permutation_Vector[v]] * beta[16];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 17 + c_Permutation_Vector[v]] * beta[17];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 18 + c_Permutation_Vector[v]] * beta[18];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 19 + c_Permutation_Vector[v]] * beta[19];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 20 + c_Permutation_Vector[v]] * beta[20];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 21 + c_Permutation_Vector[v]] * beta[21];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 22 + c_Permutation_Vector[v]] * beta[22];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 23 + c_Permutation_Vector[v]] * beta[23];
            
            break;
            
        case 25:
            
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 0 + c_Permutation_Vector[v]] * beta[0];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 1 + c_Permutation_Vector[v]] * beta[1];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 2 + c_Permutation_Vector[v]] * beta[2];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 3 + c_Permutation_Vector[v]] * beta[3];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 4 + c_Permutation_Vector[v]] * beta[4];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 5 + c_Permutation_Vector[v]] * beta[5];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 6 + c_Permutation_Vector[v]] * beta[6];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 7 + c_Permutation_Vector[v]] * beta[7];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 8 + c_Permutation_Vector[v]] * beta[8];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 9 + c_Permutation_Vector[v]] * beta[9];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 10 + c_Permutation_Vector[v]] * beta[10];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 11 + c_Permutation_Vector[v]] * beta[11];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 12 + c_Permutation_Vector[v]] * beta[12];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 13 + c_Permutation_Vector[v]] * beta[13];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 14 + c_Permutation_Vector[v]] * beta[14];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 15 + c_Permutation_Vector[v]] * beta[15];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 16 + c_Permutation_Vector[v]] * beta[16];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 17 + c_Permutation_Vector[v]] * beta[17];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 18 + c_Permutation_Vector[v]] * beta[18];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 19 + c_Permutation_Vector[v]] * beta[19];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 20 + c_Permutation_Vector[v]] * beta[20];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 21 + c_Permutation_Vector[v]] * beta[21];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 22 + c_Permutation_Vector[v]] * beta[22];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 23 + c_Permutation_Vector[v]] * beta[23];
            eps -= c_X_GLM[NUMBER_OF_VOLUMES * 24 + c_Permutation_Vector[v]] * beta[24];
            
            break;
            
        default:
            1;
            break;
    }
    
    return eps;
}






int CalculateCBeta(__private float* cbeta, __private float* beta, __constant float* c_Contrasts, int c, int NUMBER_OF_REGRESSORS)
{
    cbeta[c] = 0.0f;
    
    switch(NUMBER_OF_REGRESSORS)
    {
        case 1:
            
            cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 0] * beta[0];
            
            break;
            
        case 2:
            
            cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 0] * beta[0];
            cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 1] * beta[1];
            
            break;
            
        case 3:
            
            cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 0] * beta[0];
            cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 1] * beta[1];
            cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 2] * beta[2];
            
            break;
            
        case 4:
            
            cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 0] * beta[0];
            cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 1] * beta[1];
            cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 2] * beta[2];
            cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 3] * beta[3];
            
            break;
            
        case 5:
            
            cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 0] * beta[0];
            cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 1] * beta[1];
            cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 2] * beta[2];
            cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 3] * beta[3];
            cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 4] * beta[4];
            
            break;
            
        case 6:
            
            cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 0] * beta[0];
            cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 1] * beta[1];
            cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 2] * beta[2];
            cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 3] * beta[3];
            cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 4] * beta[4];
            cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 5] * beta[5];
            
            break;
            
        case 7:
            
            cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 0] * beta[0];
            cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 1] * beta[1];
            cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 2] * beta[2];
            cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 3] * beta[3];
            cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 4] * beta[4];
            cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 5] * beta[5];
            cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 6] * beta[6];
            
            break;
            
        case 8:
            
            cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 0] * beta[0];
            cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 1] * beta[1];
            cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 2] * beta[2];
            cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 3] * beta[3];
            cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 4] * beta[4];
            cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 5] * beta[5];
            cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 6] * beta[6];
            cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 7] * beta[7];
            
            break;
            
        case 9:
            
            cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 0] * beta[0];
            cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 1] * beta[1];
            cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 2] * beta[2];
            cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 3] * beta[3];
            cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 4] * beta[4];
            cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 5] * beta[5];
            cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 6] * beta[6];
            cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 7] * beta[7];
            cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 8] * beta[8];
            
            break;
            
        case 10:
            
            cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 0] * beta[0];
            cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 1] * beta[1];
            cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 2] * beta[2];
            cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 3] * beta[3];
            cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 4] * beta[4];
            cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 5] * beta[5];
            cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 6] * beta[6];
            cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 7] * beta[7];
            cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 8] * beta[8];
            cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 9] * beta[9];
            
            break;
            
        case 11:
            
            cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 0] * beta[0];
            cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 1] * beta[1];
            cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 2] * beta[2];
            cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 3] * beta[3];
            cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 4] * beta[4];
            cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 5] * beta[5];
            cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 6] * beta[6];
            cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 7] * beta[7];
            cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 8] * beta[8];
            cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 9] * beta[9];
            cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 10] * beta[10];
            
            break;
            
        case 12:
            
            cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 0] * beta[0];
            cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 1] * beta[1];
            cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 2] * beta[2];
            cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 3] * beta[3];
            cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 4] * beta[4];
            cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 5] * beta[5];
            cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 6] * beta[6];
            cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 7] * beta[7];
            cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 8] * beta[8];
            cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 9] * beta[9];
            cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 10] * beta[10];
            cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 11] * beta[11];
            
            break;
            
        case 13:
            
            cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 0] * beta[0];
            cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 1] * beta[1];
            cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 2] * beta[2];
            cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 3] * beta[3];
            cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 4] * beta[4];
            cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 5] * beta[5];
            cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 6] * beta[6];
            cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 7] * beta[7];
            cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 8] * beta[8];
            cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 9] * beta[9];
            cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 10] * beta[10];
            cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 11] * beta[11];
            cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 12] * beta[12];
            
            break;
            
        case 14:
            
            cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 0] * beta[0];
            cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 1] * beta[1];
            cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 2] * beta[2];
            cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 3] * beta[3];
            cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 4] * beta[4];
            cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 5] * beta[5];
            cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 6] * beta[6];
            cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 7] * beta[7];
            cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 8] * beta[8];
            cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 9] * beta[9];
            cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 10] * beta[10];
            cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 11] * beta[11];
            cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 12] * beta[12];
            cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 13] * beta[13];
            
            break;
            
        case 15:
            
            cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 0] * beta[0];
            cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 1] * beta[1];
            cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 2] * beta[2];
            cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 3] * beta[3];
            cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 4] * beta[4];
            cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 5] * beta[5];
            cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 6] * beta[6];
            cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 7] * beta[7];
            cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 8] * beta[8];
            cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 9] * beta[9];
            cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 10] * beta[10];
            cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 11] * beta[11];
            cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 12] * beta[12];
            cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 13] * beta[13];
            cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 14] * beta[14];
            
            break;
            
        case 16:
            
            cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 0] * beta[0];
            cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 1] * beta[1];
            cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 2] * beta[2];
            cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 3] * beta[3];
            cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 4] * beta[4];
            cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 5] * beta[5];
            cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 6] * beta[6];
            cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 7] * beta[7];
            cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 8] * beta[8];
            cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 9] * beta[9];
            cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 10] * beta[10];
            cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 11] * beta[11];
            cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 12] * beta[12];
            cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 13] * beta[13];
            cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 14] * beta[14];
            cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 15] * beta[15];
            
            break;
            
        case 17:
            
            cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 0] * beta[0];
            cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 1] * beta[1];
            cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 2] * beta[2];
            cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 3] * beta[3];
            cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 4] * beta[4];
            cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 5] * beta[5];
            cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 6] * beta[6];
            cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 7] * beta[7];
            cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 8] * beta[8];
            cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 9] * beta[9];
            cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 10] * beta[10];
            cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 11] * beta[11];
            cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 12] * beta[12];
            cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 13] * beta[13];
            cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 14] * beta[14];
            cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 15] * beta[15];
            cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 16] * beta[16];
            
            break;
            
        case 18:
            
            cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 0] * beta[0];
            cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 1] * beta[1];
            cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 2] * beta[2];
            cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 3] * beta[3];
            cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 4] * beta[4];
            cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 5] * beta[5];
            cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 6] * beta[6];
            cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 7] * beta[7];
            cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 8] * beta[8];
            cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 9] * beta[9];
            cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 10] * beta[10];
            cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 11] * beta[11];
            cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 12] * beta[12];
            cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 13] * beta[13];
            cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 14] * beta[14];
            cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 15] * beta[15];
            cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 16] * beta[16];
            cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 17] * beta[17];
            
            break;
            
        case 19:
            
            cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 0] * beta[0];
            cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 1] * beta[1];
            cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 2] * beta[2];
            cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 3] * beta[3];
            cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 4] * beta[4];
            cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 5] * beta[5];
            cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 6] * beta[6];
            cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 7] * beta[7];
            cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 8] * beta[8];
            cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 9] * beta[9];
            cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 10] * beta[10];
            cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 11] * beta[11];
            cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 12] * beta[12];
            cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 13] * beta[13];
            cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 14] * beta[14];
            cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 15] * beta[15];
            cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 16] * beta[16];
            cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 17] * beta[17];
            cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 18] * beta[18];
            
            break;
            
        case 20:
            
            cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 0] * beta[0];
            cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 1] * beta[1];
            cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 2] * beta[2];
            cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 3] * beta[3];
            cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 4] * beta[4];
            cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 5] * beta[5];
            cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 6] * beta[6];
            cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 7] * beta[7];
            cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 8] * beta[8];
            cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 9] * beta[9];
            cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 10] * beta[10];
            cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 11] * beta[11];
            cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 12] * beta[12];
            cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 13] * beta[13];
            cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 14] * beta[14];
            cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 15] * beta[15];
            cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 16] * beta[16];
            cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 17] * beta[17];
            cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 18] * beta[18];
            cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 19] * beta[19];
            
            break;
            
        case 21:
            
            cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 0] * beta[0];
            cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 1] * beta[1];
            cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 2] * beta[2];
            cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 3] * beta[3];
            cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 4] * beta[4];
            cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 5] * beta[5];
            cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 6] * beta[6];
            cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 7] * beta[7];
            cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 8] * beta[8];
            cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 9] * beta[9];
            cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 10] * beta[10];
            cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 11] * beta[11];
            cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 12] * beta[12];
            cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 13] * beta[13];
            cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 14] * beta[14];
            cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 15] * beta[15];
            cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 16] * beta[16];
            cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 17] * beta[17];
            cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 18] * beta[18];
            cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 19] * beta[19];
            cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 20] * beta[20];
            
            break;
            
        case 22:
            
            cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 0] * beta[0];
            cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 1] * beta[1];
            cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 2] * beta[2];
            cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 3] * beta[3];
            cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 4] * beta[4];
            cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 5] * beta[5];
            cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 6] * beta[6];
            cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 7] * beta[7];
            cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 8] * beta[8];
            cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 9] * beta[9];
            cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 10] * beta[10];
            cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 11] * beta[11];
            cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 12] * beta[12];
            cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 13] * beta[13];
            cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 14] * beta[14];
            cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 15] * beta[15];
            cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 16] * beta[16];
            cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 17] * beta[17];
            cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 18] * beta[18];
            cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 19] * beta[19];
            cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 20] * beta[20];
            cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 21] * beta[21];
            
            break;
            
        case 23:
            
            cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 0] * beta[0];
            cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 1] * beta[1];
            cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 2] * beta[2];
            cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 3] * beta[3];
            cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 4] * beta[4];
            cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 5] * beta[5];
            cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 6] * beta[6];
            cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 7] * beta[7];
            cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 8] * beta[8];
            cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 9] * beta[9];
            cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 10] * beta[10];
            cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 11] * beta[11];
            cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 12] * beta[12];
            cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 13] * beta[13];
            cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 14] * beta[14];
            cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 15] * beta[15];
            cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 16] * beta[16];
            cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 17] * beta[17];
            cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 18] * beta[18];
            cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 19] * beta[19];
            cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 20] * beta[20];
            cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 21] * beta[21];
            cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 22] * beta[22];
            
            break;
            
        case 24:
            
            cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 0] * beta[0];
            cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 1] * beta[1];
            cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 2] * beta[2];
            cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 3] * beta[3];
            cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 4] * beta[4];
            cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 5] * beta[5];
            cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 6] * beta[6];
            cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 7] * beta[7];
            cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 8] * beta[8];
            cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 9] * beta[9];
            cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 10] * beta[10];
            cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 11] * beta[11];
            cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 12] * beta[12];
            cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 13] * beta[13];
            cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 14] * beta[14];
            cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 15] * beta[15];
            cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 16] * beta[16];
            cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 17] * beta[17];
            cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 18] * beta[18];
            cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 19] * beta[19];
            cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 20] * beta[20];
            cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 21] * beta[21];
            cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 22] * beta[22];
            cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 23] * beta[23];
            
            break;
            
        case 25:
            
            cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 0] * beta[0];
            cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 1] * beta[1];
            cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 2] * beta[2];
            cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 3] * beta[3];
            cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 4] * beta[4];
            cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 5] * beta[5];
            cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 6] * beta[6];
            cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 7] * beta[7];
            cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 8] * beta[8];
            cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 9] * beta[9];
            cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 10] * beta[10];
            cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 11] * beta[11];
            cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 12] * beta[12];
            cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 13] * beta[13];
            cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 14] * beta[14];
            cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 15] * beta[15];
            cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 16] * beta[16];
            cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 17] * beta[17];
            cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 18] * beta[18];
            cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 19] * beta[19];
            cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 20] * beta[20];
            cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 21] * beta[21];
            cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 22] * beta[22];
            cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 23] * beta[23];
            cbeta[c] += c_Contrasts[NUMBER_OF_REGRESSORS * c + 24] * beta[24];			
            
            break;
            
        default:
            1;
            break;
    }
    
    return 0;
}


int CalculateCBetas(__private float* cbeta, __private float* beta, __constant float* c_Contrasts, int NUMBER_OF_REGRESSORS, int NUMBER_OF_CONTRASTS)
{
    switch(NUMBER_OF_CONTRASTS)
    {
        case 1:
            
            CalculateCBeta(cbeta, beta, c_Contrasts, 0, NUMBER_OF_REGRESSORS);
            
            break;
            
        case 2:
            
            CalculateCBeta(cbeta, beta, c_Contrasts, 0, NUMBER_OF_REGRESSORS);
            CalculateCBeta(cbeta, beta, c_Contrasts, 1, NUMBER_OF_REGRESSORS);
            
            break;
            
        case 3:
            
            CalculateCBeta(cbeta, beta, c_Contrasts, 0, NUMBER_OF_REGRESSORS);
            CalculateCBeta(cbeta, beta, c_Contrasts, 1, NUMBER_OF_REGRESSORS);
            CalculateCBeta(cbeta, beta, c_Contrasts, 2, NUMBER_OF_REGRESSORS);
            
            break;
            
        case 4:
            
            CalculateCBeta(cbeta, beta, c_Contrasts, 0, NUMBER_OF_REGRESSORS);
            CalculateCBeta(cbeta, beta, c_Contrasts, 1, NUMBER_OF_REGRESSORS);
            CalculateCBeta(cbeta, beta, c_Contrasts, 2, NUMBER_OF_REGRESSORS);
            CalculateCBeta(cbeta, beta, c_Contrasts, 3, NUMBER_OF_REGRESSORS);
            
            break;
            
        case 5:
            
            CalculateCBeta(cbeta, beta, c_Contrasts, 0, NUMBER_OF_REGRESSORS);
            CalculateCBeta(cbeta, beta, c_Contrasts, 1, NUMBER_OF_REGRESSORS);
            CalculateCBeta(cbeta, beta, c_Contrasts, 2, NUMBER_OF_REGRESSORS);
            CalculateCBeta(cbeta, beta, c_Contrasts, 3, NUMBER_OF_REGRESSORS);
            CalculateCBeta(cbeta, beta, c_Contrasts, 4, NUMBER_OF_REGRESSORS);
            
            break;
            
        case 6:
            
            CalculateCBeta(cbeta, beta, c_Contrasts, 0, NUMBER_OF_REGRESSORS);
            CalculateCBeta(cbeta, beta, c_Contrasts, 1, NUMBER_OF_REGRESSORS);
            CalculateCBeta(cbeta, beta, c_Contrasts, 2, NUMBER_OF_REGRESSORS);
            CalculateCBeta(cbeta, beta, c_Contrasts, 3, NUMBER_OF_REGRESSORS);
            CalculateCBeta(cbeta, beta, c_Contrasts, 4, NUMBER_OF_REGRESSORS);
            CalculateCBeta(cbeta, beta, c_Contrasts, 5, NUMBER_OF_REGRESSORS);
            
            break;
            
        case 7:
            
            CalculateCBeta(cbeta, beta, c_Contrasts, 0, NUMBER_OF_REGRESSORS);
            CalculateCBeta(cbeta, beta, c_Contrasts, 1, NUMBER_OF_REGRESSORS);
            CalculateCBeta(cbeta, beta, c_Contrasts, 2, NUMBER_OF_REGRESSORS);
            CalculateCBeta(cbeta, beta, c_Contrasts, 3, NUMBER_OF_REGRESSORS);
            CalculateCBeta(cbeta, beta, c_Contrasts, 4, NUMBER_OF_REGRESSORS);
            CalculateCBeta(cbeta, beta, c_Contrasts, 5, NUMBER_OF_REGRESSORS);
            CalculateCBeta(cbeta, beta, c_Contrasts, 6, NUMBER_OF_REGRESSORS);
            
            break;
            
        case 8:
            
            CalculateCBeta(cbeta, beta, c_Contrasts, 0, NUMBER_OF_REGRESSORS);
            CalculateCBeta(cbeta, beta, c_Contrasts, 1, NUMBER_OF_REGRESSORS);
            CalculateCBeta(cbeta, beta, c_Contrasts, 2, NUMBER_OF_REGRESSORS);
            CalculateCBeta(cbeta, beta, c_Contrasts, 3, NUMBER_OF_REGRESSORS);
            CalculateCBeta(cbeta, beta, c_Contrasts, 4, NUMBER_OF_REGRESSORS);
            CalculateCBeta(cbeta, beta, c_Contrasts, 5, NUMBER_OF_REGRESSORS);
            CalculateCBeta(cbeta, beta, c_Contrasts, 6, NUMBER_OF_REGRESSORS);
            CalculateCBeta(cbeta, beta, c_Contrasts, 7, NUMBER_OF_REGRESSORS);
            
            break;
            
        case 9:
            
            CalculateCBeta(cbeta, beta, c_Contrasts, 0, NUMBER_OF_REGRESSORS);
            CalculateCBeta(cbeta, beta, c_Contrasts, 1, NUMBER_OF_REGRESSORS);
            CalculateCBeta(cbeta, beta, c_Contrasts, 2, NUMBER_OF_REGRESSORS);
            CalculateCBeta(cbeta, beta, c_Contrasts, 3, NUMBER_OF_REGRESSORS);
            CalculateCBeta(cbeta, beta, c_Contrasts, 4, NUMBER_OF_REGRESSORS);
            CalculateCBeta(cbeta, beta, c_Contrasts, 5, NUMBER_OF_REGRESSORS);
            CalculateCBeta(cbeta, beta, c_Contrasts, 6, NUMBER_OF_REGRESSORS);
            CalculateCBeta(cbeta, beta, c_Contrasts, 7, NUMBER_OF_REGRESSORS);
            CalculateCBeta(cbeta, beta, c_Contrasts, 8, NUMBER_OF_REGRESSORS);
            
            break;
            
        case 10:
            
            CalculateCBeta(cbeta, beta, c_Contrasts, 0, NUMBER_OF_REGRESSORS);
            CalculateCBeta(cbeta, beta, c_Contrasts, 1, NUMBER_OF_REGRESSORS);
            CalculateCBeta(cbeta, beta, c_Contrasts, 2, NUMBER_OF_REGRESSORS);
            CalculateCBeta(cbeta, beta, c_Contrasts, 3, NUMBER_OF_REGRESSORS);
            CalculateCBeta(cbeta, beta, c_Contrasts, 4, NUMBER_OF_REGRESSORS);
            CalculateCBeta(cbeta, beta, c_Contrasts, 5, NUMBER_OF_REGRESSORS);
            CalculateCBeta(cbeta, beta, c_Contrasts, 6, NUMBER_OF_REGRESSORS);
            CalculateCBeta(cbeta, beta, c_Contrasts, 7, NUMBER_OF_REGRESSORS);
            CalculateCBeta(cbeta, beta, c_Contrasts, 8, NUMBER_OF_REGRESSORS);
            CalculateCBeta(cbeta, beta, c_Contrasts, 9, NUMBER_OF_REGRESSORS);
            
            break;		
            
        default:
            1;
            break;
    }	
    
    return 0;
}




int CalculateCTXTXCCBeta(__private float* beta, float vareps, __constant float* c_ctxtxc_GLM, __private float* cbeta, int c,  int NUMBER_OF_CONTRASTS)
{
    beta[c] = 0.0f;
    
    switch(NUMBER_OF_CONTRASTS)
    {
        case 1:
            
            beta[c] += 1.0f/vareps * c_ctxtxc_GLM[0 + c * NUMBER_OF_CONTRASTS] * cbeta[0];
            
            break;
            
        case 2:
            
            beta[c] += 1.0f/vareps * c_ctxtxc_GLM[0 + c * NUMBER_OF_CONTRASTS] * cbeta[0];
            beta[c] += 1.0f/vareps * c_ctxtxc_GLM[1 + c * NUMBER_OF_CONTRASTS] * cbeta[1];
            
            break;
            
        case 3:
            
            beta[c] += 1.0f/vareps * c_ctxtxc_GLM[0 + c * NUMBER_OF_CONTRASTS] * cbeta[0];
            beta[c] += 1.0f/vareps * c_ctxtxc_GLM[1 + c * NUMBER_OF_CONTRASTS] * cbeta[1];
            beta[c] += 1.0f/vareps * c_ctxtxc_GLM[2 + c * NUMBER_OF_CONTRASTS] * cbeta[2];
            
            break;
            
        case 4:
            
            beta[c] += 1.0f/vareps * c_ctxtxc_GLM[0 + c * NUMBER_OF_CONTRASTS] * cbeta[0];
            beta[c] += 1.0f/vareps * c_ctxtxc_GLM[1 + c * NUMBER_OF_CONTRASTS] * cbeta[1];
            beta[c] += 1.0f/vareps * c_ctxtxc_GLM[2 + c * NUMBER_OF_CONTRASTS] * cbeta[2];
            beta[c] += 1.0f/vareps * c_ctxtxc_GLM[3 + c * NUMBER_OF_CONTRASTS] * cbeta[3];
            
            break;
            
        case 5:
            
            beta[c] += 1.0f/vareps * c_ctxtxc_GLM[0 + c * NUMBER_OF_CONTRASTS] * cbeta[0];
            beta[c] += 1.0f/vareps * c_ctxtxc_GLM[1 + c * NUMBER_OF_CONTRASTS] * cbeta[1];
            beta[c] += 1.0f/vareps * c_ctxtxc_GLM[2 + c * NUMBER_OF_CONTRASTS] * cbeta[2];
            beta[c] += 1.0f/vareps * c_ctxtxc_GLM[3 + c * NUMBER_OF_CONTRASTS] * cbeta[3];
            beta[c] += 1.0f/vareps * c_ctxtxc_GLM[4 + c * NUMBER_OF_CONTRASTS] * cbeta[4];
            
            break;
            
        case 6:
            
            beta[c] += 1.0f/vareps * c_ctxtxc_GLM[0 + c * NUMBER_OF_CONTRASTS] * cbeta[0];
            beta[c] += 1.0f/vareps * c_ctxtxc_GLM[1 + c * NUMBER_OF_CONTRASTS] * cbeta[1];
            beta[c] += 1.0f/vareps * c_ctxtxc_GLM[2 + c * NUMBER_OF_CONTRASTS] * cbeta[2];
            beta[c] += 1.0f/vareps * c_ctxtxc_GLM[3 + c * NUMBER_OF_CONTRASTS] * cbeta[3];
            beta[c] += 1.0f/vareps * c_ctxtxc_GLM[4 + c * NUMBER_OF_CONTRASTS] * cbeta[4];
            beta[c] += 1.0f/vareps * c_ctxtxc_GLM[5 + c * NUMBER_OF_CONTRASTS] * cbeta[5];
            
            break;
            
        case 7:
            
            beta[c] += 1.0f/vareps * c_ctxtxc_GLM[0 + c * NUMBER_OF_CONTRASTS] * cbeta[0];
            beta[c] += 1.0f/vareps * c_ctxtxc_GLM[1 + c * NUMBER_OF_CONTRASTS] * cbeta[1];
            beta[c] += 1.0f/vareps * c_ctxtxc_GLM[2 + c * NUMBER_OF_CONTRASTS] * cbeta[2];
            beta[c] += 1.0f/vareps * c_ctxtxc_GLM[3 + c * NUMBER_OF_CONTRASTS] * cbeta[3];
            beta[c] += 1.0f/vareps * c_ctxtxc_GLM[4 + c * NUMBER_OF_CONTRASTS] * cbeta[4];
            beta[c] += 1.0f/vareps * c_ctxtxc_GLM[5 + c * NUMBER_OF_CONTRASTS] * cbeta[5];
            beta[c] += 1.0f/vareps * c_ctxtxc_GLM[6 + c * NUMBER_OF_CONTRASTS] * cbeta[6];
            
            break;
            
        case 8:
            
            beta[c] += 1.0f/vareps * c_ctxtxc_GLM[0 + c * NUMBER_OF_CONTRASTS] * cbeta[0];
            beta[c] += 1.0f/vareps * c_ctxtxc_GLM[1 + c * NUMBER_OF_CONTRASTS] * cbeta[1];
            beta[c] += 1.0f/vareps * c_ctxtxc_GLM[2 + c * NUMBER_OF_CONTRASTS] * cbeta[2];
            beta[c] += 1.0f/vareps * c_ctxtxc_GLM[3 + c * NUMBER_OF_CONTRASTS] * cbeta[3];
            beta[c] += 1.0f/vareps * c_ctxtxc_GLM[4 + c * NUMBER_OF_CONTRASTS] * cbeta[4];
            beta[c] += 1.0f/vareps * c_ctxtxc_GLM[5 + c * NUMBER_OF_CONTRASTS] * cbeta[5];
            beta[c] += 1.0f/vareps * c_ctxtxc_GLM[6 + c * NUMBER_OF_CONTRASTS] * cbeta[6];
            beta[c] += 1.0f/vareps * c_ctxtxc_GLM[7 + c * NUMBER_OF_CONTRASTS] * cbeta[7];
            
            break;
            
        case 9:
            
            beta[c] += 1.0f/vareps * c_ctxtxc_GLM[0 + c * NUMBER_OF_CONTRASTS] * cbeta[0];
            beta[c] += 1.0f/vareps * c_ctxtxc_GLM[1 + c * NUMBER_OF_CONTRASTS] * cbeta[1];
            beta[c] += 1.0f/vareps * c_ctxtxc_GLM[2 + c * NUMBER_OF_CONTRASTS] * cbeta[2];
            beta[c] += 1.0f/vareps * c_ctxtxc_GLM[3 + c * NUMBER_OF_CONTRASTS] * cbeta[3];
            beta[c] += 1.0f/vareps * c_ctxtxc_GLM[4 + c * NUMBER_OF_CONTRASTS] * cbeta[4];
            beta[c] += 1.0f/vareps * c_ctxtxc_GLM[5 + c * NUMBER_OF_CONTRASTS] * cbeta[5];
            beta[c] += 1.0f/vareps * c_ctxtxc_GLM[6 + c * NUMBER_OF_CONTRASTS] * cbeta[6];
            beta[c] += 1.0f/vareps * c_ctxtxc_GLM[7 + c * NUMBER_OF_CONTRASTS] * cbeta[7];
            beta[c] += 1.0f/vareps * c_ctxtxc_GLM[8 + c * NUMBER_OF_CONTRASTS] * cbeta[8];
            
            break;
            
        case 10:
            
            beta[c] += 1.0f/vareps * c_ctxtxc_GLM[0 + c * NUMBER_OF_CONTRASTS] * cbeta[0];
            beta[c] += 1.0f/vareps * c_ctxtxc_GLM[1 + c * NUMBER_OF_CONTRASTS] * cbeta[1];
            beta[c] += 1.0f/vareps * c_ctxtxc_GLM[2 + c * NUMBER_OF_CONTRASTS] * cbeta[2];
            beta[c] += 1.0f/vareps * c_ctxtxc_GLM[3 + c * NUMBER_OF_CONTRASTS] * cbeta[3];
            beta[c] += 1.0f/vareps * c_ctxtxc_GLM[4 + c * NUMBER_OF_CONTRASTS] * cbeta[4];
            beta[c] += 1.0f/vareps * c_ctxtxc_GLM[5 + c * NUMBER_OF_CONTRASTS] * cbeta[5];
            beta[c] += 1.0f/vareps * c_ctxtxc_GLM[6 + c * NUMBER_OF_CONTRASTS] * cbeta[6];
            beta[c] += 1.0f/vareps * c_ctxtxc_GLM[7 + c * NUMBER_OF_CONTRASTS] * cbeta[7];
            beta[c] += 1.0f/vareps * c_ctxtxc_GLM[8 + c * NUMBER_OF_CONTRASTS] * cbeta[8];
            beta[c] += 1.0f/vareps * c_ctxtxc_GLM[9 + c * NUMBER_OF_CONTRASTS] * cbeta[9];
            
            break;		
            
        default:
            1;
            break;
    }	
    
    return 0;	
}			






int CalculateCTXTXCCBetas(__private float* beta, float vareps, __constant float* c_ctxtxc_GLM, __private float* cbeta, int NUMBER_OF_CONTRASTS)
{
    switch(NUMBER_OF_CONTRASTS)
    {
        case 1:
            
            CalculateCTXTXCCBeta(beta, vareps, c_ctxtxc_GLM, cbeta, 0, NUMBER_OF_CONTRASTS);
            
            break;
            
        case 2:
            
            CalculateCTXTXCCBeta(beta, vareps, c_ctxtxc_GLM, cbeta, 0, NUMBER_OF_CONTRASTS);
            CalculateCTXTXCCBeta(beta, vareps, c_ctxtxc_GLM, cbeta, 1, NUMBER_OF_CONTRASTS);
            
            break;
            
        case 3:
            
            CalculateCTXTXCCBeta(beta, vareps, c_ctxtxc_GLM, cbeta, 0, NUMBER_OF_CONTRASTS);
            CalculateCTXTXCCBeta(beta, vareps, c_ctxtxc_GLM, cbeta, 1, NUMBER_OF_CONTRASTS);
            CalculateCTXTXCCBeta(beta, vareps, c_ctxtxc_GLM, cbeta, 2, NUMBER_OF_CONTRASTS);
            
            break;
            
        case 4:
            
            CalculateCTXTXCCBeta(beta, vareps, c_ctxtxc_GLM, cbeta, 0, NUMBER_OF_CONTRASTS);
            CalculateCTXTXCCBeta(beta, vareps, c_ctxtxc_GLM, cbeta, 1, NUMBER_OF_CONTRASTS);
            CalculateCTXTXCCBeta(beta, vareps, c_ctxtxc_GLM, cbeta, 2, NUMBER_OF_CONTRASTS);
            CalculateCTXTXCCBeta(beta, vareps, c_ctxtxc_GLM, cbeta, 3, NUMBER_OF_CONTRASTS);
            
            break;
            
        case 5:
            
            CalculateCTXTXCCBeta(beta, vareps, c_ctxtxc_GLM, cbeta, 0, NUMBER_OF_CONTRASTS);
            CalculateCTXTXCCBeta(beta, vareps, c_ctxtxc_GLM, cbeta, 1, NUMBER_OF_CONTRASTS);
            CalculateCTXTXCCBeta(beta, vareps, c_ctxtxc_GLM, cbeta, 2, NUMBER_OF_CONTRASTS);
            CalculateCTXTXCCBeta(beta, vareps, c_ctxtxc_GLM, cbeta, 3, NUMBER_OF_CONTRASTS);
            CalculateCTXTXCCBeta(beta, vareps, c_ctxtxc_GLM, cbeta, 4, NUMBER_OF_CONTRASTS);
            
            break;
            
        case 6:
            
            CalculateCTXTXCCBeta(beta, vareps, c_ctxtxc_GLM, cbeta, 0, NUMBER_OF_CONTRASTS);
            CalculateCTXTXCCBeta(beta, vareps, c_ctxtxc_GLM, cbeta, 1, NUMBER_OF_CONTRASTS);
            CalculateCTXTXCCBeta(beta, vareps, c_ctxtxc_GLM, cbeta, 2, NUMBER_OF_CONTRASTS);
            CalculateCTXTXCCBeta(beta, vareps, c_ctxtxc_GLM, cbeta, 3, NUMBER_OF_CONTRASTS);
            CalculateCTXTXCCBeta(beta, vareps, c_ctxtxc_GLM, cbeta, 4, NUMBER_OF_CONTRASTS);
            CalculateCTXTXCCBeta(beta, vareps, c_ctxtxc_GLM, cbeta, 5, NUMBER_OF_CONTRASTS);
            
            break;
            
        case 7:
            
            CalculateCTXTXCCBeta(beta, vareps, c_ctxtxc_GLM, cbeta, 0, NUMBER_OF_CONTRASTS);
            CalculateCTXTXCCBeta(beta, vareps, c_ctxtxc_GLM, cbeta, 1, NUMBER_OF_CONTRASTS);
            CalculateCTXTXCCBeta(beta, vareps, c_ctxtxc_GLM, cbeta, 2, NUMBER_OF_CONTRASTS);
            CalculateCTXTXCCBeta(beta, vareps, c_ctxtxc_GLM, cbeta, 3, NUMBER_OF_CONTRASTS);
            CalculateCTXTXCCBeta(beta, vareps, c_ctxtxc_GLM, cbeta, 4, NUMBER_OF_CONTRASTS);
            CalculateCTXTXCCBeta(beta, vareps, c_ctxtxc_GLM, cbeta, 5, NUMBER_OF_CONTRASTS);
            CalculateCTXTXCCBeta(beta, vareps, c_ctxtxc_GLM, cbeta, 6, NUMBER_OF_CONTRASTS);
            
            break;
            
        case 8:
            
            CalculateCTXTXCCBeta(beta, vareps, c_ctxtxc_GLM, cbeta, 0, NUMBER_OF_CONTRASTS);
            CalculateCTXTXCCBeta(beta, vareps, c_ctxtxc_GLM, cbeta, 1, NUMBER_OF_CONTRASTS);
            CalculateCTXTXCCBeta(beta, vareps, c_ctxtxc_GLM, cbeta, 2, NUMBER_OF_CONTRASTS);
            CalculateCTXTXCCBeta(beta, vareps, c_ctxtxc_GLM, cbeta, 3, NUMBER_OF_CONTRASTS);
            CalculateCTXTXCCBeta(beta, vareps, c_ctxtxc_GLM, cbeta, 4, NUMBER_OF_CONTRASTS);
            CalculateCTXTXCCBeta(beta, vareps, c_ctxtxc_GLM, cbeta, 5, NUMBER_OF_CONTRASTS);
            CalculateCTXTXCCBeta(beta, vareps, c_ctxtxc_GLM, cbeta, 6, NUMBER_OF_CONTRASTS);
            CalculateCTXTXCCBeta(beta, vareps, c_ctxtxc_GLM, cbeta, 7, NUMBER_OF_CONTRASTS);
            
            break;
            
        case 9:
            
            CalculateCTXTXCCBeta(beta, vareps, c_ctxtxc_GLM, cbeta, 0, NUMBER_OF_CONTRASTS);
            CalculateCTXTXCCBeta(beta, vareps, c_ctxtxc_GLM, cbeta, 1, NUMBER_OF_CONTRASTS);
            CalculateCTXTXCCBeta(beta, vareps, c_ctxtxc_GLM, cbeta, 2, NUMBER_OF_CONTRASTS);
            CalculateCTXTXCCBeta(beta, vareps, c_ctxtxc_GLM, cbeta, 3, NUMBER_OF_CONTRASTS);
            CalculateCTXTXCCBeta(beta, vareps, c_ctxtxc_GLM, cbeta, 4, NUMBER_OF_CONTRASTS);
            CalculateCTXTXCCBeta(beta, vareps, c_ctxtxc_GLM, cbeta, 5, NUMBER_OF_CONTRASTS);
            CalculateCTXTXCCBeta(beta, vareps, c_ctxtxc_GLM, cbeta, 6, NUMBER_OF_CONTRASTS);
            CalculateCTXTXCCBeta(beta, vareps, c_ctxtxc_GLM, cbeta, 7, NUMBER_OF_CONTRASTS);
            CalculateCTXTXCCBeta(beta, vareps, c_ctxtxc_GLM, cbeta, 8, NUMBER_OF_CONTRASTS);
            
            break;
            
        case 10:
            
            CalculateCTXTXCCBeta(beta, vareps, c_ctxtxc_GLM, cbeta, 0, NUMBER_OF_CONTRASTS);
            CalculateCTXTXCCBeta(beta, vareps, c_ctxtxc_GLM, cbeta, 1, NUMBER_OF_CONTRASTS);
            CalculateCTXTXCCBeta(beta, vareps, c_ctxtxc_GLM, cbeta, 2, NUMBER_OF_CONTRASTS);
            CalculateCTXTXCCBeta(beta, vareps, c_ctxtxc_GLM, cbeta, 3, NUMBER_OF_CONTRASTS);
            CalculateCTXTXCCBeta(beta, vareps, c_ctxtxc_GLM, cbeta, 4, NUMBER_OF_CONTRASTS);
            CalculateCTXTXCCBeta(beta, vareps, c_ctxtxc_GLM, cbeta, 5, NUMBER_OF_CONTRASTS);
            CalculateCTXTXCCBeta(beta, vareps, c_ctxtxc_GLM, cbeta, 6, NUMBER_OF_CONTRASTS);
            CalculateCTXTXCCBeta(beta, vareps, c_ctxtxc_GLM, cbeta, 7, NUMBER_OF_CONTRASTS);
            CalculateCTXTXCCBeta(beta, vareps, c_ctxtxc_GLM, cbeta, 8, NUMBER_OF_CONTRASTS);
            CalculateCTXTXCCBeta(beta, vareps, c_ctxtxc_GLM, cbeta, 9, NUMBER_OF_CONTRASTS);
            
            break;		
            
        default:
            1;
            break;
    }	
    
    return 0;	
}



float CalculateFTestScalar(__private float* cbeta, __private float* beta, int NUMBER_OF_CONTRASTS)
{
    float scalar = 0.0f;
    
    switch(NUMBER_OF_CONTRASTS)
    {
        case 1:
            
            scalar += cbeta[0] * beta[0];
            
            break;
            
        case 2:
            
            scalar += cbeta[0] * beta[0];
            scalar += cbeta[1] * beta[1];
            
            break;
            
        case 3:
            
            scalar += cbeta[0] * beta[0];
            scalar += cbeta[1] * beta[1];
            scalar += cbeta[2] * beta[2];
            
            break;
            
        case 4:
            
            scalar += cbeta[0] * beta[0];
            scalar += cbeta[1] * beta[1];
            scalar += cbeta[2] * beta[2];
            scalar += cbeta[3] * beta[3];
            
            break;
            
        case 5:
            
            scalar += cbeta[0] * beta[0];
            scalar += cbeta[1] * beta[1];
            scalar += cbeta[2] * beta[2];
            scalar += cbeta[3] * beta[3];
            scalar += cbeta[4] * beta[4];
            
            break;
            
        case 6:
            
            scalar += cbeta[0] * beta[0];
            scalar += cbeta[1] * beta[1];
            scalar += cbeta[2] * beta[2];
            scalar += cbeta[3] * beta[3];
            scalar += cbeta[4] * beta[4];
            scalar += cbeta[5] * beta[5];
            
            break;
            
        case 7:
            
            scalar += cbeta[0] * beta[0];
            scalar += cbeta[1] * beta[1];
            scalar += cbeta[2] * beta[2];
            scalar += cbeta[3] * beta[3];
            scalar += cbeta[4] * beta[4];
            scalar += cbeta[5] * beta[5];
            scalar += cbeta[6] * beta[6];
            scalar += cbeta[7] * beta[7];
            
            break;
            
        case 8:
            
            scalar += cbeta[0] * beta[0];
            scalar += cbeta[1] * beta[1];
            scalar += cbeta[2] * beta[2];
            scalar += cbeta[3] * beta[3];
            scalar += cbeta[4] * beta[4];
            scalar += cbeta[5] * beta[5];
            scalar += cbeta[6] * beta[6];
            scalar += cbeta[7] * beta[7];
            
            break;
            
        case 9:
            
            scalar += cbeta[0] * beta[0];
            scalar += cbeta[1] * beta[1];
            scalar += cbeta[2] * beta[2];
            scalar += cbeta[3] * beta[3];
            scalar += cbeta[4] * beta[4];
            scalar += cbeta[5] * beta[5];
            scalar += cbeta[6] * beta[6];
            scalar += cbeta[7] * beta[7];
            scalar += cbeta[8] * beta[8];
            
            break;
            
        case 10:
            
            scalar += cbeta[0] * beta[0];
            scalar += cbeta[1] * beta[1];
            scalar += cbeta[2] * beta[2];
            scalar += cbeta[3] * beta[3];
            scalar += cbeta[4] * beta[4];
            scalar += cbeta[5] * beta[5];
            scalar += cbeta[6] * beta[6];
            scalar += cbeta[7] * beta[7];
            scalar += cbeta[8] * beta[8];
            scalar += cbeta[9] * beta[9];
            
            break;		
            
        default:
            1;
            break;
    }	
    
    return scalar;
}



// Optimized kernel for calculating F-test values for permutations, second level

__kernel void CalculateStatisticalMapsGLMFTestSecondLevelPermutation(__global float* Statistical_Maps,
                                                                     __global const float* Volumes,
                                                                     __global const float* Mask,
                                                                     __constant float* c_X_GLM,
                                                                     __constant float* c_xtxxt_GLM,
                                                                     __constant float* c_Contrasts,
                                                                     __constant float* c_ctxtxc_GLM,
                                                                     __constant unsigned short int* c_Permutation_Vector,
                                                                     __private int DATA_W,
                                                                     __private int DATA_H,
                                                                     __private int DATA_D,
                                                                     __private int NUMBER_OF_VOLUMES,
                                                                     __private int NUMBER_OF_REGRESSORS,
                                                                     __private int NUMBER_OF_CONTRASTS)
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
        CalculateBetaWeightsSecondLevel(beta, value, c_xtxxt_GLM, v, c_Permutation_Vector, NUMBER_OF_VOLUMES, NUMBER_OF_REGRESSORS);
    }
    
    // Calculate the mean and variance of the error eps
    /*
     meaneps = 0.0f;
     vareps = 0.0f;
     float n = 0.0f;
     for (int v = 0; v < NUMBER_OF_VOLUMES; v++)
     {
     eps = Volumes[Calculate4DIndex(x,y,z,v,DATA_W,DATA_H,DATA_D)];
     
     // Loop over regressors using unrolled code for performance
     eps = CalculateEpsSecondLevel(eps, beta, c_X_GLM, v, c_Permutation_Vector, NUMBER_OF_VOLUMES, NUMBER_OF_REGRESSORS);
     
     n += 1.0f;
     float delta = eps - meaneps;
     meaneps += delta/n;
     vareps += delta * (eps - meaneps);
     }
     vareps = vareps / (n - 1.0f);
     */
    
    vareps = 0.0f;
    float n = 0.0f;
    for (int v = 0; v < NUMBER_OF_VOLUMES; v++)
    {
        eps = Volumes[Calculate4DIndex(x,y,z,v,DATA_W,DATA_H,DATA_D)];
        
        // Loop over regressors using unrolled code for performance
        eps = CalculateEpsSecondLevel(eps, beta, c_X_GLM, v, c_Permutation_Vector, NUMBER_OF_VOLUMES, NUMBER_OF_REGRESSORS);
        
        vareps += eps * eps;
    }
    vareps = vareps / ((float)NUMBER_OF_VOLUMES - NUMBER_OF_REGRESSORS);
    
    //-------------------------
    
    // Calculate matrix vector product C*beta (minus u)
    float cbeta[10];
    CalculateCBetas(cbeta, beta, c_Contrasts, NUMBER_OF_REGRESSORS, NUMBER_OF_CONTRASTS);
    
    // Calculate total vector matrix vector product (C*beta)^T ( 1/vareps * (C^T (X^T X)^(-1) C^T)^(-1) ) (C*beta)
    
    // Calculate right hand side, temp = ( 1/vareps * (C^T (X^T X)^(-1) C^T)^(-1) ) (C*beta)
    CalculateCTXTXCCBetas(beta, vareps, c_ctxtxc_GLM, cbeta, NUMBER_OF_CONTRASTS);
    
    // Finally calculate (C*beta)^T * temp
    float scalar = CalculateFTestScalar(cbeta,beta,NUMBER_OF_CONTRASTS);
    
    // Save F-value
    Statistical_Maps[Calculate3DIndex(x,y,z,DATA_W,DATA_H)] = scalar/(float)NUMBER_OF_CONTRASTS;
}


