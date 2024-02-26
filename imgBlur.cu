#include "libwb/wb.h"
#include "my_timer.h"
#include <iostream>

#define wbCheck(stmt)							\
  do {									\
    cudaError_t err = stmt;						\
    if (err != cudaSuccess) {						\
      wbLog(ERROR, "Failed to run stmt ", #stmt);			\
      wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));	\
      return -1;							\
    }									\
  } while (0)

#define BLUR_SIZE 21
#define TILE_SIZE 32

__device__ float getPixel(float* image, int x, int y, int width, int height) {
    if(x >= 0 && x < width && y >= 0 && y < height){
        return in[y * width + x];
    }
    else{
        return -1;
    }
}

__device__ bool isPixelInImage(int col_global, int width, int row_global, int height){
    return col_global < width && row_global < height;
}

__device__ bool isValidPixel(int pixel){
    return pixel >= 0;
}

/*
Blur Kernel
This blur kernel calculates blur by blocking the image into tile. 
Each block reads in (TILE_SIZE + 2*BLUR_SIZE) * (TILE_SIZE + 2*BLUR_SIZE) into shared memory so for each block
that is blurred, local memory is read in once and reused, resulting in minimal offchip memory access.
Each thread calculates a blur for a single pixel.
*/
__global__ void blurKernel(float *out, float *in, int width, int height) {

    __shared__ float sharedData[(TILE_SIZE + 2*BLUR_SIZE) * (TILE_SIZE + 2*BLUR_SIZE)];

    //Column and Row with respect to all blocks in GPU.
    int col_global = blockIdx.x * blockDim.x + threadIdx.x;
    int row_global = blockIdx.y * blockDim.y + threadIdx.y;
    //The X, Y "coordinates" of a thread within a block.
    int localIdx_X = threadIdx.x;
    int localIdx_Y = threadIdx.y;

    int total_number_of_vals_in_buffer = (TILE_SIZE + 2*BLUR_SIZE) * (TILE_SIZE + 2*BLUR_SIZE);
    //Local index but translated into a 1D array index.
    int local_idx = threadIdx.y * blockDim.x + threadIdx.x;

    //Since we need to load more than "block" number of pixels into local memory, each thread is responsible for
    //loading multiple pixels into shared memory.
    while (local_idx < total_number_of_vals_in_buffer){
        //Obtain the x, y coordinates of the pixels we want to load in the picture.
        int x = (blockIdx.x * blockDim.x) - BLUR_SIZE + (local_idx%(TILE_SIZE + 2*BLUR_SIZE));
        int y = (blockIdx.y * blockDim.y) - BLUR_SIZE + (local_idx/(TILE_SIZE + 2*BLUR_SIZE));
        sharedData[local_idx] = getPixel(in, x, y, width, height);
        local_idx += blockDim.x * blockDim.y;
    }
    
    //Make sure we are waiting for all threads to finish writing to local memory.
    __syncthreads();

    if(isPixelInImage(col_global, width, row_global, height)){
        float pixVal = 0;
        int pixels = 0;
        //Sum up a window around the pixel we are observing to take average for blurring.
        for(int blurRow = -BLUR_SIZE; blurRow < BLUR_SIZE+1; ++blurRow){
            for(int blurCol = -BLUR_SIZE; blurCol < BLUR_SIZE+1; ++blurCol){
                if(isValidPixel(sharedData[(localIdx_Y + BLUR_SIZE - blurRow) * (TILE_SIZE + 2*BLUR_SIZE) + (localIdx_X + BLUR_SIZE - blurCol)])){
                    pixVal += sharedData[(localIdx_Y + BLUR_SIZE - blurRow) * (TILE_SIZE + 2*BLUR_SIZE) + (localIdx_X + BLUR_SIZE - blurCol)];
                    pixels++;
                }
            }
        }
        //Take average to blur image.
        out[row_global * width + col_global] = (pixVal / pixels);
    }
}

///////////////////////////////////////////////////////

int main(int argc, char *argv[]) {
  wbArg_t args;
  int imageWidth;
  int imageHeight;
  char *inputImageFile;
  wbImage_t inputImage;
  wbImage_t outputImage;
  float *hostInputImageData;
  float *hostOutputImageData;
  float *deviceInputImageData;
  float *deviceOutputImageData;

  args = wbArg_read(argc, argv); /* parse the input arguments */

  inputImageFile = wbArg_getInputFile(args, 0);
  inputImage = wbImport(inputImageFile);

  // The input image is in grayscale, so the number of channels is 1
  imageWidth  = wbImage_getWidth(inputImage);
  imageHeight = wbImage_getHeight(inputImage);

  // Since the image is monochromatic, it only contains only one channel
  outputImage = wbImage_new(imageWidth, imageHeight, 1);

  // Get host input and output image data
  hostInputImageData  = wbImage_getData(inputImage);
  hostOutputImageData = wbImage_getData(outputImage);

  // Allocate cuda memory for device input and ouput image data
  cudaMalloc((void **)&deviceInputImageData,
  imageWidth * imageHeight * sizeof(float));
  cudaMalloc((void **)&deviceOutputImageData,
  imageWidth * imageHeight * sizeof(float));

  // Start timer
  timespec timer = tic();

  // Transfer data from CPU to GPU
  cudaMemcpy(deviceInputImageData, hostInputImageData,
             imageWidth * imageHeight * sizeof(float),
             cudaMemcpyHostToDevice);

  dim3 blockSize(TILE_SIZE, TILE_SIZE);

  //Create grid based on how many pixels there are in the image. Make sure grid rounds up to block size.
  int gridWidth = (imageWidth + blockSize.x - 1) / blockSize.x;
  int gridHeight = (imageHeight + blockSize.y - 1) / blockSize.y;

  dim3 gridSize(gridWidth, gridHeight);

  // Call GPU kernel 10 times
  for(int i = 0; i < 10; i++)
  blurKernel<<<gridSize, blockSize>>>(deviceOutputImageData,
                                    deviceInputImageData, imageWidth,
                                    imageHeight);

  // Transfer data from GPU to CPU
  cudaMemcpy(hostOutputImageData, deviceOutputImageData,
             imageWidth * imageHeight * sizeof(float),
             cudaMemcpyDeviceToHost);
  ///////////////////////////////////////////////////////
  
  // Stop and print timer
  toc(&timer, "GPU execution time (including data transfer) in seconds");

  cudaFree(deviceInputImageData);
  cudaFree(deviceOutputImageData);

  wbImage_delete(outputImage);
  wbImage_delete(inputImage);

  return 0;
}
