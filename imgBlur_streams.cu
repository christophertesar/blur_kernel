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

///////////////////////////////////////////////////////
//@@ INSERT YOUR CODE HERE
// __device__ float getPixel(float* image, int x, int y, int width, int height) {
//     // Clamp to ensure boundary conditions are handled appropriately
//     x = max(0, min(x, width - 1));
//     y = max(0, min(y, height - 1));
//     return image[y * width + x];
// }

__global__ void blurKernel(float *out, float *in, int width, int height) {

    __shared__ float sharedData[(TILE_SIZE + 2*BLUR_SIZE) * (TILE_SIZE + 2*BLUR_SIZE)];

    int col_global = blockIdx.x * blockDim.x + threadIdx.x;
    int row_global = blockIdx.y * blockDim.y + threadIdx.y;
    int localIdx_X = threadIdx.x;
    int localIdx_Y = threadIdx.y;

    //sharedData[localIdx_Y * (TILE_WIDTH + 2*BLUR_SIZE) + localIdx_X] = in[row_global * width + col_global];
    int total_number_of_vals_in_buffer = (TILE_SIZE + 2*BLUR_SIZE) * (TILE_SIZE + 2*BLUR_SIZE);
    int lidx = threadIdx.y * blockDim.x + threadIdx.x;
    while (lidx < total_number_of_vals_in_buffer){
        int x = (blockIdx.x * blockDim.x) - BLUR_SIZE + (lidx%(TILE_SIZE + 2*BLUR_SIZE));
        int y = (blockIdx.y * blockDim.y) - BLUR_SIZE + (lidx/(TILE_SIZE + 2*BLUR_SIZE));
        if(x >= 0 && x < width && y >= 0 && y < height){
            sharedData[lidx] = in[y * width + x];
        }
        else{
            sharedData[lidx] = -1;
        }
        lidx += blockDim.x * blockDim.y;
    }

    __syncthreads();

    if(col_global < width && row_global < height){
        float pixVal = 0;
        int pixels = 0;
        
        for(int blurRow = -BLUR_SIZE; blurRow < BLUR_SIZE+1; ++blurRow){
            for(int blurCol = -BLUR_SIZE; blurCol < BLUR_SIZE+1; ++blurCol){
                if(sharedData[(localIdx_Y + BLUR_SIZE - blurRow) * (TILE_SIZE + 2*BLUR_SIZE) + (localIdx_X + BLUR_SIZE - blurCol)] >= 0){
                    pixVal += sharedData[(localIdx_Y + BLUR_SIZE - blurRow) * (TILE_SIZE + 2*BLUR_SIZE) + (localIdx_X + BLUR_SIZE - blurCol)];
                    pixels++;
                }
            }
        }
        out[row_global * width + col_global] = pixVal / pixels;
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
  wbImage_t goldImage;
  float* hostInputImageData;
  float* hostOutputImageData;
  float* deviceInputImageData;
  float* deviceOutputImageData;
  float *goldOutputImageData;

  args = wbArg_read(argc, argv); /* parse the input arguments */

  inputImageFile = wbArg_getInputFile(args, 0);
  inputImage = wbImport(inputImageFile);

  char *goldImageFile = argv[2];
  goldImage = wbImport(goldImageFile);

  // The input image is in grayscale, so the number of channels is 1
  imageWidth  = wbImage_getWidth(inputImage);
  imageHeight = wbImage_getHeight(inputImage);

  // Since the image is monochromatic, it only contains only one channel
  outputImage = wbImage_new(imageWidth, imageHeight, 1);

  // Get host input and output image data
    hostInputImageData = wbImage_getData(inputImage);
    hostOutputImageData = wbImage_getData(outputImage);
  goldOutputImageData = wbImage_getData(goldImage);
 

    // cudaMalloc((void **)&deviceInputImageData,
    //             imageWidth * imageHeight * sizeof(float));
    // cudaMalloc((void **)&deviceOutputImageData,
    //             imageWidth * imageHeight * sizeof(float));

    cudaHostAlloc((void **)&deviceInputImageData, imageWidth * imageHeight * sizeof(float), cudaHostAllocDefault);
    cudaHostAlloc((void **)&deviceOutputImageData, imageWidth * imageHeight * sizeof(float), cudaHostAllocDefault);

//   cudaHostAlloc((void **)&deviceInputImageData, imageWidth * imageHeight * sizeof(float), cudaHostAllocDefault);
//   cudaHostAlloc((void **)&deviceOutputImageData, imageWidth * imageHeight * sizeof(float), cudaHostAllocDefault);

  int iterations = 10;

    cudaStream_t stream[iterations];
    for (int i = 0; i < iterations; ++i) {
        cudaStreamCreate(&stream[i]);
    }

  // Start timer
  timespec timer = tic();
  
  ////////////////////////////////////////////////
  //@@ INSERT AND UPDATE YOUR CODE HERE

  // Allocate cuda memory for device input and ouput image data


  // Transfer data from CPU to GPU
    // cudaMemcpy(deviceInputImageData, hostInputImageData,
    //             imageWidth * imageHeight * sizeof(float),
    //             cudaMemcpyHostToDevice);

    cudaMemcpy(deviceInputImageData, hostInputImageData,
        imageWidth * imageHeight * sizeof(float),
        cudaMemcpyHostToDevice);
    

  dim3 blockSize(TILE_SIZE, TILE_SIZE);
  dim3 gridSize((imageWidth + blockSize.x - 1) / blockSize.x, (imageHeight + blockSize.y - 1) / blockSize.y);

  // Call your GPU kernel 10 times
  for(int i = 0; i < iterations; i++)
  blurKernel<<<gridSize, blockSize, 0, stream[i]>>>(deviceOutputImageData,
                                    deviceInputImageData, imageWidth,
                                    imageHeight);

// Transfer data from GPU to CPU
//   cudaMemcpy(hostOutputImageData, deviceOutputImageData,
//              imageWidth * imageHeight * sizeof(float),
//              cudaMemcpyDeviceToHost);
///////////////////////////////////////////////////////

    cudaMemcpy(hostOutputImageData, deviceOutputImageData,
                    imageWidth * imageHeight * sizeof(float),
                    cudaMemcpyDeviceToHost);
  
  // Stop and print timer
  toc(&timer, "GPU execution time (including data transfer) in seconds");

for (int i = 0; i < iterations; ++i) {
    cudaStreamSynchronize(stream[i]);
}

// Destroy CUDA streams
for (int i = 0; i < iterations; ++i) {
    cudaStreamDestroy(stream[i]);
}

  // Check the correctness of your solution
  //wbSolution(args, outputImage);

    for(int i=0; i<imageHeight; i++){
        for(int j=0; j<imageWidth; j++){
            if(abs(hostOutputImageData[i*imageWidth+j]-goldOutputImageData[i*imageWidth+j])/goldOutputImageData[i*imageWidth+j]>0.01){
                printf("Incorrect output image at pixel (%d, %d): goldOutputImage = %f, hostOutputImage = %f\n", i, j, goldOutputImageData[i*imageWidth+j],hostOutputImageData[i*imageWidth+j]);
                return -1;
            }
        }
    }
   printf("Correct output image!\n");

  cudaFreeHost(deviceInputImageData);
  cudaFreeHost(deviceOutputImageData);

  wbImage_delete(outputImage);
  wbImage_delete(inputImage);

  return 0;
}
