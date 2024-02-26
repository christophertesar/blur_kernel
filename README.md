# GPU Image Blur with CUDA

This project demonstrates how to perform image blurring on the GPU using CUDA.

## Requirements

To compile and run the code, you need:

- CUDA-enabled GPU
- CUDA Toolkit installed
- `libwb` library (included)

## Usage

1. Compile the code using a CUDA compiler. For example:

    ```bash
    nvcc -o blur_image blur_image.cu -I./libwb
    ```

2. Run the executable with an input image file as an argument. For example:

    ```bash
    ./blur_image inputImage.ppm
    ```

3. The program will blur the input image using the GPU and output the result to a new image file.

## Code Overview

The code consists of two main parts:

1. `blurKernel`: The CUDA kernel responsible for applying the blur filter to the input image. The kernel is executed on the GPU and processes each pixel of the image in parallel.

2. `main`: The main function of the program. It loads an input image, allocates memory on the GPU, transfers data between the CPU and GPU, launches the CUDA kernel to perform the blur operation, and finally, saves the resulting image.

## Parameters

- `BLUR_SIZE`: The size of the blur filter window.
- `TILE_SIZE`: The size of the thread block used for parallel processing.

## Performance Considerations

The kernel utilizes shared memory to minimize off-chip memory access and improve performance. Each thread block reads a tile of the image into shared memory, ensuring that data is reused efficiently.

## Authors

- Christopher Tesar
