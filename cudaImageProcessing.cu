#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <string.h>
#include <png.h>
#include <omp.h>

extern "C"
{
#include "png_util.h"
}

#define NX 32
#define NY 32
#define dfloat float

#define Z 32

// Reads a png file and stores the RGB values in an unallocated array rgb
void read_png_1D(char* filename, unsigned char** rgb, int* width, int* height){
  png_byte color_type;
  png_byte bit_depth;

  png_structp png_ptr;
  png_infop info_ptr;
  png_bytep* row_pointers;
  char header[8];
  FILE *fp = fopen(filename, "rb");
  fread(header, 1, 8, fp);
  png_ptr = png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
  info_ptr = png_create_info_struct(png_ptr);

  png_jmpbuf(png_ptr);

  png_init_io(png_ptr, fp);
  png_set_sig_bytes(png_ptr, 8);

  png_read_info(png_ptr, info_ptr);

  *width = png_get_image_width(png_ptr, info_ptr);
  *height = png_get_image_height(png_ptr, info_ptr);
  bit_depth = png_get_bit_depth(png_ptr, info_ptr);
  color_type = png_get_color_type(png_ptr, info_ptr);

  png_read_update_info(png_ptr, info_ptr);
  setjmp(png_jmpbuf(png_ptr));

  row_pointers = (png_bytep*) malloc(sizeof(png_bytep) * *height);
  for (int i = 0; i < *height; i++){
    row_pointers[i] = (png_byte*) malloc(png_get_rowbytes(png_ptr, info_ptr));
  }
  png_read_image(png_ptr, row_pointers);

  *rgb = (unsigned char*) malloc(3 * *height * *width);
  for (int i = 0; i < *height; i++){
        for (int j = 0; j < *width*3; j++){
            (*rgb)[i*(*width)*3 + j] = row_pointers[i][j];
        }
  }

  fclose(fp);
}

// Haven't used since its easier to just copy over grayscale data to DEVICE.
__global__ void FlattenGrayscaleKernel(unsigned char* rgb, unsigned char* gray, int height, int width){
    int idx = (threadIdx.x) + blockDim.x * blockIdx.x;    
    
    if (idx < width*height){
        float px = 0.2126 * rgb[idx * 3 + 0] + 
				   0.7152 * rgb[idx * 3 + 1] + 
				   0.0722 * rgb[idx * 3 + 2];
        gray[idx] = px;
    }

}

// Converts an RGB Row major array tp greyscale
int convert_to_greyscale_flat(unsigned char* rgb, unsigned char* grey, int height, int width){
    for (int i = 0; i < height; i++){
        for (int j = 0; j < width; j++){
            int rgb_id = 3*(j + i*width);
            int id = j + i*width;
            unsigned int value = (rgb[rgb_id] + rgb[rgb_id+1] + rgb[rgb_id+2]) / 3;
            
            grey[id] = value;
        }
    }
    return 0;
}

// Converts an RGB array to Grayscale
int to_greyscale(unsigned char* rgb,  unsigned char* grey, int height, int width){
    for (int i = 0; i < height; i++){
        for (int j = 0; j < width; j++){
            int id = 3*(j + i*width);
            //int id = j + i*width;
            unsigned int value = (rgb[id] + rgb[id+1] + rgb[id+2]) / 3;
            grey[id] = value;
            grey[id+1] = value;
            grey[id+2] = value;
        }
    }
    return 0;
}

template < int sM > 
__global__ void BoxBlur(unsigned char* in, unsigned char* out, int width, int height, int k_dim) {
    int col = (blockDim.x - 2) * blockIdx.x + threadIdx.x;
    int row = (blockDim.y - 2) * blockIdx.y + threadIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int idx = row * width + col;
    
    __shared__ int s_A[sM][sM];
    s_A[ty][tx] = 0;

    if (row < height && col < width){
        
        s_A[ty][tx] = in[idx];
        __syncthreads();
            
        if ((tx >= k_dim && ty >= k_dim) && (tx < (sM - k_dim) && ty < (sM - k_dim))){

            int n = s_A[ty-1][tx-1] + s_A[ty+1][tx-1] + s_A[ty][tx-1] + s_A[ty-1][tx]
                + s_A[ty+1][tx] + s_A[ty-1][tx+1] + s_A[ty][tx+1] + s_A[ty+1][tx+1]
                + s_A[ty][tx];

            out[idx] = n / 9;

        }

    }
}

// Writes out a flat grayscale array (only 1 channel) to a png
int write_flat_greyscale(char* filename, unsigned char* grey, int width, int height, int padding){
    int pad = padding / 2;
    unsigned char temp_grey[3*(width-padding)*(height-padding)];
    for (int i = pad; i < height - pad; i++){
        for (int j = pad; j < width - pad; j++){
            int idx = (j) + (i)*(width);
            int new_idx = 3*((j - pad) + (i - pad)*(width-padding));
            temp_grey[new_idx] = grey[idx];
            temp_grey[new_idx+1] = grey[idx];
            temp_grey[new_idx+2] = grey[idx];
        }
    }
    
    FILE* outfile;
    outfile = fopen(filename, "w");
    write_png(outfile, (width-padding), (height-padding), temp_grey, NULL);

    return 0;

}

// Copies the unpadded portion of an array to an array of correct size initialized to 0
void pad_with_zeros(unsigned char* input, unsigned char* padded, int width, int height, int pad){
    for(int i = 0; i < height; i++){
        for(int j = 0; j < width; j++){
            int idx = i*(width + pad) + j + (width + pad);
            padded[idx] = input[j + i*width];
        }
    }
}

// Does a row convolution for a separable kernel
template <int sM, typename T>
__global__ void RowConvolution(unsigned char* img, unsigned char* output, T *kernel, int width, int height, int k_dim) {
    
    int k_len = k_dim / 2;

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int col = (sM - 2*k_len) * blockIdx.x + threadIdx.x;
    int row = (sM - 2*k_len) * blockIdx.y + threadIdx.y;
    int idx = row * width + col;

    int max_row = height - k_len;
    int max_col = width - k_len;
    int max_tidx = sM - k_len;

    __shared__ int s_img[sM][sM];

    if (row < height && col < width){
        s_img[ty][tx] = img[idx];
    }

    __syncthreads();

    // Check if we are inside the border to do work on
    if (row >= k_len && row < max_row && col >= k_len && col < max_col){
        if (tx >= k_len && tx < max_tidx && ty >= k_len && ty < max_tidx){
            int new_value = 0;

            #pragma unroll
            for (int i = -(k_dim / 2); i <= (k_dim / 2); i++){
                new_value += kernel[i + (k_dim/2)] * s_img[ty][tx + i];
            }
            output[idx] = new_value;
        }
    }
    
}

// Does a column convolution for a separable kernel
template <int sM, typename T>
__global__ void ColumnConvolution(unsigned char* img, unsigned char* output, T *kernel, int width, int height, int k_dim) {
    
    int k_len = k_dim / 2;

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int col = (sM - 2*k_len) * blockIdx.x + threadIdx.x;
    int row = (sM - 2*k_len) * blockIdx.y + threadIdx.y;
    int idx = row * width + col;

    int max_row = height - k_len;
    int max_col = width - k_len;
    int max_tidx = sM - k_len;

    __shared__ int s_img[sM][sM];

    if (row < height && col < width){
        s_img[ty][tx] = img[idx];
    }
    __syncthreads();
    // Check if we are inside the border to do work on
    if (row >= k_len && row < max_row && col >= k_len && col < max_col){
        if (tx >= k_len && tx < max_tidx && ty >= k_len && ty < max_tidx){
            int new_value = 0;
            #pragma unroll
            for (int i = -(k_dim / 2); i <= (k_dim / 2); i++){
                new_value += kernel[i + (k_dim/2)] * s_img[ty + i][tx];
            }
            output[idx] = new_value;
        }
    }
}

// Seperable convolution oriented towards edge detection
template <int sM, typename T>
__global__ void ConvolutionSeparated(unsigned char* img, unsigned char* output, T *kernel_1, T *kernel_2, int width, int height, int k_dim, int pad) {
    // Setup
    int k_len = k_dim / 2;

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int col = (sM - 2*k_len) * blockIdx.x + threadIdx.x;
    int row = (sM - 2*k_len) * blockIdx.y + threadIdx.y;
    int idx = row * width + col;

    int max_row = height - k_len - pad;
    int max_col = width - k_len - pad;
    int max_tidx = sM - k_len;

    __shared__ int s_img[sM][sM];
    __shared__ int s_tmp_x[sM][sM];
    __shared__ int s_tmp_y[sM][sM];

    s_img[ty][tx] = 0;
    s_tmp_x[ty][tx] = 0;
    s_tmp_y[ty][tx] = 0;
    if (row < height && col < width){
        s_img[ty][tx] = img[idx];
        s_tmp_x[ty][tx] = img[idx];
        s_tmp_y[ty][tx] = img[idx];
    }
    __syncthreads();

    if (row >= k_len+pad && row < max_row && col >= k_len+pad && col < max_col){
        if (tx >= k_len && tx < max_tidx && ty >= k_len && ty < max_tidx){
            int grad_x = 0;
            int grad_y = 0;

            // Do Column Pass:
            #pragma unroll
            for (int i = -k_len; i <= k_len; i++){
                grad_x += kernel_1[i + k_len] * s_img[ty+i][tx];
                grad_y += kernel_2[i + k_len] * s_img[ty+i][tx];
            }

            s_tmp_x[ty][tx] = grad_x;
            s_tmp_y[ty][tx] = grad_y;
            __syncthreads();

            // Do Row Pass:
            grad_x = 0;
            grad_y = 0;
            #pragma unroll
            for (int i = -k_len; i <= k_len; i++){
                grad_x += kernel_2[i + k_len] * s_tmp_x[ty][tx+i];
                grad_y += kernel_1[i + k_len] * s_tmp_y[ty][tx+i];
            }
            output[idx] = sqrt((float) ((grad_x*grad_x) + (grad_y*grad_y)));
            //output[idx] = grad_x;
        }
    }
}

// Calculates the gradient in x and y for a separable kernel
template <int sM, typename T>
__global__ void ConvolutionSeparable(unsigned char* img, unsigned char* g_x, unsigned char* g_y, T *kernel_1, T *kernel_2, int width, int height, int k_dim) {
    // Setup
    int k_len = k_dim / 2;

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int col = (sM - 2*k_len) * blockIdx.x + threadIdx.x;
    int row = (sM - 2*k_len) * blockIdx.y + threadIdx.y;
    int idx = row * width + col;

    int max_row = height - k_len;
    int max_col = width - k_len;
    int max_tidx = sM - k_len;

    __shared__ int s_img[sM][sM];

    s_img[ty][tx] = 0;
    if (row < height && col < width){
        s_img[ty][tx] = img[idx];
    }
    __syncthreads();
    // Check if we are inside the border to do work on
    if (row >= k_len && row < max_row && col >= k_len && col < max_col){
        if (tx >= k_len && tx < max_tidx && ty >= k_len && ty < max_tidx){
            int grad_x = 0;
            int grad_y = 0;

            // Do Column Pass:
            #pragma unroll
            for (int i = -(k_dim / 2); i <= (k_dim / 2); i++){
                grad_x += kernel_1[i + k_len] * s_img[ty+i][tx];
                grad_y += kernel_2[i + k_len] * s_img[ty+i][tx];
            }

            g_x[idx] = grad_x;
            g_y[idx] = grad_y;
        }
    }
}

// Takes in the x and y gradient, does the second pass, and outputs sqrt(Gx^2 + Gy^2)
template <int sM, typename T>
__global__ void EdgeOut(unsigned char* g_x, unsigned char* g_y, unsigned char* output,T *kernel_1, T *kernel_2, int width, int height, int k_dim) {
    // Setup
    int k_len = k_dim / 2;

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int col = (sM - 2*k_len) * blockIdx.x + threadIdx.x;
    int row = (sM - 2*k_len) * blockIdx.y + threadIdx.y;
    int idx = row * width + col;

    int max_row = height - k_len;
    int max_col = width - k_len;
    int max_tidx = sM - k_len;

    __shared__ int s_tmp_x[sM][sM];
    __shared__ int s_tmp_y[sM][sM];

    s_tmp_x[ty][tx] = 0;
    s_tmp_y[ty][tx] = 0;
    if (row < height && col < width){
        s_tmp_x[ty][tx] = g_x[idx];
        s_tmp_y[ty][tx] = g_y[idx];
    }
    __syncthreads();

    if (row >= k_len && row < max_row && col >= k_len && col < max_col){
        if (tx >= k_len && tx < max_tidx && ty >= k_len && ty < max_tidx){
            // Do Row Pass:
            int grad_y = 0;
            int grad_x = 0;
            #pragma unroll
            for (int i = -k_len; i <= k_len; i++){
                grad_x += kernel_1[i + k_len] * s_tmp_x[ty][tx+i];
                grad_y += kernel_2[i + k_len] * s_tmp_y[ty][tx+i];
            }
            output[idx] = sqrt((float) ((grad_x*grad_x) + (grad_y*grad_y)));
        }
    }
}


int main(int argc, char** argv) {
    
    cudaSetDevice(2);

    if (argc < 7){
        printf("Correct usage: ./processing png_input, png_output, blur_true, blur_kernel, edge_true, edge_kernel");
        return 1;
    }
    const int do_blur = atoi(argv[3]);
    char* blur_kernel = argv[4];
    const int do_edge = atoi(argv[5]);
    char* edge_kernel = argv[6];
    
    // Sets up padding depending on kernel 
    int padding = 0; 
    if (do_blur == 1) padding = 4;
    else if (do_blur == 0 && do_edge == 1) padding = 2;

    // Read in image
    unsigned char* img;
    int height, width;
    read_png_1D(argv[1], &img, &width, &height);

    // Convert the image to grayscale
    unsigned char* grey = (unsigned char*) malloc(height * width);
    convert_to_greyscale_flat(img, grey, height, width);
    unsigned char* gray_padded = (unsigned char*) calloc((height + padding) * (padding +width), 1);
    //pad_with_zeros(grey, gray_padded, width, height, padding);
    //height += padding;
    //width += padding;
    
    // Set up any needed DEVICE arrays
    unsigned char *c_img, *c_output, *c_blur, *c_temp;
    unsigned char *c_gx, *c_gy;

    cudaMalloc(&c_blur, height*width), cudaMalloc(&c_img, height*width);
    cudaMalloc(&c_gx, height*width), cudaMalloc(&c_gy, height*width);
    cudaMalloc(&c_output, height*width), cudaMalloc(&c_temp, height*width);
    cudaMemcpy(c_img, grey, height*width, cudaMemcpyHostToDevice);

    dim3 B(NX, NY, 1);
    dim3 G((width + (NX-4) - 1)/(NX-4), (height + (NY-4) - 1)/(NY-4), 1);

    // Just writes out grayscale image if no convolution to do
    if (do_blur == 0 && do_edge == 0){
        write_flat_greyscale(argv[2], grey, width, height, padding);
        return 0;
    }

    // Sets up arrays for filter kernels
    if (do_blur == 1){
        float* kernel_sep;
        
        if (strcmp(blur_kernel, "gaussian") == 0){
            float gauss_sep5[5] = {1/16.f, 4/16.f, 6/16.f, 4/16.f, 1/16.f};
            cudaMalloc(&kernel_sep, 5 * sizeof(float));
            cudaMemcpy(kernel_sep, gauss_sep5, 5*sizeof(float), cudaMemcpyHostToDevice);
        }
        else{
            float box_sep5[5] = {1/5.f, 1/5.f, 1/5.f, 1/5.f, 1/5.f};
            cudaMalloc(&kernel_sep, 5 * sizeof(float));
            cudaMemcpy(kernel_sep, box_sep5, 5*sizeof(float), cudaMemcpyHostToDevice);
        }

        //float box_sep3[3] = {1/3.f, 1/3.f, 1/3.f};
        //float* kernel_sep;

        if (do_edge == 1){
            ColumnConvolution <NX, float> <<< G, B >>> (c_img, c_temp, kernel_sep, width, height, 5);
            RowConvolution <NX, float> <<< G, B >>> (c_temp, c_blur, kernel_sep, width, height, 5);
        }
        else{
            ColumnConvolution <NX, float> <<< G, B >>> (c_img, c_blur, kernel_sep, width, height, 5);
            RowConvolution <NX, float> <<< G, B >>> (c_blur, c_output, kernel_sep, width, height, 5);
        }
        cudaFree(kernel_sep);
    }
    // Sets up arrays for edge detection kernels
    if (do_edge == 1){
        int *filter_x, *filter_y;
        cudaMalloc(&filter_x, 3 * sizeof(int)), 
        cudaMalloc(&filter_y, 3 * sizeof(int));

        if (strcmp(edge_kernel, "sobel") == 0){
            int sobel_1[3] = {1, 0, -1};
            int sobel_2[3] = {1, 2, 1};
            cudaMemcpy(filter_x, sobel_1, 3*sizeof(int), cudaMemcpyHostToDevice);
            cudaMemcpy(filter_y, sobel_2, 3*sizeof(int), cudaMemcpyHostToDevice);
        }
        else{
            int prewitt_x[3] = {1, 0, -1};
            int prewitt_y[3] = {1, 1, 1};
            cudaMemcpy(filter_x, prewitt_x, 3*sizeof(int), cudaMemcpyHostToDevice);
            cudaMemcpy(filter_y, prewitt_y, 3*sizeof(int), cudaMemcpyHostToDevice);
        }
        dim3 B(NX, NY, 1);
        dim3 H((width + (NX-2) - 1)/(NX-2), (height + (NY-2) - 1)/(NY-2), 1);
        if (do_blur == 1){
            ConvolutionSeparated <NX, int><<< H, B >>> (c_blur, c_output, filter_x, filter_y, width, height, 3, 2);
        }
        else{
            ConvolutionSeparated <NX, int><<< H, B >>> (c_img, c_output, filter_x, filter_y, width, height, 3, 0);
            //ConvolutionSeparable <NX, int><<< H, B >>> (c_img, c_gx, c_gy, filter_x, filter_y, width, height, 3);
            //EdgeOut <NX, int><<< H, B >>> (c_gx, c_gy, c_output, filter_y, filter_x, width, height, 3);
        }
        cudaFree(filter_x);
        cudaFree(filter_y);
    }

    // Allocate host array to copy over
    unsigned char* output = (unsigned char*) malloc(height * width);
    cudaMemcpy(output, c_output, height*width, cudaMemcpyDeviceToHost);
    
    write_flat_greyscale(argv[2], output, width, height, padding);

    cudaFree(c_temp), cudaFree(c_blur);
    cudaFree(c_output), cudaFree(c_img);
    cudaFree(c_gx), cudaFree(c_gy);

    return 1;

}

