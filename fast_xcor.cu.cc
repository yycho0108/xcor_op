#if GOOGLE_CUDA

#define EIGEN_USE_GPU
#define EIGEN_USE_THREADS

#include "fast_xcor.h"
#include "tensorflow/core/util/cuda_kernel_helper.h"
#include "tensorflow/core/framework/register_types.h"

#define EIGEN_USE_GPU
using namespace tensorflow;
using GPUDevice = Eigen::GpuDevice;

#define IDX2(i,j, n_j) (((i)*(n_j))+(j))
#define IDX3(i,j,k, n_j,n_k) (IDX2(IDX2(i,j,n_j), (k), (n_k)))
#define IDX4(i,j,k,l, n_j,n_k,n_l) (IDX2(IDX3(i,j,k,n_j,n_k), (l), (n_l)))
#define IDX5(i,j,k,l,m, n_j,n_k,n_l,n_m) (IDX2(IDX4(i,j,k,l,n_j,n_k,n_l), (m), (n_m)))
#define IDX6(i,j,k,l,m,n, n_j,n_k,n_l,n_m,n_n) (IDX2(IDX5(i,j,k,l,m,n_j,n_k,n_l,n_m), (n), (n_n)))

template<typename T>
__global__ void fast_xcor_kernel(
        const int batch_size,
        const int height,
        const int width,
        const int channel,
        const int delta,
        const int stride_h,
        const int stride_w,
        const T* input,
        const T* filter,
        T*       output){

    const int size = (batch_size*height*width*delta*delta*channel);
	for(int thread_id : CudaGridRangeX(size)){
		const int idx_n  = (thread_id / (height*width*delta*delta*channel));
		const int idx_h  = (thread_id / (width*delta*delta*channel)) % height;
		const int idx_w  = (thread_id / (delta*delta*channel)) % width;
		const int idx_dh = (thread_id / (delta*channel)) % delta;
		const int idx_dw = (thread_id / (channel)) % delta;
		const int idx_c  = (thread_id) % channel;

		//printf("assert %d == %d\n", IDX6(idx_n,idx_h,idx_w,idx_dh,idx_dw,idx_c,
		//			height, width, delta, delta, channel), thread_id);

		const int d = (delta-1) / 2;
		const int idx_h2  = idx_h + stride_h * (idx_dh-d);
		const int idx_w2  = idx_w + stride_w * (idx_dw-d);

		const int o_idx = IDX5(idx_n,idx_h,idx_w,idx_dh,idx_dw,
			   	height, width, delta, delta);

		// probably faster to do it here than cudaMemset(...)
		// output[o_idx] = 0;

		if(0<=idx_h2 && idx_h2<height && 0<=idx_w2 && idx_w2<width){

			float inc = input[IDX4(idx_n,idx_h,idx_w,idx_c,
					height, width, channel)] *
						filter[ IDX4(idx_n,idx_h2,idx_w2,idx_c,
					height, width, channel)] / channel;

			atomicAdd(&output[o_idx], inc);

			//for(int idx_c=0; idx_c < channel; ++idx_c){
			//	output[o_idx] +=
			//		input[  IDX4(idx_n,idx_h,idx_w,idx_c,
			//				height, width, channel)] *
			//		filter[ IDX4(idx_n,idx_h2,idx_w2,idx_c,
			//				height, width, channel)] / channel;
			//}
		}
	}
}

template<typename T>
__global__ void fast_xcor_kernel_small(
        const int batch_size,
        const int height,
        const int width,
        const int channel,
        const int delta,
        const int stride_h,
        const int stride_w,
		const T* input,
		const T* filter,
		T*       output){
	// specialization for small # of channels - loops through
	const int size = (batch_size*height*width*delta*delta);

	__shared__ T out_val; // maybe faster?

	for(int thread_id : CudaGridRangeX(size)){
		const int idx_n  = (thread_id / (height*width*delta*delta));
		const int idx_h  = (thread_id / (width*delta*delta)) % height;
		const int idx_w  = (thread_id / (delta*delta)) % width;
		const int idx_dh = (thread_id / (delta)) % delta;
		const int idx_dw = (thread_id) % delta;

		const int d = (delta-1) / 2;
		const int idx_h2  = idx_h + stride_h * (idx_dh-d);
		const int idx_w2  = idx_w + stride_w * (idx_dw-d);
		const int o_idx = IDX5(idx_n,idx_h,idx_w,idx_dh,idx_dw,
				height, width, delta, delta);

		out_val = 0.0;
		if(0<=idx_h2 && idx_h2<height && 0<=idx_w2 && idx_w2<width){
			for(int idx_c=0; idx_c < channel; ++idx_c){
				out_val += input[IDX4(idx_n,idx_h,idx_w,idx_c,
						height, width, channel)] *
					filter[ IDX4(idx_n,idx_h2,idx_w2,idx_c,
							height, width, channel)] / channel;
			}
		}

		output[o_idx] = out_val;
	}
}

#define XCOR_SMALL

template<typename T>
struct FastXCor<GPUDevice, T>{
	void operator()(
			const GPUDevice& d,
			const int n_n,
			const int n_h,
			const int n_w,
			const int n_c,
			const int n_d,
			const int s_h,
			const int s_w,

			const T* input_data,
			const T* filter_data,
			T* output_data
			){

		CudaLaunchConfig config;

#ifdef XCOR_SMALL
		config = GetCudaLaunchConfig(n_n*n_h*n_w*n_d*n_d, d);
#else
		config = GetCudaLaunchConfig(n_n*n_h*n_w*n_d*n_d*n_c, d);
#endif

        //printf("params: (n_n,n_h,n_w,n_c,n_d,s_h,s_w) = (%d,%d,%d,%d,%d,%d,%d)\n",
        //        n_n, n_h, n_w, n_c, n_d, s_h, s_w
        //        );

        //printf("config : (%dx%dx%d) (%d/%d)\n",
		//		config.virtual_thread_count,
        //        config.block_count,
        //        config.thread_per_block,
		//		config.block_count * config.thread_per_block,
		//		n_n*n_h*n_w*n_d*n_d
        //        );

#ifndef XCOR_SMALL
		cudaMemset(output_data, 0, n_n*n_h*n_w*n_d*n_d*sizeof(T));
#endif

#ifdef XCOR_SMALL
		fast_xcor_kernel_small<T> <<<config.block_count,config.thread_per_block, 0, d.stream()>>>(
#else
		fast_xcor_kernel<T> <<<config.block_count,config.thread_per_block, 0, d.stream()>>>(
#endif
                n_n,
				n_h,
				n_w,
				n_c,
				n_d,
				s_h,
				s_w,
				input_data,
				filter_data,
				output_data);

        cudaError_t cudaerr = cudaDeviceSynchronize();
        if (cudaerr != cudaSuccess)
            printf("kernel launch failed with error \"%s\".\n", cudaGetErrorString(cudaerr));
	}
};

template struct FastXCor<GPUDevice, float>;

#endif
