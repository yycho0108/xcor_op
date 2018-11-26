#ifndef __FAST_XCOR_H__
#define __FAST_XCOR_H__

#include <iostream>

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/kernels/bounds_check.h"

using namespace tensorflow;

template <typename Device, typename T>
struct FastXCor{
	void operator()(
        const Device& d,
        const int n_n,
        const int n_h,
        const int n_w,
        const int n_c,
        const int n_d,
        const int s_h,
        const int s_w,

        const T* input_data,
        const T* filter_data,
        T* output_data);
};

//#if GOOGLE_CUDA
//// Partially specialize functor for GpuDevice.
//template <typename Eigen::GpuDevice, typename T>
//struct FastXCor{
//    void operator()(const Eigen::GpuDevice& d,
//            const int n_n,
//            const int n_h,
//            const int n_w,
//            const int n_c,
//            const int n_d,
//            const int s_h,
//            const int s_w,
//
//            const T* input_data,
//            const T* filter_data,
//            T* output_data);
//};
//#endif

template <typename Device, typename T>
class FastXCorOp: public OpKernel{
	private:
		int delta_;
		int stride_h_;
		int stride_w_;
	public:
		explicit FastXCorOp(OpKernelConstruction* context) : OpKernel(context){
			OP_REQUIRES_OK(context, context->GetAttr("delta", &delta_));
			OP_REQUIRES_OK(context, context->GetAttr("stride_h", &stride_h_));
			OP_REQUIRES_OK(context, context->GetAttr("stride_w", &stride_w_));
		}

		void Compute(OpKernelContext* context) override{
			const Tensor& input = context->input(0);
			const Tensor& filter = context->input(1);

			int n_n = input.shape().dim_size(0);
			int n_h = input.shape().dim_size(1);
			int n_w = input.shape().dim_size(2);
			int n_c = input.shape().dim_size(3);
			int n_d = (2 * delta_ + 1);

            //printf("Dimension : (%d,%d,%d,%d,%d)\n", n_n, n_h, n_w, n_d, n_d);
			TensorShape output_shape({n_n,n_h,n_w,n_d,n_d}); //n,h,w,d,d

			Tensor* output = nullptr;
			OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &output));

            //if(!output){
            //    fprintf(stderr, "Output Tensor Allocation Failed?\n");
            //}else{
            //    printf("Output Tensor Allocation Should Have Succeeded");
            //    int n_size = (n_n * n_h * n_w * n_d * n_d);
            //    auto output_flat = output->flat<float>();
            //    for (int i = 0; i < n_size; i++) {
            //        printf("%d ", i);
            //        output_flat(i) = 1;
            //    }
            //}

			FastXCor<Device, T>()(
					context->eigen_device<Device>(),
					n_n,
					n_h,
					n_w,
					n_c,
					n_d,
					stride_h_,
					stride_w_,
					input.flat<T>().data(),//flat<T>().data(),
					filter.flat<T>().data(),//flat<T>().data(),
					output->flat<T>().data()//flat<T>().data()
					);
		}
};
#endif
