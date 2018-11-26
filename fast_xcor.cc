#define EIGEN_USE_THREADS

#include "fast_xcor.h"
#include "tensorflow/core/framework/common_shape_fns.h"

REGISTER_OP("FastXCor")
.Input("input: T")
.Input("filter: T")
.Output("output: T")
.Attr("delta: int")
.Attr("stride_h: int")
.Attr("stride_w: int")
.Attr("T: {float32}")
.SetShapeFn([](tensorflow::shape_inference::InferenceContext* c){
		// NxHxWxDxD
		::tensorflow::shape_inference::ShapeHandle i_feature;
		::tensorflow::shape_inference::ShapeHandle i_filter;
		::tensorflow::shape_inference::ShapeHandle input;

		::tensorflow::shape_inference::ShapeHandle output;

		TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 4, &i_feature));
		TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 4, &i_filter));
		TF_RETURN_IF_ERROR(c->Merge(i_feature, i_filter, &input));

		int delta;
		TF_RETURN_IF_ERROR(c->GetAttr("delta", &delta));

		int n_n = c->Value(c->Dim(input, 0));
		int n_h = c->Value(c->Dim(input, 1));
		int n_w = c->Value(c->Dim(input, 2));

		int n_d = (delta * 2) + 1;
		
		c->set_output(0, c->MakeShape({n_n,n_h,n_w,n_d,n_d}));
		return Status::OK();
		});

using CPUDevice = Eigen::ThreadPoolDevice;
using GPUDevice = Eigen::GpuDevice;

template<typename T>
struct FastXCor<CPUDevice, T>{
	void operator()(
			const CPUDevice& d,
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

		int d_2 = (n_d-1) / 2;


		int o_off_n, o_off_h, o_off_w, o_off_dh;
		int i_off_n, i_off_h, i_off_w;

		int o_idx, i_off, f_off;

		// for output
		int dd = n_d*n_d;
		int wdd = n_w*dd;
		int hwdd = n_h*wdd;

		// for input offsets
		int wc  = n_w*n_c;
		int hwc = n_h*wc;

		for(int i_n=0; i_n<n_n; ++i_n){
			o_off_n = i_n * hwdd;
			i_off_n = i_n * hwc;
			for(int i_h=0; i_h<n_h; ++i_h){
				o_off_h = (o_off_n) + i_h * wdd;
				i_off_h = (i_off_n) + i_h * wc;
				for(int i_w=0; i_w<n_w; ++i_w){
					o_off_w = (o_off_h) + i_w * dd;
					i_off_w = (i_off_h) + i_w * n_c;

					for(int i_dh=0; i_dh<n_d; ++i_dh){
						o_off_dh = (o_off_w) + i_dh * n_d;
						for(int i_dw=0; i_dw<n_d; ++i_dw){

							o_idx = o_off_dh + i_dw;

                            const int i_h2  = i_h + s_h*(i_dh - d_2);
                            const int i_w2  = i_w + s_w*(i_dw - d_2);

							f_off = i_off_n + \
									(i_h2) * wc + \
									(i_w2) * (n_c);

							// set to 0
							output_data[o_idx] = 0;

                            if(i_h2 < 0 || i_h2 >= n_h \
                                    || i_w2 < 0 || i_w2 >= n_w){
                                continue;
                            }

							// accumulate
							for(int i_c=0; i_c<n_c; ++i_c){
								output_data[o_idx] +=
									(input_data[i_off_w + i_c] * filter_data[f_off + i_c]) / n_c;
							}}}}}}
	}
};



#define REGISTER_CPU(T) \
	REGISTER_KERNEL_BUILDER( \
			Name("FastXCor").Device(DEVICE_CPU) \
			.TypeConstraint<T>("T"), \
			FastXCorOp<CPUDevice, T>);

REGISTER_CPU(float);
//REGISTER_CPU(double, int32);
//REGISTER_CPU(float, int64);
//REGISTER_CPU(double, int64);

#define REGISTER_GPU(T) \
	REGISTER_KERNEL_BUILDER( \
			Name("FastXCor").Device(DEVICE_GPU) \
			.TypeConstraint<T>("T"), \
			FastXCorOp<GPUDevice, T>);
REGISTER_GPU(float);
//REGISTER_GPU(float, int64);
