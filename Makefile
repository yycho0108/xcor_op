TF_CFLAGS = `python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))'`
TF_LFLAGS = `python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))'`

CC        = gcc -O2 -pthread
CXX       = g++ -g
GPUCC     = nvcc
CFLAGS    = -std=c++11 -I"$(CUDA_HOME)/include" -I"$(CUDA_HOME)/.." -DGOOGLE_CUDA=1
GPUCFLAGS = -c -gencode=arch=compute_61,code=sm_61
LFLAGS    = -pthread -shared -fPIC $(TF_LFLAGS)
GPULFLAGS = -x cu -Xcompiler -fPIC --expt-relaxed-constexpr -DNDEBUG
CGPUFLAGS = -L$(CUDA_HOME)/lib -L$(CUDA_HOME)/lib64 -lcudart -DNDEBUG

CFLAGS += -D_MWAITXINTRIN_H_INCLUDED -D_FORCE_INLINES -D__STRICT_ANSI__ -D_GLIBCXX_USE_CXX11_ABI=0


all: fast_xcor.so

#.PHONY: test

#test: fast_xcor.so test_fast_xcor.py
#	python test_fast_xcor.py

#fast_xcor.so: fast_xcor.cu.o fast_xcor.cc
#	g++ -std=c++11 -shared -o fast_xcor.so fast_xcor.cc fast_xcor.cu.o -I /usr/local/lib/python2.7/dist-packages/tensorflow/include -fPIC -L${CUDA_HOME}/lib64 -lcudart -O2 -D_GLIBCXX_USE_CXX11_ABI=0

#fast_xcor.so: fast_xcor.cu.o fast_xcor.cc
#	$(CXX) $(CFLAGS) $(TF_CFLAGS) -shared -o fast_xcor.so fast_xcor.cc fast_xcor.cu.o -fPIC $(CGPUFLAGS) $(LFLAGS) -O2
#
#fast_xcor.cu.o: fast_xcor.h fast_xcor.cu.cc
#	$(GPUCC) $(CFLAGS) $(TF_CFLAGS) $(GPUCFLAGS) -o fast_xcor.cu.o fast_xcor.cu.cc $(GPULFLAGS)

fast_xcor.cu.o: fast_xcor.h fast_xcor.cu.cc
	nvcc -std=c++11 -c -arch compute_61 -o fast_xcor.cu.o fast_xcor.cu.cc \
	$(TF_CFLAGS) -D GOOGLE_CUDA=1 -DNDEBUG -x cu -Xcompiler -fPIC --expt-relaxed-constexpr -I"$(CUDA_HOME)/include" -I"$(CUDA_HOME)/.."

fast_xcor.so: fast_xcor.cc fast_xcor.h fast_xcor.cu.o
	g++ -std=c++11 -shared -o fast_xcor.so fast_xcor.cc \
	fast_xcor.cu.o $(TF_CFLAGS) -fPIC -L$(CUDA_HOME)/lib -L$(CUDA_HOME)/lib64 -lcudart $(TF_LFLAGS)

clean:
	rm fast_xcor.cu.o fast_xcor.so
