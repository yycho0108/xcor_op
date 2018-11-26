# XCor Op

Cross Correlation in Tensorflow, similar to the approach described in [FlowNet](https://arxiv.org/abs/1504.06852).

I initially thought it would be fast, which is why the op is called "FastXCor". Unfortunately, the current implementation is about half as fast as the implementation [here](https://github.com/sampepose/flownet2-tf/tree/master/src/ops/correlation).

Since the implementation itself is quite intuitive here, I thought it's worthwhile to just keep it.

Sample Execution Output:
```
/usr/local/lib/python2.7/dist-packages/h5py/__init__.py:36: FutureWarning: Conversion of the second argument of issubdtype from `float` to `np.floating` is deprecated. In future, it will be treated as `np.float64 == np.dtype(float).type`.
  from ._conv import register_converters as _register_converters
2018-11-26 05:59:51.339091: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:964] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
2018-11-26 05:59:51.339544: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1432] Found device 0 with properties: 
name: GeForce GTX 1080 Ti major: 6 minor: 1 memoryClockRate(GHz): 1.582
pciBusID: 0000:01:00.0
totalMemory: 10.92GiB freeMemory: 10.26GiB
2018-11-26 05:59:51.339563: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1511] Adding visible gpu devices: 0
2018-11-26 05:59:51.530329: I tensorflow/core/common_runtime/gpu/gpu_device.cc:982] Device interconnect StreamExecutor with strength 1 edge matrix:
2018-11-26 05:59:51.530357: I tensorflow/core/common_runtime/gpu/gpu_device.cc:988]      0 
2018-11-26 05:59:51.530362: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1001] 0:   N 
2018-11-26 05:59:51.530496: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1115] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 9916 MB memory) -> physical GPU (device: 0, name: GeForce GTX 1080 Ti, pci bus id: 0000:01:00.0, compute capability: 6.1)
img_a
('shape', (8, 16, 24, 1024))
('(mean, max, min, sum)', (-9.41915238396068e-05, 5.555315311185846, -4.985507616018486, -296.3009139049186))
img_b
('shape', (8, 16, 24, 1024))
('(mean, max, min, sum)', (-0.0012316047995986938, 5.017306447252902, -5.035156321772521, -3874.293703032))
======================================
[cpu-mine] : Took 0.718 Seconds
[gpu-mine] : Took 0.175 Seconds
[gpu-reference] : Took 0.123 Seconds
cpu
('shape', (8, 16, 24, 11, 11))
('(mean, max, min, sum)', (-3.0418676e-05, 0.14160293, -0.15298975, -11.306987))
gpu
('shape', (8, 16, 24, 11, 11))
('(mean, max, min, sum)', (-3.0418676e-05, 0.14160293, -0.15298975, -11.306987))
gpu-true
('shape', (8, 16, 24, 121))
('(mean, max, min, sum)', (-3.0418681e-05, 0.14160295, -0.15298976, -11.306989))
[0.9999998 0.9999999 0.9999999 0.9999998]
======================================
```
