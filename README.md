# XCor Op

Cross Correlation in Tensorflow, similar to the approach described in [FlowNet](https://arxiv.org/abs/1504.06852).

I initially thought it would be fast, which is why the op is called "FastXCor". Unfortunately, the current implementation is about half as fast as the implementation [here](https://github.com/sampepose/flownet2-tf/tree/master/src/ops/correlation).

Since the implementation itself is quite intuitive here, I thought it's worthwhile to just keep it.
