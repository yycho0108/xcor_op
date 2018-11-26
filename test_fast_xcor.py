#!/usr/bin/env python2
import tensorflow as tf
import numpy as np
from tensorflow.python.client import timeline
from timer import Timer
import sys

from tensorflow.python import debug as tf_debug

root = '/home/jamiecho/Repos/tf_flownet2'
sys.path.append(root)
from FlowNet2_src.correlation import correlation as xcor_truth

#np.random.seed(0);
fast_xcor_module = tf.load_op_library('./fast_xcor.so')
fast_xcor = fast_xcor_module.fast_x_cor



n_repeat = 10
N = 8
H = 16
W = 24
C = 1024
d = 5 # D == 2*d+1!

def stat(x):
    print(x.shape)
    s = x.mean(), x.max(), x.min(), x.sum()
    print('(mean, max, min, sum)', s)
    return s

##t = tf.placeholder(tf.float32, [B,C])
#
#num_seg = tf.constant(NS, shape=[], dtype=tf.int32)
#_t = np.random.random((B,C))
#_s_id = np.sort(np.random.randint(NS, size=B), axis=0).astype(np.int32)
#
#c_t = tf.placeholder_with_default(tf.random_normal(shape=[B,C]), shape=[B,C])
#c_s_id = tf.placeholder_with_default(_s_id, shape=[B])
#with tf.device('/cpu:0'):
#    c_o = segment_argmax_module.segment_argmax(c_t, c_s_id, num_seg=NS)


img_a_np = np.random.normal(size=[N,H,W,C])#, dtype=np.float32)
print('img_a')
stat(img_a_np)

img_b_np = np.random.normal(size=[N,H,W,C])#, dtype=np.float32)
print('img_b')
stat(img_b_np)

with tf.device('/gpu:0'):
    img_a_g = tf.placeholder(dtype=tf.float32, shape=[N,H,W,C])
    img_b_g = tf.placeholder(dtype=tf.float32, shape=[N,H,W,C])
    xcor_g = fast_xcor(img_a_g, img_b_g, d, 1, 1)
    xcor_g_gt = xcor_truth(img_a_g, img_b_g, 1, d, 1, 1, d)

with tf.device('/cpu:0'):
    img_a_c = tf.placeholder(dtype=tf.float32, shape=[N,H,W,C])
    img_b_c = tf.placeholder(dtype=tf.float32, shape=[N,H,W,C])
    xcor_c = fast_xcor(img_a_c, img_b_c, d, 1, 1)

#config = tf.ConfigProto(
#        device_count = {'GPU': 0})
config=None

with tf.Session(config=config) as sess:
    #sess = tf_debug.LocalCLIDebugWrapperSession(sess)

    for i in range(2):
        # warm-up
        xc = sess.run(xcor_c, {img_a_c:img_a_np, img_b_c:img_b_np})
        xg = sess.run(xcor_g, {img_a_g:img_a_np, img_b_g:img_b_np})

    with Timer('c'):
        xc = sess.run(xcor_c, {img_a_c:img_a_np, img_b_c:img_b_np})
    with Timer('g'):
        for _ in range(n_repeat):
            xg = sess.run(xcor_g, {img_a_g:img_a_np, img_b_g:img_b_np})
    with Timer('g-def'):
        for _ in range(n_repeat):
            xg2 = sess.run(xcor_g_gt, {img_a_g:img_a_np, img_b_g:img_b_np})

    print('cpu')
    stat(xc)

    print('gpu')
    xg_s = stat(xg)

    print('gpu-true')
    xg2_s = stat(xg2)

    print(np.divide(xg_s, xg2_s))


#g_t = tf.placeholder_with_default(tf.random_normal(shape=[B,C]), shape=[B,C])
#g_s_id = tf.placeholder_with_default(_s_id, shape=[B])
#with tf.device('/gpu:0'):
#    g_o = segment_argmax_module.segment_argmax(g_t, g_s_id, num_seg=NS)
#
#run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE, output_partition_graphs=True)
#run_metadata = tf.RunMetadata()
#with tf.Session() as sess:
#
#    with Timer('argmax-cpu'):
#        for i in range(10):
#            _c_o = sess.run(c_o,
#                    options=run_options,
#                    run_metadata=run_metadata,
#                    feed_dict = {
#                        c_t : _t,
#                        c_s_id : _s_id
#                        }
#                    )
#        trace = timeline.Timeline(step_stats = run_metadata.step_stats)
#        ctf = trace.generate_chrome_trace_format()
#        with open('cpu.json', 'w') as f:
#            f.write(ctf)
#    with Timer('argmax-gpu'):
#        for i in range(10):
#            _g_t, _g_s_id, _g_o = sess.run([g_t, g_s_id, g_o],
#                    options=run_options,
#                    run_metadata=run_metadata,
#                    feed_dict = {
#                        g_t : _t,
#                        g_s_id : _s_id
#                        }
#                    )
#
#
#        trace = timeline.Timeline(step_stats = run_metadata.step_stats)
#        ctf = trace.generate_chrome_trace_format()
#        with open('gpu.json', 'w') as f:
#            f.write(ctf)
#
#    print _c_o.idx
#    print _c_o.val
#
#    print _g_o.idx
#    print np.clip(_g_o.val, 0.0, 1.0)
