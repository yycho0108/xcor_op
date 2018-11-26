import tensorflow as tf
import os
p = os.path.dirname(os.path.abspath(__file__))
l = os.path.join(p, 'fast_xcor.so')
fast_xcor = tf.load_op_library(l).fast_xcor
