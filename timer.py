import time

class Timer(object):
    def __init__(self, name='timer'):
        self.name = name
        self.start = 0
    def __enter__(self):
        self.start = time.time()
        return self
    def __exit__(self,t,v,tb):
        dt = time.time() - self.start
        print '[%s] : Took %.3f Seconds' % (self.name, dt)
