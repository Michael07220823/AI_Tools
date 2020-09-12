import time 
import math

class Timer(object):
    def __init__(self, string):
        self._string = string

    def __enter__(self):
        self._start = time.time()

    def __exit__(self, *args):
        self.secends = time.time() - self._start
        
        if self.secends/60 <1:
            print("[INFO] %s cost time %.2f secs." % (self._string, self.secends))
        elif self.secends/60 >= 1 and self.secends/3600 < 1:
            self.minutes = math.floor(self.secends/60 )
            self.secends = self.secends - self.minutes*60
            print("[INFO] %s cost time %d mins %.2f secs." % (self._string, self.minutes ,self.secends))
        elif self.secends/3600 >= 1 :
            self.hours = math.floor(self.secends/3600)
            self.minutes = math.floor((self.secends - self.hours*3600) / 60)
            self.secends = self.secends - self.hours*3600 - self.minutes*60
            print("[INFO] %s cost time %d hours %d mins %.2f secs." % (self._string, self.hours, self.minutes ,self.secends))