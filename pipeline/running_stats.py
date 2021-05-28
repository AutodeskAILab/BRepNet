"""
A class to compute mean and variance for data without having
it all loaded into memory at once.
Based on http://www.johndcook.com/standard_deviation.html
"""
import math

class RunningStats:

    def __init__(self):
        self.n = 0
        self.old_m = 0
        self.new_m = 0
        self.old_s = 0
        self.new_s = 0

        self.min_val = None
        self.max_val = None

    def clear(self):
        self.n = 0
        self.old_m = 0
        self.new_m = 0
        self.old_s = 0
        self.new_s = 0
        self.min_val = None
        self.max_val = None

    def push(self, x):
        self.n += 1

        # For debugging keep track of the min and max values
        if self.min_val == None:
            self.min_val = x
        if self.max_val == None:
            self.max_val = x
        if x < self.min_val:
            self.min_val = x
        if x > self.max_val:
            self.max_val = x

        if self.n == 1:
            self.old_m = self.new_m = x
            self.old_s = 0
        else:
            self.new_m = self.old_m + (x - self.old_m) / self.n
            self.new_s = self.old_s + (x - self.old_m) * (x - self.new_m)

            self.old_m = self.new_m
            self.old_s = self.new_s

    def mean(self):
        assert self.min_val <= self.new_m 
        assert self.new_m  <= self.max_val 
        return self.new_m if self.n else 0.0

    def variance(self):
        var2 = self.new_s / (self.n - 1)
        m = max(abs(self.max_val),  abs(self.min_val))
        assert var2 <= m*m
        return self.new_s / (self.n - 1) if self.n > 1 else 0.0

    def standard_deviation(self):
        return math.sqrt(self.variance())