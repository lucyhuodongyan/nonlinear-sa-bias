#!/usr/bin/env python
# coding: utf-8



import numpy as np

class LogReg:
    def __init__(self, eps, d=1, seed = None):
        
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        else:
            self.rng = np.random.default_rng()
            
        random_vec = self.rng.normal(0, 1, d)
        norm = np.linalg.norm(random_vec)
        self.w = random_vec / norm
        self.theta_star = self.w
        
        self.eps = eps
        self.d = d

