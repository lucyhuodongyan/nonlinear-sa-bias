#!/usr/bin/env python
# coding: utf-8


import datetime
import numpy as np
import matplotlib.pyplot as plt
from numpy import genfromtxt
from scipy.stats import norm

import ray
ray.init()

from logreg_class import LogReg
import functions as fn

now = datetime.datetime.now()

log_reg_problem = LogReg(0.9, d=1, seed = 1)

eps_length = 10**6
reg_para = 0.0001
stepsize = 0.8
mode = 0 # 0--Mkv; 1--iid
repeat = 1000

theta_output = fn.log_reg_sim_parallel(log_reg_problem, stepsize, eps_length, reg_para, mode, repeat)
theta_output_rescaled = (theta_output - np.mean(theta_output)) * np.sqrt(eps_length)

# Plot the QQ-plot.
fn.plot_qq(theta_output_rescaled, now, plt_save = True)
fn.save_output(theta_output, now)


# Plot the histogram.
plt.figure()
mu, std = norm.fit (theta_output_rescaled) # mean and standard deviation

benchmark = np.maximum(np.abs(np.max(theta_output_rescaled)), np.abs(np.min(theta_output_rescaled)))
counts, bins, patches = plt.hist(theta_output_rescaled,range=[-1.2 * benchmark, 1.2 * benchmark], bins = 50)

# Calculate the bin width to scale the normal density appropriately
bin_width = bins[1] - bins[0]
scale_factor = len(theta_output_rescaled) * bin_width

xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 1000)
p = norm.pdf(x, mu, std)
 
plt.plot(x, p * scale_factor, 'k', linewidth=2)
plt.savefig(now.strftime("%m%d_%H%M")+'_histogram.pdf', bbox_inches='tight')

plt.show()


ray.shutdown()











