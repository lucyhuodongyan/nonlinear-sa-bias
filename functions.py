#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  9 23:47:21 2023

@author: huodongyan
"""
import numpy as np

import statsmodels.api as sm
import matplotlib.pyplot as plt

import ray


def get_rrcoeff(alpha_vec):
    
    M = len(alpha_vec)
    
    if M > 1:
        vand_mat = np.array([[alpha **np.arange(0,M)] for alpha in alpha_vec]).squeeze().transpose()
        coeff_vec = np.linalg.solve(vand_mat, np.hstack([1,np.zeros(M-1)]))
        return coeff_vec
    
    if M == 1:
        return np.array([1])

def get_idxs(eps_length):
    
    nIdx = 10**3
    Idxs = 10 ** (np.log10(eps_length)/nIdx * np.arange(nIdx))
    Idxs = np.unique(np.hstack([0,Idxs]).astype(int))
    
    return Idxs

@ray.remote
def log_reg_sim(log_problem, stepsize, eps_length, reg_para, mode):
    

    problem = log_problem
    
    if mode not in [0,1]:
        # 0 -- for Markovian, 1 -- for iid
        raise ValueError('Invalid sampling mode')
    
    alpha = stepsize
    
    eps = problem.eps
    steady_var = 1/(1-eps**2)
    w = problem.w
    d = problem.d
    
    x = np.random.multivariate_normal(np.zeros(d),steady_var * np.identity(d)) # initiate from stationary distr
    
    k = 0
    theta_vec = np.zeros(d)

    theta_mean = theta_vec
    
    while k < eps_length:
        
        y = np.random.binomial(1, 1/(1+np.exp(-w @ x)))
    
        theta_vec = theta_vec - 2 * alpha * (1 / (1+np.exp(-x @ theta_vec)) - y) * x[:, np.newaxis] - alpha * reg_para * theta_vec
 
        if mode ==0:
            x = eps * x + np.random.multivariate_normal(np.zeros(d), np.identity(d)) # Markovian data
        elif mode ==1:
            x = np.random.multivariate_normal(np.zeros(d),steady_var * np.identity(d)) # iid from stationary distr
            
        theta_mean = (theta_mean * k + theta_vec) / (k + 1)
              
        k = k + 1
        
    return np.squeeze(theta_mean)
    

def log_reg_sim_parallel(log_problem, stepsize, eps_length, reg_para, mode, repeat):

    repeat = int(repeat)

    sim_output = []
    sim_output.extend([log_reg_sim.remote(log_problem, stepsize, eps_length, reg_para, mode) for _ in range(repeat)])

    theta_output = []
    
    for i in range(repeat):
        
        current_output = ray.get(sim_output[i])
        theta_output.append(current_output)
        
            
    theta_output = np.array(theta_output)
    
    return theta_output


def log_reg_traj(log_problem, stepsize, stepsize_num, rr_rate, rr_order, eps_length, reg_para, mode):

## logic is the same as log_reg_sim. Only difference is that log_reg_traj is for the convergence/bias plot, while log_reg_sim is for CLT plot (only keep track of mean and no RR).
    

    problem = log_problem
    
    if mode not in [0,1]:
        # 0 -- for Markovian, 1 -- for iid
        raise ValueError('Invalid sampling mode')

    alpha = stepsize
    
    np.random.default_rng() #shuffle the seed
    
    # assuming geometric decay
    alpha_vec = alpha * ((1/rr_rate) ** np.arange(stepsize_num))
    # print(alpha_vec)
    rr_coeff_mat = fn.get_rrcoeff(alpha_vec[:rr_order])
    # for the stepsizes always decay geometrically, so the ratio stays the same,
    # only getting the first set coefficient suffices
    
    eps = problem.eps
    steady_var = 1/(1-eps**2)
    w = problem.w
    d = problem.d
    
    x = np.random.multivariate_normal(np.zeros(d),steady_var * np.identity(d)) # initiate from stationary distr
    
    k = 0
    theta_vec = np.ones((d, stepsize_num)) * -5
    theta_tile = np.lib.stride_tricks.sliding_window_view(theta_vec,(d,rr_order))
    thetaRR_vec = theta_tile @ rr_coeff_mat
    
    thetaDimin_vec = np.ones((d, 1)) * -5
    
    theta_traj = []
    thetaRR_traj = []
    thetaDimin_traj = []
    
    theta_traj.append(theta_vec)
    thetaRR_traj.append(thetaRR_vec)
    thetaDimin_traj.append(thetaDimin_vec)
    
    while k < eps_length:
        
        theta_traj.append(theta_vec)
        thetaRR_traj.append(thetaRR_vec)
        thetaDimin_traj.append(thetaDimin_vec)
        
        alpha_k = alpha/(k+1)** 0.5
        
        y = np.random.binomial(1, 1/(1+np.exp(-w @ x)))
    
        theta_vec = theta_vec - 2 * alpha_vec * (1 / (1+np.exp(-x @ theta_vec)) - y) * x[:, np.newaxis] - alpha_vec * reg_para * theta_vec
        theta_tile = np.lib.stride_tricks.sliding_window_view(theta_vec,(d,rr_order))
        thetaRR_vec = theta_tile @ rr_coeff_mat
        
        thetaDimin_vec = thetaDimin_vec - 2 * alpha_k * (1 / (1+np.exp(-x @ thetaDimin_vec)) - y) * x[:, np.newaxis] - alpha_k * reg_para * thetaDimin_vec
        
        if mode ==0:
            x = eps * x + np.random.multivariate_normal(np.zeros(d), np.identity(d)) # Markovian data
        elif mode ==1:
            x = np.random.multivariate_normal(np.zeros(d),steady_var * np.identity(d)) # iid from stationary distr
            
        
        k = k + 1
    
    return np.squeeze(np.array(theta_traj)), np.squeeze(np.array(thetaRR_traj)), np.squeeze(np.array(thetaDimin_traj))

def plot_qq(theta_output, now, plt_save = True):
    
    #create Q-Q plot with 45-degree line added to plot
    fig = sm.qqplot(theta_output[:], line='45', fit = 'True')
    plt.show()
            
    if plt_save is True:
        fig.savefig(now.strftime("%m%d_%H%M") + "_CLTQQ.png")


def save_output(theta_output, now):
    # now refers to the datetime, as identifier,
    # to avoide overwriting of log files
    
    np.savetxt(now.strftime("%m%d_%H%M") + "_thetaOutput.csv", theta_output, delimiter=",")

