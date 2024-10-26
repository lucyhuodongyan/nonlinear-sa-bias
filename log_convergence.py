#!/usr/bin/env python
# coding: utf-8


import numpy as np
import matplotlib.pyplot as plt

import functions as fn
from logreg_class import LogReg
import datetime


eps_length = 10**7
reg_para = 0.0001

stepsize = 0.8
rr_rate = 2
stepsize_num = 3
rr_order = 2


log_reg_problem = LogReg(0.9, d=1, seed = 1)


theta_traj, thetaRR_traj, thetaDimin_traj = fn.log_reg_traj(log_reg_problem, stepsize, stepsize_num, rr_rate, rr_order, eps_length, reg_para, 0)



thetaPR = np.cumsum(theta_traj,0)/np.squeeze(np.tile((np.arange(eps_length+1)+1),(log_reg_problem.d,stepsize_num,1)).transpose(2,0,1))
thetaRR = np.cumsum(thetaRR_traj,0)/np.squeeze(np.tile((np.arange(eps_length+1)+1),(log_reg_problem.d,stepsize_num - rr_order + 1,1)).transpose(2,0,1))
thetaDimin = np.cumsum(thetaDimin_traj,0)/ np.squeeze(np.tile((np.arange(eps_length+1)+1),(log_reg_problem.d,1)).transpose())


theta_star = 1


thetaPR_mse = np.abs(thetaPR - np.squeeze(np.tile(theta_star,(eps_length+1,stepsize_num,1)).transpose(0,2,1)))
thetaRR_mse = np.abs(thetaRR - np.squeeze(np.tile(theta_star,(eps_length+1,stepsize_num - rr_order + 1,1)).transpose(0,2,1))) 
thetaDimin_mse = np.abs(thetaDimin - theta_star)


theta_traj_iid, thetaRR_traj_iid, thetaDimin_traj_iid = log_reg_sim(log_reg_problem, stepsize, stepsize_num, rr_rate, rr_order, eps_length, reg_para, 1)


thetaPR_iid = np.cumsum(theta_traj_iid,0)/np.squeeze(np.tile((np.arange(eps_length+1)+1),(log_reg_problem.d,stepsize_num,1)).transpose(2,0,1))
thetaRR_iid = np.cumsum(thetaRR_traj_iid,0)/np.squeeze(np.tile((np.arange(eps_length+1)+1),(log_reg_problem.d,stepsize_num - rr_order + 1,1)).transpose(2,0,1))


thetaPR_iid_mse = np.abs(thetaPR_iid - np.squeeze(np.tile(theta_star,(eps_length+1,stepsize_num,1)).transpose(0,2,1)))
thetaRR_iid_mse = np.abs(thetaRR_iid - np.squeeze(np.tile(theta_star,(eps_length+1,stepsize_num - rr_order + 1,1)).transpose(0,2,1))) 



plt.figure(figsize=(12, 8))

plt.loglog(thetaPR_mse[:,0],'--', linewidth=2, color='orange', label='alpha='+str(stepsize)+' Mkv PR')
plt.loglog(thetaPR_mse[:,1],'--', linewidth=2, color='royalblue', label='alpha='+str(stepsize/rr_rate)+' Mkv PR')
plt.loglog(thetaPR_mse[:,2],'--', linewidth=2, color='green', label='alpha='+str(stepsize/rr_rate**2)+' Mkv PR')

plt.loglog(thetaRR_mse[:,0],'-', color='orange', linewidth=3, label='alpha='+str(stepsize)+' Mkv RR')
plt.loglog(thetaRR_mse[:,1],'-', color='royalblue', linewidth=3, label='alpha='+str(stepsize/rr_rate)+' Mkv RR')

plt.legend(fontsize="15", loc="lower left")

plt.ylim(10**-3.5, 10**1.5)


now = datetime.datetime.now()

plt.savefig(now.strftime("%m%d_%H%M") + '_convergence.pdf', bbox_inches='tight')
#plt.show()




plt.figure(figsize=(12, 8))

plt.loglog(thetaPR_iid_mse[:,0],'-.', linewidth=2, color='gold', label='alpha='+str(stepsize)+' iid PR')
plt.loglog(thetaPR_mse[:,0],'--', linewidth=2, color='orange', label='alpha='+str(stepsize)+' Mkv PR')

plt.loglog(thetaPR_iid_mse[:,1],'-.', linewidth=2, color='skyblue', label='alpha='+str(stepsize/rr_rate)+' iid PR')
plt.loglog(thetaPR_mse[:,1],'--', linewidth=2, color='royalblue', label='alpha='+str(stepsize/rr_rate)+' Mkv PR')

plt.loglog(thetaRR_iid_mse[:,0],'-', color='gold', linewidth=2, label='iid RR')
plt.loglog(thetaRR_mse[:,0],'-', color='orange', linewidth=2, label='Mkv RR')

plt.legend(fontsize="15", loc="lower left")

plt.ylim(10**-3.5, 10**1.5)



plt.savefig(now.strftime("%m%d_%H%M") + '_convergence2.pdf', bbox_inches='tight')
#plt.show()
