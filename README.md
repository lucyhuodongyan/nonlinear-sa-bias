
# The Collusion of Memory and Nonlinearity in Stochastic Approximation with Constant Stepsize

This repository contains the code accompanying the paper **"The Collusion of Memory and Nonlinearity in Stochastic Approximation with Constant Stepsize"**, presented at NeurIPS 2024 (Spotlight). The paper studies the interplay between the nonlinearity of SA operator and the memory effects of the Markvoian correlated data in constant stepsize stochastic approximation and its impact on bias.

## Requirement

- Python 3: numpy, scipy, statsmodels, datetime, matplotlib, ray (for parallel processing)


## Files and Structure

- functions.py        # Core SA and bias functions
- log_convergence.py  # Script for convergence experiments
- log_clt.py          # Script for CLT analysis (histogram and QQ plot)
- plots/              # Folder for generated plot outputs


## Experiments Setup

The experiments were run on a high-performance server equipped with:
- **Two sockets of Intel(R) Xeon(R) Gold 6154 CPU @ 3.00GHz**, providing a total of 36 physical cores and 72 threads.
- **566GB of RAM** to handle memory-intensive computations.
