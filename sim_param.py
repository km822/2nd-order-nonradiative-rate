#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from sys_param import file_path

##############################
###   Simulation options   ###
##############################

simtemp = 300 # simulation temperature in Kelvin (1eV = 11600K)
order = 2     # perturbative order

nseg     = 1      # number of segments of CF over which the convergence is assessed
nsec     = 3        # number of sections into which CFs are divided for evaluations with distinct time steps
tlim     = [0.0, 900, 3600, 8000] # Lower, middle, and upper integration limit
tstep    = [3.0, 6.0, 10.0]   # Size of time step for 2 sections
sign_sec = [1, -1, -1]  # Sign of rho0 for each section 
integrator = 'boole' # Choose from 'trapezoidal', 'simpson', 'boole', or 'romberg'
width    = 0 # Gaussian envelope width in cm-1. "0" corresponds to the unmodified Delta function

#max_nrow   = 18   # Maximum number of rows. Only for Romberg integration. 
#tol = 10**-3      # error tolerance. Only for Romberg integration. 
                  # Integration stops when the relative difference between the two 
                  # lowest diagonals are smaller than tol

# Electronic coupling cutoff factor
# Only couplings greater than (1/factor)-th of its absolute maximum will be
# included in the simulation.
# The greater the factor, the more couplings included.
coup_cutoff_1 = 100 # for rank-1 CF
coup_cutoff_2 = 100 # for rank-2 CF
coup_cutoff_3 = 100 # for rank-3 CF
coup_cutoff_4 = 100 # for rank-4 CF

nodes = 1 
ppn = 1 # 1 seems to be the best number on my mac. Let python handle parallelization
node_rank = 1 # Change this only when vib modes pairs are distributed into multiple nodes (as in BPs)

ts_format = nsec*'ts{:<.3f}_'
tm_format = (nsec-1)*'tm{:<.2f}_'

out_path = file_path \
            + '{:>d}K_'.format(simtemp) \
                + 'gwidth{:>d}cm-1_'.format(width) \
                    + ts_format.format(*tstep) \
                        + 'ti{:<.1f}_'.format(tlim[0]) \
                            + tm_format.format(*tlim[1:-1]) \
                                + 'tf{:<.2f}_coupcut{:<d}_{:<s}/'.format(tlim[-1],coup_cutoff_1,integrator)
                    