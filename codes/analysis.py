#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  1 04:07:57 2024

@author: kenmiyazaki
"""

import numpy as np
from fun import read_freq
from sys_param import scale
from sim_param import nsec
from constants import *

import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
mpl.rcParams.update(mpl.rcParamsDefault)
mpl.rcParams['mathtext.rm'] = 'Times'
mpl.rcParams['mathtext.it'] = 'Times:italic'
mpl.rcParams['mathtext.default'] = 'it'
mpl.rcParams['mathtext.fontset'] = 'cm'

"""
DABNA-1
"""
loc = '/Users/user/Desktop/dabna-1_tpssh/rate_calculations/631g_sp_nac/300K_gwidth0cm-1_ts6.000_ts9.000_ts12.000_ti0.0_tm3000.00_tm7500.00_tf15000.00_coupcut100_boole/'
rate_file = 'rate_ts6.000_ts9.000_ts12.000_ti0.0_tm3000.0_tm7500.0_tf15000.0_ccut100_boole.out'
freq_file_loc = '/Users/user/Desktop/dabna-1_tpssh/rate_calculations/631g_sp_nac/'
ffreq_file = 's1_freq_tpssh_631gd_qchem'
ifreq_file = 't1_freq_tpssh_631gd_qchem'


"""
A6AP-Cz
"""
#loc = '/Users/user/Desktop/a6ap-cz/rate_calculations/no_pcm_flat/631gd_sp_nac/displaced_geo_soc/300K_gwidth0cm-1_ts3.000_ts6.000_ts10.000_ti0.0_tm900.00_tm3600.00_tf8000.00_coupcut100_boole/'
# freq_file_loc = '/Users/user/Desktop/a6ap-cz/rate_calculations/no_pcm_flat/631gd_sp_nac/flat_geo_soc/'
# ffreq_file    = 's1_freq_b3lyp_631gd_qchem'
# ifreq_file    = 't1_freq_b3lyp_631gd_qchem'
# rate_file_loc = freq_file_loc + '300K_ts3.0_ts6.0_ts10.0_ti0.0_tm900_tm3600_tf8000_coupcut100_boole_soc_at_s1/'
# rate_file     = 'rate_ts3.000_ts6.000_ts10.000_ti0.0_tm900.0_tm3600.0_tf8000.0_ccut100_boole.out'


"""
List of integration output files
"""
intgl0 = ['integral_nQ.out', 'integral_QnQ.out']
intgl1 = ['integral_nP.out', 'integral_QnP.out', 'integral_QnPQ.out']
intgl2 = ['integral_PnP.out', 'integral_PQnP.out', 'integral_PnPQ.out', 'integral_PQnPQ.out']


"""
Other input parameters
"""
nintmed = 1 # number of intermediate states
rate_sign_factor = 1 # Sign of overall rate. Either 1 or -1
contrib_cutoff = 0.20 # Cutoff ratio of rate contributions. This number x max rate or below will be excluded from the plot


"""
Read output files
"""
with open(rate_file_loc + 'integral_nQ.out', 'r') as f:
    idx_rate_nQ = []
    nentry = int(f.readline().split()[-1])
    for r in range(nentry):
        x = f.readline().split()
        idx_rate_nQ.append([int(x[0]), rate_sign_factor * float(x[1])])

with open(rate_file_loc + 'integral_QnQ.out', 'r') as f:
    pair_QnQ = []
    rate_QnQ = []
    nentry = int(f.readline().split()[-1])
    for r in range(nentry):
        x = f.readline().split()
        pair_QnQ.append([int(x[0]), int(x[1])])
        rate_QnQ.append(rate_sign_factor * float(x[2]))

# with open(rate_file_loc + 'integral_nP.out', 'r') as f:
#     idx_rate_nP = []
#     for m1 in range(nintmed):
#         idx_rate_nP_tmp = []
#         nentry = int(f.readline().split()[-1])
#         for r in range(nentry):
#             x = f.readline().split()
#             idx_rate_nP_tmp.append([int(x[0]), rate_sign_factor * float(x[1])])
#         idx_rate_nP.append(idx_rate_nP_tmp)

# with open(rate_file_loc + 'integral_nPQ.out', 'r') as f:
#     pair_nPQ = []
#     rate_nPQ = []
#     for m1 in range(nintmed):
#         pair_nPQ_tmp, rate_nPQ_tmp = [], []
#         nentry = int(f.readline().split()[-1])
#         for r in range(nentry):
#             x = f.readline().split()
#             pair_nPQ_tmp.append([int(x[0]), int(x[1])])
#             rate_nPQ_tmp.append(rate_sign_factor * float(x[2]))
#         pair_nPQ.append(pair_nPQ_tmp)
#         rate_nPQ.append(rate_nPQ_tmp)

# with open(rate_file_loc + 'integral_QnP.out', 'r') as f:
#     pair_QnP = []
#     rate_QnP = []
#     for m1 in range(nintmed):
#         pair_QnP_tmp = []
#         rate_QnP_tmp = []
#         nentry = int(f.readline().split()[-1])
#         for r in range(nentry):
#             x = f.readline().split()
#             pair_QnP_tmp.append([int(x[0]), int(x[1])])
#             rate_QnP_tmp.append(rate_sign_factor * float(x[2]))
#         pair_QnP.append(pair_QnP_tmp)
#         rate_QnP.append(rate_QnP_tmp)

# with open(rate_file_loc + 'integral_QnPQ.out', 'r') as f:
#     rate_QnPQ, triad_QnPQ = [], []
#     for m1 in range(nintmed):
#         triad_QnPQ_tmp = []
#         rate_QnPQ_tmp = []
#         nentry = int(f.readline().split()[-1])
#         for r in range(nentry):
#             x = f.readline().split()
#             triad_QnPQ_tmp.append([int(x[0]), int(x[1]), int(x[2])])
#             rate_QnPQ_tmp.append(rate_sign_factor * float(x[3]))
#         triad_QnPQ.append(triad_QnPQ_tmp)
#         rate_QnPQ.append(rate_QnPQ_tmp)

with open(rate_file_loc + 'integral_PnP.out', 'r') as f:
    pair_PnP, rate_PnP = [], []
    for m1 in range(nintmed):
        rate_PnP_tmp2, pair_PnP_tmp2 = [], []
        for m2 in range(nintmed):
            rate_PnP_tmp, pair_PnP_tmp = [], []
            nentry = int(f.readline().split()[-1])
            for r in range(nentry):
                x = f.readline().split()
                pair_PnP_tmp.append([int(x[0]), int(x[1])])
                rate_PnP_tmp.append(rate_sign_factor * float(x[2]))
            pair_PnP_tmp2.append(pair_PnP_tmp)
            rate_PnP_tmp2.append(rate_PnP_tmp)
        pair_PnP.append(pair_PnP_tmp2)
        rate_PnP.append(rate_PnP_tmp2)

with open(rate_file_loc + 'integral_PQnP.out', 'r') as f:
    rate_PQnP, triad_PQnP = [], []
    for m1 in range(nintmed):
        rate_PQnP_tmp2, triad_PQnP_tmp2 = [], []
        for m2 in range(nintmed):
            rate_PQnP_tmp, triad_PQnP_tmp = [], []
            nentry = int(f.readline().split()[-1])
            for r in range(nentry):
                x = f.readline().split()
                triad_PQnP_tmp.append([int(x[0]), int(x[1]), int(x[2])])
                rate_PQnP_tmp.append(rate_sign_factor * float(x[3]))
            triad_PQnP_tmp2.append(triad_PQnP_tmp)
            rate_PQnP_tmp2.append(rate_PQnP_tmp)
        triad_PQnP.append(triad_PQnP_tmp2)
        rate_PQnP.append(rate_PQnP_tmp2)

with open(rate_file_loc + 'integral_PnPQ.out', 'r') as f:
    rate_PnPQ, triad_PnPQ = [], []
    for m1 in range(nintmed):
        rate_PnPQ_tmp2, triad_PnPQ_tmp2 = [], []
        for m2 in range(nintmed):
            rate_PnPQ_tmp, triad_PnPQ_tmp = [], []
            nentry = int(f.readline().split()[-1])
            for r in range(nentry):
                x = f.readline().split()
                triad_PnPQ_tmp.append([int(x[0]), int(x[1]), int(x[2])])
                rate_PnPQ_tmp.append(rate_sign_factor * float(x[3]))
            triad_PnPQ_tmp2.append(triad_PnPQ_tmp)
            rate_PnPQ_tmp2.append(rate_PnPQ_tmp)
        triad_PnPQ.append(triad_PnPQ_tmp2)
        rate_PnPQ.append(rate_PnPQ_tmp2)

with open(rate_file_loc + 'integral_PQnPQ.out', 'r') as f:
    rate_PQnPQ, tetrad_PQnPQ = [], []
    for m1 in range(nintmed):
        rate_PQnPQ_tmp2, tetrad_PQnPQ_tmp2 = [], []
        for m2 in range(nintmed):
            rate_PQnPQ_tmp, tetrad_PQnPQ_tmp = [], []
            nentry = int(f.readline().split()[-1])
            for r in range(nentry):
                x = f.readline().split()
                tetrad_PQnPQ_tmp.append([int(x[0]), int(x[1]), int(x[2]), int(x[3])])
                rate_PQnPQ_tmp.append(rate_sign_factor * float(x[4]))
            tetrad_PQnPQ_tmp2.append(tetrad_PQnPQ_tmp)
            rate_PQnPQ_tmp2.append(rate_PQnPQ_tmp)
        tetrad_PQnPQ.append(tetrad_PQnPQ_tmp2)
        rate_PQnPQ.append(rate_PQnPQ_tmp2)


### Read normal mode frequencies
wno_f, redmas_f, vec_f, Nimf_f, ffreq_package = read_freq(freq_file_loc, ffreq_file)
wno_i, redmas_i, vec_i, Nimf_i, ifreq_package = read_freq(freq_file_loc, ifreq_file)
wno_f = (wno_f * scale)[6:]
wno_i = (wno_i * scale)[6:]

#%%
"""""""""""""""""
      Plots
"""""""""""""""""

'''
CF_nQ
'''
# List of unique indices
k_unique = [idx_rate_nQ[i][0] for i in range(len(idx_rate_nQ))]
# Array of rates
rates = np.array([idx_rate_nQ[i][1] for i in range(len(idx_rate_nQ))])
max_rate = np.max(abs(rates))

# Extract only significant contributors above 'thres'
thres = max_rate * 0.32 #contrib_cutoff
key_k = []
for k in range(len(k_unique)):
    if abs(rates[k]) > thres:
        if k_unique[k] not in key_k:
            key_k.append(k_unique[k])

key_rates = np.zeros(len(key_k))
for i, k in enumerate(key_k):
    key_rates[i] = rates[k_unique.index(k)]
# Normalized absolute rates
key_rates = key_rates/max_rate

# Vector plot
fig, ax = plt.subplots(nrows=1, ncols=1, sharex=False, sharey=False, figsize=(8, 8))
plt_nQ = ax.imshow(np.expand_dims(key_rates, axis=1), cmap='seismic', vmin=-1.0, vmax=1.0)
ax.invert_yaxis()
ax.set_ylabel('$S_{1}$ normal mode frequency [cm$^{-1}$]', fontsize=24, **{'fontname':'Times'})
ax.set_yticks(np.arange(0,len(key_k),1), np.round(wno_f[key_k]).astype(int), fontsize=24, **{'fontname':'Times'})
ax.set_xlabel('')
ax.set_xticks([])

# # Color bar settings
# divider = make_axes_locatable(ax)
# cbar_ax = fig.add_axes([0.92, 0.110, 0.05, 0.77])
# cbar2 = fig.colorbar(plt_nQ, cax=cbar_ax)
# #cbar2.ax.locator_params(nbins=6) # Set to a maximum of 5 ticks
# cbar2.ax.set_yticklabels(['{:.2f}'.format(x) for x in np.arange(-1.0,
#                         1.0 + 0.25, 0.25)], fontsize=24, **{'fontname':'Times'})

# Save plot as a file
plt.savefig(rate_file_loc + 'modes_nQ.png', dpi=600)
plt.show()

#%%
'''
CF_QnQ
'''
# List of unique indices
k_indices = [pair_QnQ[i][0] for i in range(len(pair_QnQ))]
k_unique = []
for k in k_indices:
    if k not in k_unique:
        k_unique.append(k) # list of unique k indices

# Array of rates
rates = np.zeros((len(k_unique), len(k_unique)))
for i, pair in enumerate(pair_QnQ):
    rates[k_unique.index(pair[0]), k_unique.index(pair[1])] = rate_QnQ[i]
# Normalized absolute rates
max_rate = np.max(abs(rates))

# Extract only significant contributors above 'thres'
thres = max_rate * 0.07 #contrib_cutoff
key_k1, key_k2 = [], []
for k1 in range(len(k_unique)):
    for k2 in range(1, len(k_unique)):
        if abs(rates[k1, k2]) > thres:
            if k_unique[k1] not in key_k1:
                key_k1.append(k_unique[k1])
            if k_unique[k2] not in key_k2:
                    key_k2.append(k_unique[k2])
key_k2 = sorted(key_k2)

key_rates = np.zeros((len(key_k1), len(key_k2)))
for i, k1 in enumerate(key_k1):
    for j, k2 in enumerate(key_k2):
        key_rates[i, j] = rates[k_unique.index(k1),k_unique.index(k2)]
# Normalized absolute rates
key_rates = key_rates/max_rate

# Matrix plot
fig, ax = plt.subplots(nrows=1, ncols=1, sharex=False, sharey=False, figsize=(8, 8))
plt_QnQ = ax.imshow(key_rates, cmap='seismic', vmin=-1.0, vmax=1.0)
ax.invert_yaxis()
#ax.set_ylabel('$S_{1}$ normal mode frequency [cm$^{-1}$]', fontsize=24, **{'fontname':'Times'})
ax.set_yticks(np.arange(0, len(key_k1), 1), np.round(wno_f[key_k1]).astype(int), fontsize=24, **{'fontname':'Times'})
ax.set_xlabel('$S_{1}$ normal mode frequency [cm$^{-1}$]', fontsize=24, **{'fontname':'Times'})
ax.set_xticks(np.arange(0, len(key_k2), 1), np.round(wno_f[key_k2]).astype(int), fontsize=24, rotation=0, **{'fontname':'Times'})

# Color bar settings
divider = make_axes_locatable(ax)
cbar_ax = fig.add_axes([0.92, 0.110, 0.05, 0.77])
cbar2 = fig.colorbar(plt_QnQ, cax=cbar_ax)
#cbar2.ax.locator_params(nbins=6) # Set to a maximum of 5 ticks
cbar2.ax.set_yticklabels(['{:.2f}'.format(x) for x in np.arange(-1.0,
                        1.0 + 0.25, 0.25)], fontsize=24, **{'fontname':'Times'})


# Save plot as a file
plt.savefig(rate_file_loc + 'modes_QnQ.png', dpi=600, bbox_inches='tight')
plt.show()

#%%
'''
CF_nP
'''
# Need to loop over intermediate states
for m1 in range(nintmed):
    idx_rate_nP_m1 = idx_rate_nP[m1]
    
    # List of unique indices
    k_unique = [idx_rate_nP_m1[i][0] for i in range(len(idx_rate_nP_m1))]
    # Array of rates
    rates = np.array([idx_rate_nP_m1[i][1] for i in range(len(idx_rate_nP_m1))])
    max_rate = max(abs(rates))
    # Extract only significant contributors above 'thres'
    thres = max_rate * contrib_cutoff
    key_k, key_rates = [], []
    for i in range(len(rates)):
        if abs(rates[i]) > thres:
            key_k.append(k_unique[i])
            key_rates.append(rates[i])  
    # Normalized absolute rates
    key_rates = key_rates/max_rate
    
    # Vector plot
    fig, ax = plt.subplots(nrows=1, ncols=1, sharex=False, sharey=False, figsize=(8, 8))
    plt_nP = ax.imshow(np.expand_dims(key_rates, axis=1), cmap='seismic', vmin=-1.0, vmax=1.0)
    ax.invert_yaxis()
    ax.set_ylabel('Normal mode index', fontsize=20, **{'fontname':'Times'})
    ax.set_yticks(np.arange(0,len(key_k),1), np.array(key_k)+1, fontsize=15, **{'fontname':'Times'})
    ax.set_xlabel('')
    ax.set_xticks([])
    
    # # Color bar settings
    # divider = make_axes_locatable(ax)
    # cbar_ax = fig.add_axes([0.62, 0.285, 0.03, 0.43])
    # cbar2 = fig.colorbar(plt_nQ, cax=cbar_ax)
    # cbar2.ax.set_yticklabels(['{:.1f}'.format(x) for x in np.arange(0,
    #                         (1) + 0.2, 0.2)], fontsize=15, **{'fontname':'Times'})
    
    # Save plot as a file
    plt.savefig(rate_file_loc + 'modes_nP_m{:d}.png'.format(m1), dpi=600)
    plt.show()
    
#%%
'''
CF_nPQ

For A6AP-Cz RISC, set cutoff ratio to 0.4  and sign factor for overall rate to -1
'''
# Need to loop over intermediate states
for m1 in range(nintmed):
    rate_nPQ_m1 = rate_nPQ[m1]
    pair_nPQ_m1 = pair_nPQ[m1]

    # List of unique indices
    k_indices = [pair_nPQ_m1[i][0] for i in range(len(pair_nPQ_m1))]
    k_unique = []
    for k in k_indices:
        if k not in k_unique:
            k_unique.append(k) # list of unique k indices
    l_indices = [pair_nPQ_m1[i][1] for i in range(len(pair_nPQ_m1))]
    l_unique = []
    for l in l_indices:
        if l not in l_unique:
            l_unique.append(l) # list of unique l indices
            
    # Array of rates
    rates = np.zeros((len(k_unique), len(l_unique)))
    for i, pair in enumerate(pair_nPQ_m1):
        rates[k_unique.index(pair[0]), l_unique.index(pair[1])] = rate_nPQ_m1[i]
    max_rate = np.max(abs(rates))
    
    # Extract only significant contributors above 'thres'
    thres = max_rate * contrib_cutoff
    key_k, key_l = [], []
    for k in range(len(k_unique)):
        for l in range(len(l_unique)):
            if abs(rates[k,l]) > thres:
                if k_unique[k] not in key_k:
                    key_k.append(k_unique[k])
                if l_unique[l] not in key_l:
                    key_l.append(l_unique[l])
    key_l = sorted(key_l)

    key_rates = np.zeros((len(key_k), len(key_l)))
    for i, k in enumerate(key_k):
        for j, l in enumerate(key_l):
            key_rates[i,j] = rates[k_unique.index(k),l_unique.index(l)]
    # Normalized absolute rates
    key_rates = key_rates/max_rate
    
    # Matrix plot
    fig, ax = plt.subplots(nrows=1, ncols=1, sharex=False, sharey=False, figsize=(8, 7))
    plt_nPQ = ax.imshow(key_rates, cmap='seismic', vmin=-1.0, vmax=1.0)
    ax.invert_yaxis()
    ax.set_ylabel('$T_{1}$ normal mode frequency [cm$^{-1}$]', fontsize=24, **{'fontname':'Times'})
    ax.set_yticks(np.arange(0,len(key_k),1), np.round(wno_i[key_k]).astype(int), fontsize=24, **{'fontname':'Times'})
    ax.set_xlabel('$S_{1}$ normal mode frequency [cm$^{-1}$]', fontsize=24, **{'fontname':'Times'})
    ax.set_xticks(np.arange(0,len(key_l),1), np.round(wno_f[key_l]).astype(int), fontsize=24, rotation=0, **{'fontname':'Times'})
    
    # Color bar settings
    divider = make_axes_locatable(ax)
    cbar_ax = fig.add_axes([0.93, 0.145, 0.05, 0.70])
    cbar2 = fig.colorbar(plt_nPQ, cax=cbar_ax)
    cbar2.ax.set_yticklabels(['{:.2f}'.format(x) for x in np.arange(-1.0,
                            (1.0) + 0.25, 0.25)], fontsize=20, **{'fontname':'Times'})
    
    # Save plot as a file
    plt.savefig(rate_file_loc + 'modes_nPQ_m{:d}.png'.format(m1+1), bbox_inches="tight", dpi=600)
    plt.show()

#%%
'''
CF_QnP

For A6AP-Cz RISC, set cutoff ratio to 0.35 and sign factor for overall rate to -1
'''
# Need to loop over intermediate states
for m1 in range(nintmed):
    rate_QnP_m1 = rate_QnP[m1]
    pair_QnP_m1 = pair_QnP[m1]

    # List of unique indices
    k_indices = [pair_QnP_m1[i][0] for i in range(len(pair_QnP_m1))]
    k_unique = []
    for k in k_indices:
        if k not in k_unique:
            k_unique.append(k) # list of unique k indices
    l_indices = [pair_QnP_m1[i][1] for i in range(len(pair_QnP_m1))]
    l_unique = []
    for l in l_indices:
        if l not in l_unique:
            l_unique.append(l) # list of unique l indices
            
    # Array of rates
    rates = np.zeros((len(k_unique), len(l_unique)))
    for i, pair in enumerate(pair_QnP_m1):
        rates[k_unique.index(pair[0]), l_unique.index(pair[1])] = rate_QnP_m1[i]
    max_rate = np.max(abs(rates))
    
    # Extract only significant contributors above 'thres'
    thres = max_rate * contrib_cutoff
    key_k, key_l = [], []
    for k in range(len(k_unique)):
        for l in range(len(l_unique)):
            if abs(rates[k,l]) > thres:
                if k_unique[k] not in key_k:
                    key_k.append(k_unique[k])
                if l_unique[l] not in key_l:
                    key_l.append(l_unique[l])
    # sort index l (k is already in an ascending order)
    key_l = sorted(key_l)

    key_rates = np.zeros((len(key_k), len(key_l)))
    for i, k in enumerate(key_k):
        for j, l in enumerate(key_l):
            key_rates[i,j] = rates[k_unique.index(k),l_unique.index(l)]
    # Normalized absolute rates
    key_rates = key_rates/max_rate
    
    # Matrix plot
    fig, ax = plt.subplots(nrows=1, ncols=1, sharex=False, sharey=False, figsize=(8, 8))
    plt_QnP = ax.imshow(key_rates, cmap='seismic', vmin=-1.0, vmax=1.0)
    ax.invert_yaxis()
    ax.set_ylabel('$T_{1}$ normal mode index', fontsize=24, **{'fontname':'Times'})
    ax.set_yticks(np.arange(0,len(key_k),1), np.round(wno_i[key_k]).astype(int), fontsize=24, **{'fontname':'Times'})
    ax.set_xlabel('$T_{1}$ normal mode index', fontsize=24, **{'fontname':'Times'})
    ax.set_xticks(np.arange(0,len(key_l),1), np.round(wno_i[key_l]).astype(int), fontsize=24, **{'fontname':'Times'}, rotation=0)
    
    # Color bar settings
    divider = make_axes_locatable(ax)
    cbar_ax = fig.add_axes([0.93, 0.110, 0.05, 0.77])
    cbar2 = fig.colorbar(plt_nPQ, cax=cbar_ax)
    cbar2.ax.set_yticklabels(['{:.2f}'.format(x) for x in np.arange(-1.0,
                            (1.0) + 0.25, 0.25)], fontsize=20, **{'fontname':'Times'})
    
    # Save plot as a file
    plt.savefig(rate_file_loc + 'modes_QnP_m{:d}.png'.format(m1+1), bbox_inches='tight', dpi=600)
    plt.show()

#%%
'''
CF_PnP

For DABNA-1 RISC, set cutoff ratio to 0.15 and sign factor for overall rate to 1
For A6AP-Cz RISC, set cutoff ratio to 0.20
'''
# Need to loop over intermediate states
for m1 in range(nintmed):
    for m2 in range(nintmed):
        rate_PnP_mm = rate_PnP[m1][m2]
        pair_PnP_mm = pair_PnP[m1][m2]
    
        # List of unique indices
        k_indices = [pair_PnP_mm[i][0] for i in range(len(pair_PnP_mm))]
        k_unique = []
        for k in k_indices:
            if k not in k_unique:
                k_unique.append(k) # list of unique k indices
        l_indices = [pair_PnP_mm[i][1] for i in range(len(pair_PnP_mm))]
        l_unique = []
        for l in l_indices:
            if l not in l_unique:
                l_unique.append(l) # list of unique l indices
                
        # Array of rates
        rates = np.zeros((len(k_unique), len(l_unique)))
        for i, pair in enumerate(pair_PnP_mm):
            rates[k_unique.index(pair[0]), l_unique.index(pair[1])] = rate_PnP_mm[i]
        max_rate = np.max(abs(rates))
        
        # Extract only significant contributors above 'thres'
        thres = max_rate * contrib_cutoff
        key_k, key_l = [], []
        for k in range(len(k_unique)):
            for l in range(len(l_unique)):
                if abs(rates[k,l]) > thres:
                    if k_unique[k] not in key_k:
                        key_k.append(k_unique[k])
                    if l_unique[l] not in key_l:
                        key_l.append(l_unique[l])
        # sort index l (k is already in an ascending order)
        key_l = sorted(key_l)
    
        key_rates = np.zeros((len(key_k), len(key_l)))
        for i, k in enumerate(key_k):
            for j, l in enumerate(key_l):
                key_rates[i,j] = rates[k_unique.index(k),l_unique.index(l)]
        # Normalized absolute rates
        key_rates = key_rates/max_rate
        
        # Matrix plot
        fig, ax = plt.subplots(nrows=1, ncols=1, sharex=False, sharey=False, figsize=(8, 8))
        plt_PnP = ax.imshow(key_rates, cmap='seismic', vmin=-1.0, vmax=1.0)
        ax.invert_yaxis()
        ax.set_ylabel('$T_{1}$ normal mode frequency [cm$^{-1}$]', fontsize=24, **{'fontname':'Times'})
        ax.set_yticks(np.arange(0, len(key_k), 1), np.round(wno_i[key_k]).astype(int), fontsize=24, **{'fontname':'Times'})
        ax.set_xlabel('$T_{1}$ normal mode frequency [cm$^{-1}$]', fontsize=24, **{'fontname':'Times'})
        ax.set_xticks(np.arange(0, len(key_l), 1), np.round(wno_i[key_k]).astype(int), fontsize=24, **{'fontname':'Times'}, rotation=0)
        
        # Color bar settings
        divider = make_axes_locatable(ax)
        cbar_ax = fig.add_axes([0.92, 0.110, 0.05, 0.77])
        cbar2 = fig.colorbar(plt_PnP, cax=cbar_ax)
        #cbar2.ax.locator_params(nbins=6) # Set to a maximum of 5 ticks
        cbar2.ax.set_yticklabels(['{:.2f}'.format(x) for x in np.arange(-1.0,
                                1.0 + 0.25, 0.25)], fontsize=24, **{'fontname':'Times'})
        
        # Save plot as a file
        plt.savefig(rate_file_loc + 'modes_PnP_m{:d}m{:d}.png'.format(m1+1, m2+1), bbox_inches='tight', dpi=600)
        plt.show()

#%%
'''
CF_QnPQ
'''
# Need to loop over intermediate states
for m1 in range(nintmed):
    rate_QnPQ_mm = rate_QnPQ[m1]
    triad_QnPQ_mm = triad_QnPQ[m1]

    # List of unique indices
    k_indices = [triad_QnPQ_mm[i][0] for i in range(len(triad_QnPQ_mm))]
    k_unique = []
    for k in k_indices:
        if k not in k_unique:
            k_unique.append(k) # list of unique k indices
    l_indices = [triad_QnPQ_mm[i][1] for i in range(len(triad_QnPQ_mm))]
    l_unique = []
    for l in l_indices:
        if l not in l_unique:
            l_unique.append(l) # list of unique l indices
    m_indices = [triad_QnPQ_mm[i][2] for i in range(len(triad_QnPQ_mm))]
    m_unique = []
    for m in m_indices:
        if m not in m_unique:
            m_unique.append(m) # list of unique m indices
            
    # Array of rates
    rates = np.zeros((len(k_unique), len(l_unique), len(m_unique)))
    for i, triad in enumerate(triad_QnPQ_mm):
        rates[k_unique.index(triad[0]), l_unique.index(triad[1]), m_unique.index(triad[2])] = rate_QnPQ_mm[i]
    max_rate = np.max(abs(rates))
    print('m={:d} | max. rate = {:e}'.format(m1, max_rate))
    
    # Extract only significant contributors above 'thres'
    thres = max_rate * contrib_cutoff
    key_k, key_l, key_m = [], [], []
    for k in range(len(k_unique)):
        for l in range(len(l_unique)):
            for m in range(len(m_unique)):
                if abs(rates[k,l, m]) > thres:
                    if k_unique[k] not in key_k:
                        key_k.append(k_unique[k])
                    if l_unique[l] not in key_l:
                        key_l.append(l_unique[l])
                    if m_unique[m] not in key_m:
                        key_m.append(m_unique[m])
    # sort indices (k is already in an ascending order)
    key_l = sorted(key_l)
    key_m = sorted(key_m)

    key_rates = np.zeros((len(key_k), len(key_l), len(key_m)))
    for a, k in enumerate(key_k):
        for b, l in enumerate(key_l):
            for c, m in enumerate(key_m):
                key_rates[a,b,c] = rates[k_unique.index(k), l_unique.index(l), m_unique.index(m)]
    # Normalized absolute rates
    key_rates = key_rates/max_rate
      
    # Matrix plot
    if m1 == 0:
        fig, ax = plt.subplots(nrows=1, ncols=len(key_k), sharex=True, sharey=True, figsize=(2.6, 10.4))
        for kk in range(len(key_k)):
            plt_QnPQ = ax[kk].imshow(key_rates[kk], cmap='seismic', vmin=-1.0, vmax=1.0)
            ax[kk].invert_yaxis()
            ax[kk].set_xticks(np.arange(0,len(key_m),1), np.array(key_m)+1, fontsize=22, **{'fontname':'Times'}, rotation=45)
            ax[kk].set_title(r'$k={:d}$'.format(key_k[kk]+1), fontsize=25, **{'fontname':'Times'})
            
        #ax[1].set_xlabel('Normal mode index ($m$)', fontsize=20, **{'fontname':'Times'})
        ax[0].set_ylabel('Normal mode index ($l$)', fontsize=25, **{'fontname':'Times'})
        ax[0].set_yticks(np.arange(0,len(key_l),1), np.array(key_l)+1, fontsize=22, **{'fontname':'Times'})
        fig.text(0.50, 0.01, 'Normal mode\nindex ($m$)', ha='center', fontsize=25, **{'fontname':'Times'})
        plt.subplots_adjust(wspace=0.0)
    elif m1 == 1:
        fig, ax = plt.subplots(nrows=len(key_k), ncols=1, sharex=True, sharey=True, figsize=(8, 8))
        for kk in range(len(key_k)):
            plt_QnPQ = ax[kk].imshow(key_rates[kk], cmap='seismic', vmin=-1.0, vmax=1.0)
            ax[kk].invert_yaxis()
            ax[kk].set_yticks(np.arange(0,len(key_l),1), np.array(key_l)+1, fontsize=20, **{'fontname':'Times'})
            ax[kk].set_title(r'$k={:d}$'.format(key_k[kk]+1), fontsize=20, **{'fontname':'Times'})
            
        ax[1].set_ylabel('Normal mode index ($l$)', fontsize=20, **{'fontname':'Times'})
        ax[-1].set_xlabel('Normal mode index ($m$)', fontsize=20, **{'fontname':'Times'})
        ax[-1].set_xticks(np.arange(0,len(key_m),1), np.array(key_m)+1, fontsize=20, **{'fontname':'Times'}, rotation=45)
        plt.subplots_adjust(hspace=0.3)

    # Save plot as a file
    plt.savefig(rate_file_loc + 'modes_QnPQ_m{:d}.png'.format(m1+1), dpi=600)
    plt.show()

#%%
'''
CF_PQnP
'''
# Need to loop over intermediate states
for m1 in range(nintmed):
    for m2 in range(nintmed):
        rate_PQnP_mm = rate_PQnP[m1][m2]
        triad_PQnP_mm = triad_PQnP[m1][m2]
    
        # List of unique indices
        k_indices = [triad_PQnP_mm[i][0] for i in range(len(triad_PQnP_mm))]
        k_unique = []
        for k in k_indices:
            if k not in k_unique:
                k_unique.append(k) # list of unique k indices
        l_indices = [triad_PQnP_mm[i][1] for i in range(len(triad_PQnP_mm))]
        l_unique = []
        for l in l_indices:
            if l not in l_unique:
                l_unique.append(l) # list of unique l indices
        m_indices = [triad_PQnP_mm[i][2] for i in range(len(triad_PQnP_mm))]
        m_unique = []
        for m in m_indices:
            if m not in m_unique:
                m_unique.append(m) # list of unique m indices
                
        # Array of rates
        rates = np.zeros((len(k_unique), len(l_unique), len(m_unique)))
        for i, triad in enumerate(triad_PQnP_mm):
            rates[k_unique.index(triad[0]), l_unique.index(triad[1]), m_unique.index(triad[2])] = rate_PQnP_mm[i]
        max_rate = np.max(abs(rates))
        
        # Extract only significant contributors above 'thres'
        thres = max_rate * contrib_cutoff
        key_k, key_l, key_m = [], [], []
        for k in range(len(k_unique)):
            for l in range(len(l_unique)):
                for m in range(len(m_unique)):
                    if abs(rates[k,l, m]) > thres:
                        if k_unique[k] not in key_k:
                            key_k.append(k_unique[k])
                        if l_unique[l] not in key_l:
                            key_l.append(l_unique[l])
                        if m_unique[m] not in key_m:
                            key_m.append(m_unique[m])
        # sort indices (k is already in an ascending order)
        key_l = sorted(key_l)
        key_m = sorted(key_m)
    
        key_rates = np.zeros((len(key_k), len(key_l), len(key_m)))
        for a, k in enumerate(key_k):
            for b, l in enumerate(key_l):
                for c, m in enumerate(key_m):
                    key_rates[a,b,c] = rates[k_unique.index(k), l_unique.index(l), m_unique.index(m)]
        # Normalized absolute rates
        key_rates = key_rates/max_rate
          
        # Matrix plot
        if m1 == 0 and m2 == 0:
            fig, ax = plt.subplots(nrows=1, ncols=len(key_k), sharex=True, sharey=True, figsize=(7, 8))
            for kk in range(len(key_k)):
                plt_PQnP = ax[kk].imshow(key_rates[kk], cmap='seismic', vmin=-1.0, vmax=1.0)
                ax[kk].invert_yaxis()
                ax[kk].set_xticks(np.arange(0,len(key_m),1), np.array(key_m)+1, fontsize=20, **{'fontname':'Times'}, rotation=45)
                ax[kk].set_title(r'$k={:d}$'.format(key_k[kk]), fontsize=20, **{'fontname':'Times'})
                
            #ax.set_xlabel('Normal mode index ($m$)', fontsize=20, **{'fontname':'Times'})
            ax[0].set_ylabel('Normal mode index ($l$)', fontsize=20, **{'fontname':'Times'})
            ax[0].set_yticks(np.arange(0,len(key_l),1), np.array(key_l)+1, fontsize=20, **{'fontname':'Times'})
            fig.text(0.50, 0.02, 'Normal mode index ($m$)', ha='center', fontsize=20, **{'fontname':'Times'})
            plt.subplots_adjust(wspace=0.0)

        elif m1 == 0 and m2 == 1:
            fig, ax = plt.subplots(nrows=1, ncols=len(key_k), sharex=True, sharey=True, figsize=(2.0, 10))
            for kk in range(len(key_k)):
                plt_PQnP = ax[kk].imshow(key_rates[kk], cmap='seismic', vmin=-1.0, vmax=1.0)
                ax[kk].invert_yaxis()
                ax[kk].set_title(r'$k={:d}$'.format(key_k[kk]), fontsize=20, **{'fontname':'Times'})
                ax[kk].set_xticks(np.arange(0,len(key_m),1), np.array(key_m)+1, fontsize=20, **{'fontname':'Times'}, rotation=45)
                
            ax[0].set_yticks(np.arange(0,len(key_l),1), np.array(key_l)+1, fontsize=20, **{'fontname':'Times'})
            ax[0].set_ylabel('Normal mode index ($l$)', fontsize=20, **{'fontname':'Times'})
            fig.text(0.50, 0.02, 'Normal mode\nindex ($m$)', ha='center', fontsize=20, **{'fontname':'Times'})
            plt.subplots_adjust(wspace=0.0)
        
        elif m1 == 1 and m2 == 0:
            key_rates = np.transpose(key_rates, axes=(1, 0, 2))
            fig, ax = plt.subplots(nrows=1, ncols=len(key_l), sharex=True, sharey=True, figsize=(3.0, 10))
            if len(key_rates) > 1:
                for ll in range(len(key_l)):
                    plt_PQnP = ax[ll].imshow(key_rates[ll], cmap='seismic', vmin=-1.0, vmax=1.0)
                    ax[ll].invert_yaxis()
                    ax[ll].set_title(r'$l={:d}$'.format(key_l[ll]), fontsize=20, **{'fontname':'Times'})
                    ax[ll].set_xticks(np.arange(0,len(key_m),1), np.array(key_m)+1, fontsize=20, **{'fontname':'Times'}, rotation=45)
                    
                ax[0].set_yticks(np.arange(0,len(key_k),1), np.array(key_k)+1, fontsize=20, **{'fontname':'Times'})
                ax[0].set_ylabel('Normal mode index ($k$)', fontsize=20, **{'fontname':'Times'})
                fig.text(0.50, 0.02, 'Normal mode\nindex ($m$)', ha='center', fontsize=20, **{'fontname':'Times'})
                plt.subplots_adjust(wspace=0.0)
            else:
                plt_PQnP = ax.imshow(key_rates[0], cmap='seismic', vmin=-1.0, vmax=1.0)
                ax.invert_yaxis()
                ax.set_title(r'$l={:d}$'.format(key_l[0]), fontsize=20, **{'fontname':'Times'})
                ax.set_xticks(np.arange(0,len(key_m),1), np.array(key_m)+1, fontsize=20, **{'fontname':'Times'}, rotation=45)
                ax.set_xlabel('Normal mode\nindex ($m$)', fontsize=20, **{'fontname':'Times'})
                ax.set_ylabel('Normal mode index ($k$)', fontsize=20, **{'fontname':'Times'})
                ax.set_yticks(np.arange(0,len(key_k),1), np.array(key_k)+1, fontsize=20, **{'fontname':'Times'})
        
        elif m1 == 1 and m2 == 1:
            fig, ax = plt.subplots(nrows=1, ncols=len(key_k), sharex=True, sharey=True, figsize=(8, 8))
            if len(key_rates) > 1:
                for kk in range(len(key_k)):
                    plt_PQnP = ax[kk].imshow(key_rates[kk], cmap='seismic', vmin=-1.0, vmax=1.0)
                    ax[kk].invert_yaxis()
                    ax[kk].set_title(r'$k={:d}$'.format(key_k[kk]), fontsize=20, **{'fontname':'Times'})
                    ax[kk].set_xticks(np.arange(0,len(key_m),1), np.array(key_m)+1, fontsize=20, **{'fontname':'Times'}, rotation=45)
                    
                ax[0].set_yticks(np.arange(0,len(key_l),1), np.array(key_l)+1, fontsize=20, **{'fontname':'Times'})
                ax[0].set_ylabel('Normal mode index ($l$)', fontsize=20, **{'fontname':'Times'})
                fig.text(0.50, 0.02, 'Normal mode\nindex ($m$)', ha='center', fontsize=20, **{'fontname':'Times'})
                plt.subplots_adjust(wspace=0.0)
            else:
                plt_PQnP = ax.imshow(key_rates[0], cmap='seismic', vmin=-1.0, vmax=1.0)
                ax.invert_yaxis()
                ax.set_xticks(np.arange(0,len(key_m),1), np.array(key_m)+1, fontsize=20, **{'fontname':'Times'}, rotation=45)
                ax.set_title(r'$k={:d}$'.format(key_k[0]), fontsize=20, **{'fontname':'Times'})
                ax.set_xlabel('Normal mode index ($m$)', fontsize=20, **{'fontname':'Times'})
                ax.set_ylabel('Normal mode index ($l$)', fontsize=20, **{'fontname':'Times'})
                ax.set_yticks(np.arange(0,len(key_l),1), np.array(key_l)+1, fontsize=20, **{'fontname':'Times'})
    
        # Save plot as a file
        plt.savefig(rate_file_loc + 'modes_PQnP_m{:d}m{:d}.png'.format(m1+1, m2+1), dpi=600)
        plt.show()


'''
CF_PnPQ
'''
# Need to loop over intermediate states
for m1 in range(nintmed):
    for m2 in range(nintmed):
        rate_PnPQ_mm = rate_PnPQ[m1][m2]
        triad_PnPQ_mm = triad_PnPQ[m1][m2]

        # List of unique indices
        k_indices = [triad_PnPQ_mm[i][0] for i in range(len(triad_PnPQ_mm))]
        k_unique = []
        for k in k_indices:
            if k not in k_unique:
                k_unique.append(k) # list of unique k indices
        l_indices = [triad_PnPQ_mm[i][1] for i in range(len(triad_PnPQ_mm))]
        l_unique = []
        for l in l_indices:
            if l not in l_unique:
                l_unique.append(l) # list of unique l indices
        m_indices = [triad_PnPQ_mm[i][2] for i in range(len(triad_PnPQ_mm))]
        m_unique = []
        for m in m_indices:
            if m not in m_unique:
                m_unique.append(m) # list of unique m indices

        # Array of rates
        rates = np.zeros((len(k_unique), len(l_unique), len(m_unique)))
        for i, triad in enumerate(triad_PnPQ_mm):
            rates[k_unique.index(triad[0]), l_unique.index(triad[1]), m_unique.index(triad[2])] = rate_PnPQ_mm[i]
        max_rate = np.max(abs(rates))

        # Extract only significant contributors above 'thres'
        thres = max_rate * contrib_cutoff
        key_k, key_l, key_m = [], [], []
        for k in range(len(k_unique)):
            for l in range(len(l_unique)):
                for m in range(len(m_unique)):
                    if abs(rates[k,l, m]) > thres:
                        if k_unique[k] not in key_k:
                            key_k.append(k_unique[k])
                        if l_unique[l] not in key_l:
                            key_l.append(l_unique[l])
                        if m_unique[m] not in key_m:
                            key_m.append(m_unique[m])
        # sort indices (k is already in an ascending order)
        key_l = sorted(key_l)
        key_m = sorted(key_m)

        key_rates = np.zeros((len(key_k), len(key_l), len(key_m)))
        for a, k in enumerate(key_k):
            for b, l in enumerate(key_l):
                for c, m in enumerate(key_m):
                    key_rates[a,b,c] = rates[k_unique.index(k), l_unique.index(l), m_unique.index(m)]
        # Normalized absolute rates
        key_rates = key_rates/max_rate

        # Matrix plot
        if m1 == 0 and m2 == 0:
            key_rates = np.transpose(key_rates, axes=(2,0,1))
            fig, ax = plt.subplots(nrows=1, ncols=len(key_m), sharex=True, sharey=True, figsize=(15, 15))
            for mm in range(len(key_m)):
                plt_PnPQ = ax[mm].imshow(key_rates[mm], cmap='seismic', vmin=-1.0, vmax=1.0)
                ax[mm].invert_yaxis()
                ax[mm].set_title(r'$m={:d}$'.format(key_m[mm]), fontsize=20, **{'fontname':'Times'})
                ax[mm].set_xticks(np.arange(0,len(key_l),1), np.array(key_l)+1, fontsize=20, **{'fontname':'Times'}, rotation=45)

            ax[0].set_yticks(np.arange(0,len(key_k),1), np.array(key_k)+1, fontsize=20, **{'fontname':'Times'})
            ax[0].set_ylabel('Normal mode index ($k$)', fontsize=20, **{'fontname':'Times'})
            fig.text(0.50, 0.30, 'Normal mode index ($l$)', ha='center', fontsize=20, **{'fontname':'Times'})
            plt.subplots_adjust(wspace=0.0)
        
        if m1 == 0 and m2 == 1:
            key_rates = np.transpose(key_rates, axes=(2,0,1))
            fig, ax = plt.subplots(nrows=1, ncols=len(key_m), sharex=True, sharey=True, figsize=(2.6, 12))
            for mm in range(len(key_m)):
                plt_PnPQ = ax[mm].imshow(key_rates[mm], cmap='seismic', vmin=-1.0, vmax=1.0)
                ax[mm].invert_yaxis()
                ax[mm].set_title(r'$m={:d}$'.format(key_m[mm]), fontsize=20, **{'fontname':'Times'})
                ax[mm].set_xticks(np.arange(0,len(key_l),1), np.array(key_l)+1, fontsize=20, **{'fontname':'Times'}, rotation=45)

            ax[0].set_yticks(np.arange(0,len(key_k),1), np.array(key_k)+1, fontsize=20, **{'fontname':'Times'})
            ax[0].set_ylabel('Normal mode index ($k$)', fontsize=20, **{'fontname':'Times'})
            fig.text(0.50, 0.05, 'Normal mode index ($l$)', ha='center', fontsize=20, **{'fontname':'Times'})
            plt.subplots_adjust(wspace=0.0)
        
        if m1 == 1 and m2 == 0:
            key_rates = np.transpose(key_rates, axes=(2,0,1))
            fig, ax = plt.subplots(nrows=1, ncols=len(key_m), sharex=True, sharey=True, figsize=(17, 16))
            for mm in range(len(key_m)):
                plt_PnPQ = ax[mm].imshow(key_rates[mm], cmap='seismic', vmin=-1.0, vmax=1.0)
                ax[mm].invert_yaxis()
                ax[mm].set_title(r'$m={:d}$'.format(key_m[mm]), fontsize=20, **{'fontname':'Times'})
                ax[mm].set_xticks(np.arange(0,len(key_l),1), np.array(key_l)+1, fontsize=20, **{'fontname':'Times'}, rotation=45)

            ax[0].set_yticks(np.arange(0,len(key_k),1), np.array(key_k)+1, fontsize=20, **{'fontname':'Times'})
            ax[0].set_ylabel('Normal mode index ($k$)', fontsize=20, **{'fontname':'Times'})
            fig.text(0.50, 0.30, 'Normal mode index ($l$)', ha='center', fontsize=20, **{'fontname':'Times'})
            plt.subplots_adjust(wspace=0.0)
        
        if m1 == 1 and m2 == 1:
            key_rates = np.transpose(key_rates, axes=(0,1,2))
            fig, ax = plt.subplots(nrows=1, ncols=len(key_k), sharex=True, sharey=True, figsize=(6, 6))
            for kk in range(len(key_k)):
                plt_PnPQ = ax[kk].imshow(key_rates[kk], cmap='seismic', vmin=-1.0, vmax=1.0)
                ax[kk].invert_yaxis()
                ax[kk].set_title(r'$k={:d}$'.format(key_k[kk]), fontsize=16, **{'fontname':'Times'})
                ax[kk].set_xticks(np.arange(0,len(key_m),1), np.array(key_m)+1, fontsize=16, **{'fontname':'Times'})

            ax[0].set_yticks(np.arange(0,len(key_l),1), np.array(key_l)+1, fontsize=16, **{'fontname':'Times'})
            ax[0].set_ylabel('Normal mode index ($l$)', fontsize=20, **{'fontname':'Times'})
            fig.text(0.50, 0.22, 'Normal mode index ($k$)', ha='center', fontsize=16, **{'fontname':'Times'})
            plt.subplots_adjust(wspace=0.0)

        # Save plot as a file
        plt.savefig(rate_file_loc + 'modes_PnPQ_m{:d}m{:d}.png'.format(m1+1, m2+1), dpi=600)
        plt.show()
        
#%%
'''
CF_PQnPQ
'''
contrib_cutoff = 0.10

# Need to loop over intermediate states
for m1 in range(nintmed):
    for m2 in range(nintmed):
        rate_PQnPQ_mm = rate_PQnPQ[m1][m2]
        tetrad_PQnPQ_mm = tetrad_PQnPQ[m1][m2]

        # List of unique indices
        k_indices = [tetrad_PQnPQ_mm[i][0] for i in range(len(tetrad_PQnPQ_mm))]
        k_unique = []
        for k in k_indices:
            if k not in k_unique:
                k_unique.append(k) # list of unique k indices
        l_indices = [tetrad_PQnPQ_mm[i][1] for i in range(len(tetrad_PQnPQ_mm))]
        l_unique = []
        for l in l_indices:
            if l not in l_unique:
                l_unique.append(l) # list of unique l indices
        m_indices = [tetrad_PQnPQ_mm[i][2] for i in range(len(tetrad_PQnPQ_mm))]
        m_unique = []
        for m in m_indices:
            if m not in m_unique:
                m_unique.append(m) # list of unique m indices
        n_indices = [tetrad_PQnPQ_mm[i][3] for i in range(len(tetrad_PQnPQ_mm))]
        n_unique = []
        for n in n_indices:
            if n not in n_unique:
                n_unique.append(n) # list of unique n indices

        # Array of rates
        rates = np.zeros((len(k_unique), len(l_unique), len(m_unique), len(n_unique)))
        for i, tetrad in enumerate(tetrad_PQnPQ_mm):
            rates[k_unique.index(tetrad[0]), l_unique.index(tetrad[1]), m_unique.index(tetrad[2]), n_unique.index(tetrad[3])] = rate_PQnPQ_mm[i]
        max_rate = np.max(abs(rates))

        # Extract only significant contributors above 'thres'
        thres = max_rate * contrib_cutoff
        key_k, key_l, key_m, key_n = [], [], [], []
        for k in range(len(k_unique)):
            for l in range(len(l_unique)):
                for m in range(len(m_unique)):
                    for n in range(len(n_unique)):
                        if abs(rates[k,l, m, n]) > thres:
                            if k_unique[k] not in key_k:
                                key_k.append(k_unique[k])
                            if l_unique[l] not in key_l:
                                key_l.append(l_unique[l])
                            if m_unique[m] not in key_m:
                                key_m.append(m_unique[m])
                            if n_unique[n] not in key_n:
                                key_n.append(n_unique[n])
        # sort indices (k is already in an ascending order)
        key_l = sorted(key_l)
        key_m = sorted(key_m)
        key_n = sorted(key_n)

        key_rates = np.zeros((len(key_k), len(key_l), len(key_m), len(key_n)))
        for a, k in enumerate(key_k):
            for b, l in enumerate(key_l):
                for c, m in enumerate(key_m):
                    for d, n in enumerate(key_n):
                        key_rates[a,b,c,d] = rates[k_unique.index(k), l_unique.index(l), m_unique.index(m), n_unique.index(n)]
        # Normalized absolute rates
        key_rates = key_rates/max_rate

        # Matrix plot
        if m1 == 0 and m2 == 0:
            # Re-order indices so the first two are along the final state modes and the last two are the initial state modes
            key_rates = np.transpose(key_rates, axes=(0,3,1,2))
            fig, ax = plt.subplots(nrows=len(key_k), ncols=len(key_n), sharex=True, sharey=True, figsize=(18, 18))
            plot_id = 0
            for kk in range(len(key_k)):
                for nn in range(len(key_n)):                    
                    plt_PQnPQ = ax[kk, nn].imshow(key_rates[kk, nn], cmap='seismic', vmin=-1.0, vmax=1.0)
                    ax[kk, nn].invert_yaxis()
                    ax[kk, nn].set_title('{:d}'.format(np.round(wno_f[key_k[kk]]).astype(int)) 
                                         + 'cm$^{-1}$, ' 
                                         + '{:d}'.format(np.round(wno_f[key_n[nn]]).astype(int)) 
                                         + 'cm$^{-1}$', fontsize=18, **{'fontname':'Times'})
                    ax[kk, nn].set_xticks(np.arange(0, len(key_m), 1), np.round(wno_i[key_m]).astype(int), fontsize=18, **{'fontname':'Times'}, rotation=60)
                    if ((kk+1)*len(key_k)+(nn+1)) % len(key_n) == 1:
                        ax[kk, nn].set_yticks(np.arange(0, len(key_l), 1), np.round(wno_i[key_l]).astype(int), fontsize=18, **{'fontname':'Times'})
                        #ax[kk, nn].set_ylabel('$T_{1}$ normal mode frequency [cm$^{-1}$]', fontsize=18, **{'fontname':'Times'})
                    plot_id += 1
            
            # x-axis label
            fig.text(0.50, 0.04, '$T_{1}$ normal mode frequency [cm$^{-1}$]', ha='center', fontsize=20, **{'fontname':'Times'})
            # y-axis label
            fig.text(0.08, 0.34, '$T_{1}$ normal mode frequency [cm$^{-1}$]', ha='center', fontsize=20, **{'fontname':'Times'}, rotation=90)
            # Adjust spacing between figure grids
            plt.subplots_adjust(wspace=0.5, hspace=0.62)
        
        # Add more if cases here in case there are multiple intermediate states

        # Save plot as a file
        plt.savefig(rate_file_loc + 'modes_PQnPQ_m{:d}m{:d}.png'.format(m1+1, m2+1), dpi=600)
        plt.show()

#%%
"""
Rate visualization routine for A6AP-Cz
"""
# rate_file_2_loc = '/Users/user/Desktop/a6ap-cz/rate_calculations/no_pcm_flat/631gd_sp_nac/flat_geo_soc/300K_gwidth0cm-1_ts3.000_ts6.000_ts10.000_ti0.0_tm900.00_tm3600.00_tf8000.00_coupcut100_boole/'
# rate_file_2     = 'rate_ts3.000_ts6.000_ts10.000_ti0.0_tm900.0_tm3600.0_tf8000.0_ccut100_boole.out'
from matplotlib.pyplot import cm
color = cm.rainbow(np.linspace(0, 1, 11))
width = 0.6  # the width of the bars: can also be len(x) sequence
fig, ax = plt.subplots(figsize=(8.0, 6.0))

### Experimental rates
expt_rate   = 1.9e+07
# expt_rate = 9.9e+03

### Read rates from output files
with open(rate_file_loc + rate_file, 'r') as f:
    for line in f:
        if 'Computation result' in line:
            break
    [f.readline() for i in range(2)]
    
    total_rate = np.real(complex(f.readline().split()[-1]))
    #total_rate = np.log10(total_rate)
    
    [f.readline() for i in range(5)]
    k0_0  = float(f.readline().split()[-1])
    [f.readline() for i in range(nsec+1)]
    k0_Q  = float(f.readline().split()[-1])
    [f.readline() for i in range(nsec+1)]
    k0_QQ = float(f.readline().split()[-1])
    
    [f.readline() for i in range(14)]
    
    # k1_nPi     = np.real(complex(f.readline().split()[-1]))
    # k1_nPiQf   = np.real(complex(f.readline().split()[-1]))
    # k1_QinPi   = np.real(complex(f.readline().split()[-1]))
    # k1_QinPiQf = np.real(complex(f.readline().split()[-1]))
    # # k1_nPf     = float(f.readline().split()[-1])
    # # k1_nQiPf   = float(f.readline().split()[-1])
    # # k1_QinPf   = float(f.readline().split()[-1])
    # # k1_QinQiPf = float(f.readline().split()[-1])
    
    # [f.readline() for i in range(3)]
    
    k2_PinPi     = float(f.readline().split()[-1])
    k2_PinPiQf   = float(f.readline().split()[-1])
    k2_QfPinPi   = float(f.readline().split()[-1])
    k2_QfPinPiQf = float(f.readline().split()[-1])
    # k2_PinPf     = float(f.readline().split()[-1])
    # k2_PinQiPf   = float(f.readline().split()[-1])
    # k2_QfPinPf   = float(f.readline().split()[-1])
    # k2_QfPinQiPf = float(f.readline().split()[-1])
    # k2_PfnPi     = float(f.readline().split()[-1])
    # k2_PfnPiQf   = float(f.readline().split()[-1])
    # k2_PfQinPi   = float(f.readline().split()[-1])
    # k2_PfQinPiQf = float(f.readline().split()[-1])
    # k2_PfnPf     = float(f.readline().split()[-1])
    # k2_PfnQiPf   = float(f.readline().split()[-1])
    # k2_PfQinPf   = float(f.readline().split()[-1])
    # k2_PfQinQiPf = float(f.readline().split()[-1])
    
# ### Read rates from output files set 2
# with open(rate_file_2_loc + rate_file_2, 'r') as f:
#     for line in f:
#         if 'Computation result' in line:
#             break
#     [f.readline() for i in range(2)]
    
#     total_rate_2 = np.real(complex(f.readline().split()[-1]))
#     #total_rate = np.log10(total_rate)
    
#     [f.readline() for i in range(5)]
#     k0_0_2  = float(f.readline().split()[-1])
#     [f.readline() for i in range(nsec+1)]
#     k0_Q_2  = float(f.readline().split()[-1])
#     [f.readline() for i in range(nsec+1)]
#     k0_QQ_2 = float(f.readline().split()[-1])
    
#     [f.readline() for i in range(14)]
    
#     # k1_nPi_2     = np.real(complex(f.readline().split()[-1]))
#     # k1_nPiQf_2   = np.real(complex(f.readline().split()[-1]))
#     # k1_QinPi_2   = np.real(complex(f.readline().split()[-1]))
#     # k1_QinPiQf_2 = np.real(complex(f.readline().split()[-1]))
#     # # k1_nPf_2     = float(f.readline().split()[-1])
#     # # k1_nQiPf_2   = float(f.readline().split()[-1])
#     # # k1_QinPf_2   = float(f.readline().split()[-1])
#     # # k1_QinQiPf_2 = float(f.readline().split()[-1])
    
#     # [f.readline() for i in range(3)]
    
#     k2_PinPi_2     = float(f.readline().split()[-1])
#     k2_PinPiQf_2   = float(f.readline().split()[-1])
#     k2_QfPinPi_2   = float(f.readline().split()[-1])
#     k2_QfPinPiQf_2 = float(f.readline().split()[-1])
#     # k2_PinPf_2     = float(f.readline().split()[-1])
#     # k2_PinQiPf_2   = float(f.readline().split()[-1])
#     # k2_QfPinPf_2   = float(f.readline().split()[-1])
#     # k2_QfPinQiPf_2 = float(f.readline().split()[-1])
#     # k2_PfnPi_2     = float(f.readline().split()[-1])
#     # k2_PfnPiQf_2   = float(f.readline().split()[-1])
#     # k2_PfQinPi_2   = float(f.readline().split()[-1])
#     # k2_PfQinPiQf_2 = float(f.readline().split()[-1])
#     # k2_PfnPf_2     = float(f.readline().split()[-1])
#     # k2_PfnQiPf_2   = float(f.readline().split()[-1])
#     # k2_PfQinPf_2   = float(f.readline().split()[-1])
#     # k2_PfQinQiPf_2 = float(f.readline().split()[-1])

rates1 = [k0_0, k0_Q, k0_QQ, k2_PinPi, k2_PinPiQf, k2_QfPinPi, k2_QfPinPiQf, total_rate, expt_rate]
log_rates1 = np.round(np.log10(abs(np.array(rates1))), decimals=2)
# rates2 = [k0_0_2, k0_Q_2, k0_QQ_2, k2_PinPi_2, k2_PinPiQf_2, k2_QfPinPi_2, k2_QfPinPiQf_2, total_rate_2]
# log_rates2 = np.round(np.log10(abs(np.array(rates2))), decimals=2)
labels = ['$0$', '$Q_{S_1}$', '$Q_{S_1}$-$Q_{S_1}$', 
          '$P_{T_1}$-$P_{T_1}$', '$P_{T_1}$-$P_{T_1}Q_{S_1}$', '$Q_{S_1}P_{T_1}$-$P_{T_1}$', '$Q_{S_1}P_{T_1}$-$P_{T_1}Q_{S_1}$',
          '$k_{\mathrm{RISC}}$', 'Expt.']

# List of colors
colors1, colors2 = [], []
for idx, rate in enumerate(rates1):
    if idx == len(rates1) - 1:
        colors1.append('darkslategrey')
    else:
        if rate < 0:
            colors1.append('dodgerblue')
        else:
            colors1.append('crimson')
# for rate in rates2:
#     if rate < 0:
#         colors2.append('lightsteelblue')
#     else:
#         colors2.append('lightpink')

# List of bar width
#width1 = ([width] * len(rates1)) + [width*2]
width1 = [width] * len(rates1)
# width2 = [width] * len(rates2)

# List of y-positions of bars
displacements1 = np.array(width1)/2
displacements1[-1] = 0
# displacements2 = np.array(width2)/2
y_positions1 = np.arange(len(rates1)) #- displacements1
# y_positions2 = np.arange(len(rates2)) + displacements2

# Plot the first set of bars
bars1 = ax.barh(y_positions1, log_rates1, height=width1, color=colors1)
# Add labels to the bars
formatted_labels1 = [r'{:.1f}$\times 10^{:d}$'.format(rates1[i]/10**int(log_rates1[i]), int(log_rates1[i])) for i in range(len(rates1))]
ax.bar_label(bars1, 
             labels=formatted_labels1,
             label_type='edge', padding=6,
             fontsize=17, fontweight='bold', **{'fontname':'Times'})
# # Plot the second set of bars
# bars2 = ax.barh(y_positions2, log_rates2, height=width2, color=colors2, alpha=0.7)
# # Add labels to the bars
# formatted_labels2 = [r'{:.1f}$\times 10^{:d}$'.format(rates2[i]/10**int(log_rates2[i]), int(log_rates2[i])) for i in range(len(rates2))]
# ax.bar_label(bars2, 
#              labels=formatted_labels2,
#              label_type='center', padding=0,
#              color='k',
#              fontsize=15, fontweight='bold', **{'fontname':'Times'})
    
from matplotlib import ticker
formatter = ticker.ScalarFormatter(useMathText=True)
ax.xaxis.set_major_formatter(formatter)
ax.xaxis.offsetText.set_fontsize(18)
ax.xaxis.offsetText.set_fontname('Times')
ax.invert_yaxis()  # labels read top-to-bottom

ax.set_yticks(np.arange(len(rates1)), labels=labels, fontsize=18, **{'fontname':'Times'})
#ax.set_yticklabels(labels)

plt.ticklabel_format(axis='x', style='sci', scilimits=(0,4))
plt.xticks(fontsize=18, **{'fontname':'Times'})
#plt.xlim(min(rate_data.values())*1.1, max(rate_data.values())*1.1)
plt.xlim(0.0, 8.9)
plt.ylim(8.5,-0.5)
#plt.vlines(0, -0.5, len(rate_data)-0.5, color='k', linestyle='-')
plt.hlines(2.5, 0.0, 8.9, color='k', linestyle=':')
plt.hlines(6.5, 0.0, 8.9, color='k', linestyle='-')
#plt.hlines(26.5, min(rate_data.values())*1.1, max(rate_data.values())*1.1, color='k', linestyle='-')
plt.xlabel(r'$log_{10}(|k[s^{-1}]|)$', fontsize=18, **{'fontname':'Times'})
# L = ax.legend()
# plt.setp(L.texts, family='Times', fontsize=13)
#plt.savefig(loc + 'rate_breakdown.png', dpi=600, bbox_inches='tight')
plt.show()

#%%
"""
Rate visualization routine for DABNA-1
"""
from matplotlib.pyplot import cm
color = cm.rainbow(np.linspace(0, 1, 11))
width = 0.60  # the width of the bars: can also be len(x) sequence

### Read rates from output files
with open(rate_file_loc + rate_file, 'r') as f:
    for line in f:
        if 'Computation result' in line:
            break
    [f.readline() for i in range(2)]
    
    total_rate = np.real(complex(f.readline().split()[-1]))
    #total_rate = np.log10(total_rate)
    
    [f.readline() for i in range(5)]
    k0_0  = float(f.readline().split()[-1])
    [f.readline() for i in range(nsec+1)]
    k0_Q  = float(f.readline().split()[-1])
    [f.readline() for i in range(nsec+1)]
    k0_QQ = float(f.readline().split()[-1])
    
    [f.readline() for i in range(14)]
    
    # k1_nPi     = np.real(complex(f.readline().split()[-1]))
    # k1_nPiQf   = np.real(complex(f.readline().split()[-1]))
    # k1_QinPi   = np.real(complex(f.readline().split()[-1]))
    # k1_QinPiQf = np.real(complex(f.readline().split()[-1]))
    # # k1_nPf     = float(f.readline().split()[-1])
    # # k1_nQiPf   = float(f.readline().split()[-1])
    # # k1_QinPf   = float(f.readline().split()[-1])
    # # k1_QinQiPf = float(f.readline().split()[-1])
    
    # [f.readline() for i in range(3)]
    
    k2_PinPi     = float(f.readline().split()[-1])
    k2_PinPiQf   = float(f.readline().split()[-1])
    k2_QfPinPi   = float(f.readline().split()[-1])
    k2_QfPinPiQf = float(f.readline().split()[-1])
    # k2_PinPf     = float(f.readline().split()[-1])
    # k2_PinQiPf   = float(f.readline().split()[-1])
    # k2_QfPinPf   = float(f.readline().split()[-1])
    # k2_QfPinQiPf = float(f.readline().split()[-1])
    # k2_PfnPi     = float(f.readline().split()[-1])
    # k2_PfnPiQf   = float(f.readline().split()[-1])
    # k2_PfQinPi   = float(f.readline().split()[-1])
    # k2_PfQinPiQf = float(f.readline().split()[-1])
    # k2_PfnPf     = float(f.readline().split()[-1])
    # k2_PfnQiPf   = float(f.readline().split()[-1])
    # k2_PfQinPf   = float(f.readline().split()[-1])
    # k2_PfQinQiPf = float(f.readline().split()[-1])
    
rate_data = {
    # k0 rates
    '$0$': k0_0,
    '$Q_{S_1}$': k0_Q,
    '$Q_{S_1}$-$Q_{S_1}$': k0_QQ,
    
    # k1 rates
    # '$P_{T_1}$': k1_nPi,
    # '$P_{T_1}Q_{S_1}$': k1_nPiQf,
    # '$Q_{T_1}$-$P_{T_1}$': k1_QinPi,
    # 'Qi-PiQf': k1_QinPiQf,
    # 'Pf': k1_nPf,
    # 'QiPf': k1_nQiPf,
    # 'Qi-Pf': k1_QinPf,
    # 'Qi-QiPf': k1_QinQiPf,
    
    # k2 rates
    '$P_{T_1}$-$P_{T_1}$': k2_PinPi,
    '$P_{T_1}$-$P_{T_1}Q_{S_1}$': k2_PinPiQf,
    '$Q_{S_1}P_{T_1}$-$P_{T_1}$': k2_QfPinPi,
    '$Q_{S_1}P_{T_1}$-$P_{T_1}Q_{S_1}$': k2_QfPinPiQf,
    # 'Pi-Pf': k2_PinPf,
    # 'Pi-QiPf': k2_PinQiPf,
    # 'QfPi-Pf': k2_QfPinPf,
    # 'QfPi-QiPf': k2_QfPinQiPf,
    # 'Pf-Pi': k2_PfnPi,
    # 'Pf-PiQf': k2_PfnPiQf,
    # 'PfQi-Pi': k2_PfQinPi,
    # 'PfQi-PiQf': k2_PfQinPiQf,
    # 'Pf-Pf': k2_PfnPf,
    # 'Pf-QiPf': k2_PfnQiPf,
    # 'PfQi-Pf': k2_PfQinPf,
    # 'PfQi-QiPf': k2_PfQinQiPf,
    
    '$k_{\mathrm{RISC}}$': total_rate,
    'Expt.': 9.9e+03,
    }

fig, ax = plt.subplots(figsize=(8.0, 6.0))
color_idx = 10
for corr, rt in rate_data.items():
    if corr == 'Expt.':
        c = 'darkslategrey'
    else:
        if rt < 0:
            c  = 'dodgerblue'
        else:
            c  = 'crimson'
    label_type = 'edge'
    padding = 6
    a = 1.0
    
    rt_sign = np.sign(rt)
    rt_log = np.log10(abs(rt))
    custom_label = '{:.1e}'.format(rt_sign * 10**rt_log)
    bars = ax.barh(corr, rt_log, width, label=corr, color=c, alpha=a)
    ax.bar_label(bars, #fmt=lambda x: f'{rt_sign * 10**x:.2e}',
                 fmt=r'{:.1f}$\times 10^{:d}$'.format(rt/10**int(rt_log), int(rt_log)), 
                 label_type=label_type, padding=padding,
                 fontsize=17, **{'fontname':'Times'})

    color_idx -= 1

from matplotlib import ticker
formatter = ticker.ScalarFormatter(useMathText=True)
ax.xaxis.set_major_formatter(formatter)
ax.xaxis.offsetText.set_fontsize(18)
ax.xaxis.offsetText.set_fontname('Times')
ax.invert_yaxis()  # labels read top-to-bottom

#ax.set_title('Rate breakdown')
plt.yticks(fontsize=18, **{'fontname':'Times'})
plt.ticklabel_format(axis='x', style='sci', scilimits=(0,4))
plt.xticks(fontsize=18, **{'fontname':'Times'})
#plt.xlim(min(rate_data.values())*1.1, max(rate_data.values())*1.1)
plt.xlim(0.0, 6.0)
plt.ylim(8.5,-0.5)
#plt.vlines(0, -0.5, len(rate_data)-0.5, color='k', linestyle='-')
plt.hlines(2.5, 0.0, 6.0, color='k', linestyle=':')
plt.hlines(6.5, 0.0, 6.0, color='k', linestyle='-')
#plt.hlines(26.5, min(rate_data.values())*1.1, max(rate_data.values())*1.1, color='k', linestyle='-')
plt.xlabel(r'$log_{10}(|k[s^{-1}]|)$', fontsize=18, **{'fontname':'Times'})
# L = ax.legend()
# plt.setp(L.texts, family='Times', fontsize=13)
plt.savefig(rate_file_loc + 'rate_breakdown.png', dpi=600, bbox_inches='tight')
plt.show()


