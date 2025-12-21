#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 12:34:56 2024

@author: kenmiyazaki
"""

"""
2nd order rate formulation for IC and ISC. 
- Molecular vibrational modes are taken as harmonic          
- Singularity associated with propagator Green's function
  is analytically removed as presented in K. Miyazaki & N. Ananth,
  JCP 156 044111 (2022).
- An extensive list of correlation functions is computed for k0 (FGR), 
  k1, & k2, each consisting of varying sets of thermal vibration correlation 
  functions (TVCFs), integrating both IC & ISC, and accomodating a variable 
  number of intermediate states.
"""

import numpy as np
from numpy import linalg as LA
from scipy.linalg import inv
import os
from multiprocessing import Pool
from functools import partial
import time
import datetime
import subprocess
from sys_param import *
from sim_param import *
from constants import *
from fun import *
subprocess.run(["mkdir", out_path])
__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))


'''
Beginning of code
'''
start_time = time.time()

"""
Read frequencies, mode symmetries, reduced masses and Cartesian vectors.
"""
nmode = 3*natom
wno_f, redmas_f, vec_f, Nimf_f, ffreq_package = read_freq(file_path, ffreq_file)
wno_i, redmas_i, vec_i, Nimf_i, ifreq_package = read_freq(file_path, ifreq_file)
# Set the number of imag modes to the larger one of initial and final states
if Nimf_i >= Nimf_f:
    Nimf = Nimf_i
elif Nimf_f >= Nimf_i:
    Nimf = Nimf_f
nvib = nmode - Nimf - 6

if 'qchem' in fhess_file[8:]:
    vec_f = read_hess(fhess_file)
if  'qchem' in ihess_file[8:]:
    vec_i = read_hess(ihess_file)


# Convert freq in wavenumber into herz and a.u.
hz_i, hz_f = np.zeros(nvib),np.zeros(nvib)
frqau_i, frqau_f = np.zeros(nvib), np.zeros(nvib)
pad = 6+Nimf
for i in range(nvib):
    hz_i[i] += wno_i[i+pad] * light*100 * scale
    hz_f[i] += wno_f[i+pad] * light*100 * scale
    frqau_i[i] += wno_i[i+pad] * light*100 * autime2s * scale
    frqau_f[i] += wno_f[i+pad] * light*100 * autime2s * scale
omg_i , omg_f = np.diag(2*pi*frqau_i), np.diag(2*pi*frqau_f)


"""
Identify the density of vibrational levels of the final electronic state
and the Gaussian width parameter eta
"""
# dos = 0 # Density of vibrational states 
# for ifrq in np.diag(omg_f):
#     nquanta = 1.0/ifrq # The number of vibrational quanta per unit energy (a.u.) 
#     dos += nquanta
# width = 1/dos
width_au = width * wno2j/eh2j


"""
Read AMU of each atom in the molecule
"""
masamu, atm_sym = read_atomic_amu(mass_file)
masamu_ext = np.zeros(nmode)
for iatom in range(natom):
    masamu_ext[iatom*3:(iatom+1)*3] = masamu[iatom]
# Convert Mass into atomic unit
masau_ext = masamu_ext * amu2au
masau_mat = np.diag(masau_ext)


"""
Calculate mass-weighted Cartesian vectors
"""
mwvec_f, prenorm_mwvec_f = get_mass_weighted_nvec(ffreq_package, vec_f, masamu)
mwvec_i, prenorm_mwvec_i = get_mass_weighted_nvec(ifreq_package, vec_i, masamu)


"""
Calculate Duschinsky (orthonormal) matrix between mwvec_f and mwvec_i
"""
S = np.zeros((nvib, nvib))
for i in range(nvib):
    for j in range(nvib):
        S[i,j] += np.dot(mwvec_i[i+pad], mwvec_f[j+pad])
        
# # Conversion to dimensionless normal coordinates
# S = np.linalg.multi_dot([np.diag(np.sqrt(np.diag(omg_i))), S, np.diag(np.diag(omg_f)**-0.5)])


"""
Calculate the displacement vectors d along normal modes 
"""
crd_f, crd_i = read_geo(file_path, fgeo_file, igeo_file)
crddif2 = crd_f - crd_i
crddif1 = crd_i - crd_f
mwdisp2, mwdisp1 = np.zeros(3*natom), np.zeros(3*natom)
for i in range(natom):
    mwdisp2[3*i:3*i+3] += crddif2[i] * (masamu[i]**0.5) #in ang*amu**0.5
    mwdisp1[3*i:3*i+3] += crddif1[i] * (masamu[i]**0.5) #in ang*amu**0.5

d2 = np.matmul(mwvec_i, mwdisp2)
d1 = np.matmul(mwvec_f, mwdisp1)

d2au, d1au = d2 * (amu2au**0.5) * ang2bohr, d1 * (amu2au**0.5) * ang2bohr
d2si, d1si = d2 * (amu2kg**0.5) * (10**-10), d1 * (amu2kg**0.5) * (10**-10)
# Remove displacement along translational & rotational modes
d2au = np.delete(d2au, (0,1,2,3,4,5)) 
d1au = np.delete(d1au, (0,1,2,3,4,5))
for i in range(Nimf):
    d2au = np.delete(d2au, (0))
    d1au = np.delete(d1au, (0))    
dispau = d2au
# # Conversion to dimensionless normal coordinates
# dispau = np.matmul(np.sqrt(omg_i), dispau)

omg_S_d = [omg_i, omg_f, S, dispau] # List of arrays necessary for later computation


"""
Rotate the optimized geometries into normal coordinates
"""
# Rotate Cartesian geometries into normal coordinates
ncgeo_i = np.matmul(mwvec_i, np.matmul(masau_mat**0.5, (crd_i * ang2bohr).reshape(-1)))
ncgeo_f = np.matmul(mwvec_f, np.matmul(masau_mat**0.5, (crd_f * ang2bohr).reshape(-1))) 


"""
Define the optimized geometries in unitless normal coordinates
"""
# Multiply sqrt(freq) to the above normal coords to make them unitless
frqau_i = np.insert(frqau_i, 0, (0,0,0,0,0,0))
frqau_f = np.insert(frqau_f, 0, (0,0,0,0,0,0))
unitless_ncgeo_i = ncgeo_i * (frqau_i**0.5)
unitless_ncgeo_f = ncgeo_f * (frqau_f**0.5)


"""
Classify each of the electronic transition processes into either IC or ISC

process_id : 
    Pairs of integers to indicate the process type of each transition.
    
    In a transition i --> m1 --> f, a pair of integer [X1, Y1] is defined such that 
    the process type of i --> m1 is indicated by X1 and that of m1 --> f by Y1.
    X1 and Y1 can either be 0 (N/A), 1 (IC), or 2 (ISC).
    
    For the direct transition i --> f, the pair will be [X0, 0], where X0 indicates
    the process type of i --> f.
"""
init_sm  = sm[0] # Spin multiplicity symbols of initial state
final_sm = sm[-1] # Spin multiplicity symbols of final state
if order > 1:
    intmed_sm = sm[1:-1] # Spin multiplicity symbols of intermediate states
    nintmed = len(intmed_sm) # Number of intermediate states
else:
    intmed_sm = [] # No intermediate states in 1st-order simulation
    nintmed = 0    

process_id = np.zeros((nintmed+1, 2), dtype=int) 
if init_sm == final_sm:
    ''' Case 1: Overall internal conversion '''
    process_id[0,0] = 1    
    if order > 1:
        for m in range(len(intmed_sm)):
            if intmed_sm[m] == init_sm: # initial-->intermediate and intermediate-->final are both IC
                process_id[m+1,0], process_id[m+1,1] = 1, 1
            else: # initial-->intermediate and intermediate-->final are both ISC
                process_id[m+1,0], process_id[m+1,1] = 2, 2 

else: 
    ''' Case 2: Overall intersystem crossing '''
    process_id[0,0] = 2 
    if order > 1:
        for m in range(len(intmed_sm)):
            if intmed_sm[m] == init_sm:  
                process_id[m+1,0], process_id[m+1,1] = 1, 2 
            else: 
                process_id[m+1,0], process_id[m+1,1] = 2, 1 


"""
Check linearity of SOC around the equil geometries by plotting
"""
# # SOC on grids for graphic purpose
# mode_idx = []
# disp_mode_ffreq, disp_mode_ifreq = [], []
# ndisp = len(disp_size)
# soc_data  = np.zeros((nintmed+1, 2, 10*nlot, 2*ndisp+1))
# soc_data[0, 0, :, 2] = soc0_if
# for idisp in range(ndisp):
#     for ilot in range(nlot):
#         soc_filename = soc_if_filename_format.format(ilot+1, disp_size[idisp])
#         with open(file_path + soc_filename, 'r') as f:
#             line_number = 0
#             for line in f:
#                 x = line.split()
#                 soc_data[0, 0, 10*ilot+line_number, -idisp+ndisp-1] = float(x[1])
#                 soc_data[0, 0, 10*ilot+line_number, idisp+ndisp+1]  = float(x[2])
#                 modeidx = int(x[0])
#                 disp_mode_ifreq.append(wno_i[modeidx-1])
#                 mode_idx.append([int(x[0]) - 7, float(x[1]), soc0_if, float(x[2])])
#                 line_number += 1


# # # SOC on grids for graphic purpose
# # mode_idx = []
# # soc_data = np.zeros((nintmed+1, 2, 10*nlot, 3))
# # disp_mode_ffreq, disp_mode_ifreq = [], []
# # if process_id[0,0] == 2: # If the direct transition is ISC
# #     for ilot in range(nlot):
# #         # SOC between initial and final states on grid points
# #         line_number = 0
# #         with open(file_path + soc_if_files[ilot], 'r') as f0:
# #             for line0 in f0:
# #                 x0 = line0.split()
# #                 soc_data[0, 0, 10*ilot+line_number] = np.array([float(x0[1]), soc0_if, float(x0[2])]) 
# #                 mode_idx.append([int(x0[0]) - 7, float(x0[1]), soc0_if, float(x0[2])])
# #                 modeidx = int(x0[0])
# #                 disp_mode_ifreq.append(wno_i[modeidx-1])
# #                 line_number += 1

# if nintmed > 0:
#     for m in range(nintmed):
#         isc_where = np.where(process_id[m+1] == 2)[0]
#         if len(isc_where) > 0:
#             for isc_idx in isc_where:
#                 soc_data[m+1, isc_idx, :, 2] = soc0_intmed[m][isc_idx]
#                 for ilot in range(nlot):
#                     for idisp in range(ndisp):
#                         soc_filename = soc_intmed_filename_format[m].format(ilot+1, disp_size[idisp])
#                         with open(file_path + soc_filename, 'r') as f:
#                             line_number = 0
#                             for line in f:
#                                 x = line.split()
#                                 soc_data[m+1, isc_idx, 10*ilot+line_number, -idisp+ndisp-1] = float(x[1])
#                                 soc_data[m+1, isc_idx, 10*ilot+line_number, idisp+ndisp+1]  = float(x[2])
#                                 modeidx = int(x[0])
#                                 disp_mode_ffreq.append(wno_f[modeidx-1])
#                                 line_number += 1



"""
Plot the SOC over geometry displacement
"""
# import matplotlib.pyplot as plt
# import matplotlib as mpl
# from matplotlib.pyplot import cm
# import numpy as np
# eps = list(reversed(-1*np.array(disp_size))) + [0.0] + disp_size # Unitless normal coordinates grids # Unitless normal coordinates grids

# #from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
# mpl.rcParams.update(mpl.rcParamsDefault)
# mpl.rcParams['mathtext.rm'] = 'Times'
# mpl.rcParams['mathtext.it'] = 'Times:italic'
# mpl.rcParams['mathtext.default'] = 'it'
# mpl.rcParams['mathtext.fontset'] = 'cm'

# color = cm.rainbow(np.linspace(0, 1, nmodes_per_lot))
# xticks = [str(eps[i]) for i in range(len(eps))]

# # SOC between initial and final state
# for ilot in range(nlot):
#     fig, ax = plt.subplots()
#     for i in range(nmodes_per_lot):
#         plt.plot(eps, soc_data[0, 0, nmodes_per_lot*ilot + i], marker='o', linestyle='-', color=color[i], 
#                   label=r'%d cm$^{-1}$' %(disp_mode_ffreq[i]))
        
#     plt.xlabel(r'$\varepsilon$', fontsize=20, **{'fontname':'Times'})
#     plt.xticks(eps, xticks, fontsize=16, **{'fontname':'Times'})
#     plt.ylabel(r'$V^{soc}_{T_{1}-S_{1}}$ [cm$^{-1}$]', fontsize=20, **{'fontname':'Times'})
#     plt.yticks(fontsize=18, **{'fontname':'Times'})
#     minim = np.min(soc_data[0, 0, nmodes_per_lot*ilot:nmodes_per_lot*(ilot+1)])
#     maxim = np.max(soc_data[0, 0, nmodes_per_lot*ilot:nmodes_per_lot*(ilot+1)])
#     diff = maxim - minim
#     plt.ylim(minim - 0.05*diff, maxim + 0.05*diff)
#     L = ax.legend(ncols=1, loc=(1.05, 0.05))
#     plt.setp(L.texts, family='Times', fontsize=14)
#     plt.savefig(file_path + 'soc_fi_gradients.png', dpi=600, bbox_inches='tight')
#     plt.show()

# # SOC involving intermediate state
# if nintmed > 0:
#     for m in range(nintmed):
#         isc_where = np.where(process_id[m+1] == 2)[0]
#         if len(isc_where) > 0:
#             for isc_idx in isc_where:
#                 for ilot in range(nlot):
#                     fig, ax = plt.subplots()
#                     for i in range(nmodes_per_lot):
#                         plt.plot(eps, soc_data[m+1, isc_idx, nmodes_per_lot*ilot + i], marker='o', linestyle='-', color=color[i], 
#                                   label=r'%d cm$^{-1}$' %(disp_mode_ifreq[i]))
        
#                     plt.xlabel(r'$\varepsilon$', fontsize=20, **{'fontname':'Times'})
#                     plt.xticks(eps, xticks, fontsize=16, **{'fontname':'Times'})
#                     plt.ylabel(r'$V^{soc}_{T_{1}-S_{2}}$ [cm$^{-1}$]', fontsize=20, **{'fontname':'Times'})
#                     plt.yticks(fontsize=18, **{'fontname':'Times'})
#                     minim = np.min(soc_data[m+1, isc_idx, nmodes_per_lot*ilot:nmodes_per_lot*(ilot+1)])
#                     maxim = np.max(soc_data[m+1, isc_idx, nmodes_per_lot*ilot:nmodes_per_lot*(ilot+1)])
#                     diff = maxim - minim
#                     plt.ylim(minim - 0.05*diff, maxim + 0.05*diff)
#                     L = ax.legend(ncols=1, loc=(1.05, 0.05))
#                     plt.setp(L.texts, family='Times', fontsize=14)
#                     plt.savefig(file_path + 'soc_gradients_{:s}.png'.format(sm[m+1]), dpi=600, bbox_inches='tight')
#                     plt.show()



"""
Construct the array of SOCs at displaced geometries according to the "process_id"

-h and +h displacements of SOC to compute the derivatives using central difference
Order of entry: [SOC(x-h), SOC(x+h)]

soc_grid[m, b, c, d] : 
    m = 0: i --> f direct transition
    m = n: transition involving n-th intermediate state
    
    b = 0 (or 1): i --> m (or m --> f). When m = 0, b = 0 only.
    
    c: SOC along c-th vibrational mode
    
    d = 0 (or 1): -h displacement (or +h displacement)
"""
soc_grid = np.zeros((len(process_id), order, nvib, 2))
if process_id[0, 0] == 2:
    for ilot in range(nlot):
        try:
            with open(file_path + soc_if_files[ilot], 'r') as f:
                for line in f:
                    x = line.split()
                    soc_grid[0, 0, int(x[0])-7, 0] = float(x[1]) * wno2j * eh2j**-1 # Need to subtract 7 because the index is padded by 7
                    soc_grid[0, 0, int(x[0])-7, 1] = float(x[2]) * wno2j * eh2j**-1 # (6 for translation+rotation and 1 for python indexing)
        except IOError:
            print('{:d}-th lot of i --> f SOCs displaced along vibrational modes is not provided.\n'.format(ilot+1))
            
        if order > 1:
            for m in range(nintmed):
                soc_idx = np.where(process_id[m+1] == 2)[0][0]
                try:
                    with open(file_path + soc_intmed_files[m][ilot][soc_idx], 'r') as f:
                        for line in f:
                            x = line.split()
                            soc_grid[m+1, soc_idx, int(x[0])-7, 0] = float(x[1]) * wno2j * eh2j**-1 # Need to subtract 7 because the index is padded by 7
                            soc_grid[m+1, soc_idx, int(x[0])-7, 1] = float(x[2]) * wno2j * eh2j**-1 # (6 for translation+rotation and 1 for python indexing)
                except IOError:
                    print('{:d}-th lot of SOCs involving {:d}-th intermediate state\ndisplaced along vibrational modes is not provided.\n'.format(ilot+1, m+1))


"""
1. Construct the array of NAC along each vibrational mode
2. Construct the array of SOC and its derivatives along each vibrational mode
"""
if process_id[0, 0] == 1: # "i --> f is IC"
    # Direct coupling between initial and final states
    try:
        with open(file_path + nac_if_file, 'r') as f:
            for i in range(3):
                f.readline()
            nac_if_xyz = np.zeros(nmode)
            for i in range(natom):
                x = f.readline().split()
                for j in range(3):
                    nac_if_xyz[j + 3*i] += float(x[j+2])
                    
        '''Rotate Cartesian NAC into normal coordinate'''
        # In total, need the factor of mas^-1 and rotation by L.T        
        nac_if = np.zeros(nvib)
        for i in range(nvib):
            nac_if[i] += np.dot(vec_f[i+6+Nimf], nac_if_xyz) # in amu**-0.5 * bohr^-1
        nac_if *= (amu2au)**(-0.5)
        
    except IOError:
        print('Initial state and final state are claimed to have the same multiplicity\nbut no NACME file was provided. \n')
        
    if order > 1:
        nac_intmed  = np.zeros((nintmed, 2, nvib)) # (intermediate state index, i --> m or m --> f, vib mode index)
        soc0_intmed = np.array([[soc0_intmed[s][0]*wno2j*(eh2j**-1), soc0_intmed[s][1]*wno2j*(eh2j**-1)] for s in range(nintmed)])
        soc1_intmed = np.zeros((nintmed, 2, nvib))
        for m in range(nintmed):
            if process_id[m+1, 0] == 1 and process_id[m+1, 1] == 1: # "Both i --> m and m --> f are IC"
                try:
                    with open(os.path.join(file_path, nac_intmed_file[m][0]), 'r') as f:
                        nac_im_tmp = np.zeros(nmode)
                        [f.readline() for i in range(4)]
                        for iatom in range(natom):
                            x = f.readline().split()[1:]
                            for xyz in range(3):
                                nac_im_tmp[iatom*3+xyz] = float(x[xyz])
                        nac_xyz = nac_im_tmp
                        
                    '''Rotate Cartesian NAC into normal coordinate'''
                    # In total, need the factor of mas^-1 and rotation by L.T        
                    for i in range(nvib):
                        nac_intmed[m, 0, i] += np.dot(vec_i[i+pad], nac_xyz) # in amu**-0.5 * bohr^-1
                    nac_intmed[m, 0] *= (amu2au)**(-0.5)
                    
                except IOError:
                    print('Initial state and {:d}-th intermediate state are claimed\nto have the same multiplicity but no NACME file was provided. \n'.format(m+1))
                    
                try:
                    with open(os.path.join(file_path, nac_intmed_file[m][1]), 'r') as f:
                        nac_mf_tmp = np.zeros(nmode)
                        [f.readline() for i in range(4)]
                        for iatom in range(natom):
                            x = f.readline().split()[1:]
                            for xyz in range(3):
                                nac_mf_tmp[iatom*3+xyz] = float(x[xyz])
                        nac_xyz = nac_mf_tmp
                        
                    '''Rotate Cartesian NAC into normal coordinate'''
                    # In total, need the factor of mas^-1 and rotation by L.T        
                    for i in range(nvib):
                        nac_intmed[m, 1, i] += np.dot(vec_f[i+pad], nac_xyz) # in amu**-0.5 * bohr^-1
                    nac_intmed[m, 1] *= (amu2au)**(-0.5)
                    
                except IOError:
                    print('{:d}-th intermediate state and final state are claimed\nto have the same multiplicity but no NACME file was provided. \n'.format(m+1))
                    
            elif process_id[m+1, 0] == 2 and process_id[m+1, 1] == 2: # "Both i --> m and m --> f are ISC"
                for ivib in range(nvib):
                    ### sqrt of frequency is multiplied to change the unitless normal coord into mass-weighted normal coord.
                    soc1_intmed[m, 0, ivib] = np.sqrt(omg_i[ivib, ivib]) * (soc_grid[m+1, 0, ivib, 1] - soc_grid[m+1, 0, ivib, 0])/(2 * NM_disp_factor)
                    soc1_intmed[m, 1, ivib] = np.sqrt(omg_f[ivib, ivib]) * (soc_grid[m+1, 1, ivib, 1] - soc_grid[m+1, 1, ivib, 0])/(2 * NM_disp_factor)
                
elif process_id[0, 0] == 2: # "i --> f is ISC"   
    # Direct coupling between initial and final states
    soc0_if = soc0_if * wno2j * (eh2j**-1) # Conversion to a.u.
    
    # Compute derivatives by finite difference    
    soc1_if = np.zeros(nvib)
    for ivib in range(nvib):
        ### sqrt of frequency is multiplied to change the unitless normal coord into mass-weighted normal coord.
        soc1_if[ivib] = np.sqrt(omg_f[ivib,ivib]) * (soc_grid[0,0,ivib,1] - soc_grid[0,0,ivib,0])/(2 * NM_disp_factor)
        
    # Coupling involving intermediate states  
    if order > 1:
        nac_intmed  = np.zeros((nintmed, 2, nvib)) # (intermediate state index, i --> m or m --> f, vib mode index)
        soc0_intmed = np.array([[soc0_intmed[s][0]*wno2j*(eh2j**-1), soc0_intmed[s][1]*wno2j*(eh2j**-1)] for s in range(nintmed)])
        soc1_intmed = np.zeros((nintmed, 2, nvib))
        for m in range(nintmed):
            if process_id[m+1, 0] == 1 and process_id[m+1, 1] == 2: # "i --> m is IC and m --> f is ISC"
                try:
                    with open(os.path.join(file_path, nac_intmed_file[m][0]), 'r') as f:
                        nac_im_tmp = np.zeros(nmode)
                        [f.readline() for i in range(4)]
                        for iatom in range(natom):
                            x = f.readline().split()[1:]
                            for xyz in range(3):
                                nac_im_tmp[iatom*3+xyz] = float(x[xyz])
                        nac_xyz = nac_im_tmp
                        
                    '''Rotate Cartesian NAC into normal coordinate'''
                    # In total, need the factor of mas^-1 and rotation by L.T        
                    for i in range(nvib):
                        nac_intmed[m, 0, i] += np.dot(vec_i[i+pad], nac_xyz) # in amu**-0.5 * bohr^-1
                    nac_intmed[m, 0] *= (amu2au)**(-0.5)
                    
                except IOError:
                    print('Initial state and {:d}-th intermediate state are claimed\nto have the same multiplicity but no NACME file was provided. \n'.format(m+1))
                    
                for ivib in range(nvib):
                    ### sqrt of frequency is multiplied to change the unitless normal coord into mass-weighted normal coord.
                    soc1_intmed[m, 1, ivib] = np.sqrt(omg_f[ivib, ivib]) * (soc_grid[m+1, 1, ivib, 1] - soc_grid[m+1, 1, ivib, 0])/(2 * NM_disp_factor)
                    
            elif process_id[m+1, 0] == 2 and process_id[m+1, 1] == 1: # "i --> m is ISC and m --> f is IC"
                try:
                    with open(os.path.join(file_path, nac_intmed_file[m][1]), 'r') as f:
                        nac_mf_tmp = np.zeros(nmode)
                        [f.readline() for i in range(4)]
                        for iatom in range(natom):
                            x = f.readline().split()[1:]
                            for xyz in range(3):
                                nac_mf_tmp[iatom*3+xyz] = float(x[xyz])
                        nac_xyz = nac_mf_tmp
                        
                    '''Rotate Cartesian NAC into normal coordinate'''
                    # In total, need the factor of mas^-1 and rotation by L.T        
                    for i in range(nvib):
                        nac_intmed[m, 1, i] += np.dot(vec_f[i+pad], nac_xyz) # in amu**-0.5 * bohr^-1
                    nac_intmed[m, 1] *= (amu2au)**(-0.5)
                    
                except IOError:
                    print('{:d}-th intermediate state and final state are claimed\nto have the same multiplicity but no NACME file was provided. \n'.format(m+1))
                    
                for ivib in range(nvib):
                    ### sqrt of frequency is multiplied to change the unitless normal coord into mass-weighted normal coord.
                    soc1_intmed[m, 0, ivib] = np.sqrt(omg_i[ivib, ivib]) * (soc_grid[m+1, 0, ivib, 1] - soc_grid[m+1, 0, ivib, 0])/(2 * NM_disp_factor)




"""
Calculate the product of couplings 
Note the overall process goes down as i --> {m1, m2, ...} (intermediate) --> f,
where the spins of i, m, & f depend on the problem at hand.

Subsequently, form the lists of indices, index pairs, index triads, 
and index tetrads of vibrational mdoes.  
- Here, couplings smaller than 1/coup_cutoff of the abs max of them
  are omitted from the rest of the simulation to save papers 

##############################################################################
Note: While NACMEs <psi2|d/dQ|psi1> are anti-symmetric, the matrix elements of 
      nuclear momentum operator <psi2|P|psi1> is symmetric. 
##############################################################################
"""

def store_indices_for_k2(m, mp, p_p, p_pq, pq_p, pq_pq, coup_p_p, coup_p_pq, coup_pq_p, coup_pq_pq):
    '''
    Selection of normal modes indices for k2 correlation functions

    Parameters
    ----------
    m, mp : Integer indices for intermediate states
    p_p, p_pq, pq_p, pq_pq : Lists of normal indices to update
    coup_p_p : ARRAY
        coupPnP
    coup_p_pq : ARRAY
        coupPnPQ and coupPnQP
    coup_pq_p : ARRAY
        coupPQnP and coupQPnP
    coup_pq_pq : ARRAY
        coupPQnPQ, coupPQnQP, coupQPnPQ, and coupQPnQP

    Returns
    -------
    p_p, p_pq, pq_p, pq_pq

    '''
    p_p_tmp, p_pq_tmp, pq_p_tmp, pq_pq_tmp = [], [], [], []
    coup_p_p_max   = max([max(abs(coup_p_p[m,mp,i])) for i in range(nvib)])
    tmp1 = abs(coup_p_pq[m,mp])
    coup_p_pq_max  = max([max([max(tmp1[i,j]) for j in range(10*nlot)]) for i in range(10*nlot)])
    tmp2 = abs(coup_pq_p[m,mp])
    coup_pq_p_max  = max([max([max(tmp2[i,j]) for j in range(10*nlot)]) for i in range(10*nlot)])
    tmp3 = abs(coup_pq_pq[m,mp])
    coup_pq_pq_max = max([max([max([max(tmp3[i,j,k]) for k in range(10*nlot)]) for j in range(10*nlot)]) for i in range(10*nlot)])
    for k in range(nvib):
        for l in range(nvib):
            if abs(coup_p_p[m,mp,k,l]) > coup_p_p_max/coup_cutoff_2:
                p_p_tmp.append([k, l])
            for n in range(10*nlot):
                if tmp1[k,l,n] > coup_p_pq_max/coup_cutoff_3:
                    p_pq_tmp.append([k, l, n])
                if tmp2[k,l,n] > coup_pq_p_max/coup_cutoff_3:
                    pq_p_tmp.append([k, l, n])
                for o in range(10*nlot):
                    if tmp3[k,l,n,o] > coup_pq_pq_max/coup_cutoff_4:
                        pq_pq_tmp.append([k, l, n, o])
    p_p.append(np.array(p_p_tmp))
    p_pq.append(np.array(p_pq_tmp))
    pq_p.append(np.array(pq_p_tmp))
    pq_pq.append(np.array(pq_pq_tmp))
    return p_p, p_pq, pq_p, pq_pq


# For 0th-order rate
if process_id[0,0] == 1: # If the overall process is IC
    coupPnP = np.outer(nac_if, nac_if)
    # Store the indices of normal modes along which abs(couplings) > threshold
    klPnP = []
    coupPnP_max   = max([max(abs(coupPnP[i])) for i in range(nvib)])
    for k in range(nvib):
        for l in range(nvib):
            if abs(coupPnP[k,l]) > coupPnP_max/coup_cutoff_2:
                klPnP.append([k, l])
    klPnP = np.array(klPnP)

elif process_id[0,0] == 2: # If the overall process is ISC
    coup0   = soc0_if * soc0_if
    coup_nQ = soc0_if * soc1_if # vector
    coupQnQ = np.outer(soc1_if, soc1_if) # matrix
    # Store the indices of normal modes along which abs(couplings) > threshold
    k_nQ, klQnQ = [], []
    coup_nQ_max   = max(abs(coup_nQ))
    coupQnQ_max   = max([max(abs(coupQnQ[i])) for i in range(nvib)])
    for k in range(nvib):
        if abs(coup_nQ[k]) > coup_nQ_max/coup_cutoff_1:
            k_nQ.append(k)
        for l in range(nvib):
            if abs(coupQnQ[k,l]) > coupQnQ_max/coup_cutoff_2:
                klQnQ.append([k, l])
    k_nQ  = np.array(k_nQ)
    klQnQ = np.array(klQnQ)

    if order > 1:
        # For 1st-order rate
        # Possible ramification over intermediate states m
        coup_nP  = np.zeros((nintmed, nvib)) # encompass Pi & Pf
        coup_nPQ = np.zeros((nintmed, nvib, nvib)) # encompass PiQf & QiPf
        coupQnP  = np.zeros((nintmed, nvib, nvib)) # encompass Qi_Pi & Qi_Pf
        coupQnPQ = np.zeros((nintmed, nvib, nvib, nvib)) # encompass Qi_PiQf & Qi_QiPf
        k_nP, kl_nPQ, klQnP, klmQnPQ = [], [], [], []
        for m in range(nintmed):
            dE = Ei - Em[m]
            if process_id[m+1, 0] == 1 and process_id[m+1, 1] == 2: # IC & ISC
                coup_nP[m]   = soc0_if * nac_intmed[m, 0] * soc0_intmed[m, 1] / dE # vector
                coup_nPQ[m]  = soc0_if * np.outer(nac_intmed[m, 0], soc1_intmed[m, 1]) / dE # matrix
                coupQnP[m]   = np.outer(soc1_if, nac_intmed[m, 0]) * soc0_intmed[m, 1] / dE # matrix
                coupQnPQ[m] = np.multiply.outer(np.outer(soc1_if, nac_intmed[m, 0]), soc1_intmed[m, 1]) / dE # rank-3 tensor
                
                # Store the indices of normal modes along which abs(couplings) > threshold
                k_nP_tmp, kl_nPQ_tmp, klQnP_tmp, klmQnPQ_tmp = [], [], [], []
                coup_nP_max   = max(abs(coup_nP[m]))
                coup_nPQ_max  = max([max(abs(coup_nPQ[m,i])) for i in range(nvib)])
                coupQnP_max   = max([max(abs(coupQnP[m,i])) for i in range(nvib)])
                coupQnPQ_max  = max([max([max(abs(coupQnPQ[m,i,j])) for j in range(nvib)]) for i in range(nvib)])
                for k in range(nvib):
                    if abs(coup_nP[m,k]) > coup_nP_max/coup_cutoff_1:
                        k_nP_tmp.append(k)
                    for l in range(nvib):
                        if abs(coup_nPQ[m,k,l]) > coup_nPQ_max/coup_cutoff_2:
                            kl_nPQ_tmp.append([k, l])
                        if abs(coupQnP[m,k,l]) > coupQnP_max/coup_cutoff_2:
                            klQnP_tmp.append([k, l])
                        for n in range(nvib):
                            if abs(coupQnPQ[m,k,l,n]) > coupQnPQ_max/coup_cutoff_3:
                                klmQnPQ_tmp.append([k, l, n])
                k_nP.append(np.array(k_nP_tmp))
                kl_nPQ.append(np.array(kl_nPQ_tmp))
                klQnP.append(np.array(klQnP_tmp))
                klmQnPQ.append(np.array(klmQnPQ_tmp))
                                
            elif process_id[m+1, 0] == 2 and process_id[m+1, 1] == 1: # ISC & IC
                coup_nP[m]   = soc0_if * soc0_intmed[m, 0] * nac_intmed[m, 1] / dE # vector
                coup_nPQ[m]  = soc0_if * np.outer(soc1_intmed[m, 0], nac_intmed[m, 1]) / dE # matrix
                coupQnP[m]   = np.outer(soc1_if * soc0_intmed[m, 0], nac_intmed[m, 1]) / dE # matrix
                coupQnPQ[m] = np.multiply.outer(np.outer(soc1_if, soc1_intmed[m, 0]), nac_intmed[m, 1]) / dE # rank-3 tensor
                
                # Store the indices of normal modes along which abs(couplings) > threshold
                k_nP_tmp, kl_nPQ_tmp, klQnP_tmp, klmQnPQ_tmp = [], [], [], []
                coup_nP_max   = max(abs(coup_nP[m]))
                coup_nPQ_max  = max([max(abs(coup_nPQ[m,i])) for i in range(nvib)])
                coupQnP_max   = max([max(abs(coupQnP[m,i])) for i in range(nvib)])
                coupQnPQ_max  = max([max([max(abs(coupQnPQ[m,i,j])) for j in range(nvib)]) for i in range(nvib)])
                for k in range(nvib):
                    if abs(coup_nP[m,k]) > coup_nP_max/coup_cutoff_1:
                        k_nP_tmp.append(k)
                    for l in range(nvib):
                        if abs(coup_nPQ[m,k,l]) > coup_nPQ_max/coup_cutoff_2:
                            kl_nPQ_tmp.append([k, l])
                        if abs(coupQnP[m,k,l]) > coupQnP_max/coup_cutoff_2:
                            klQnP_tmp.append([k, l])
                        for n in range(nvib):
                            if abs(coupQnPQ[m,k,l,n]) > coupQnPQ_max/coup_cutoff_3:
                                klmQnPQ_tmp.append([k, l, n])
                k_nP.append(np.array(k_nP_tmp))
                kl_nPQ.append(np.array(kl_nPQ_tmp))
                klQnP.append(np.array(klQnP_tmp))
                klmQnPQ.append(np.array(klmQnPQ_tmp))
        
        # For 2nd-order rate
        # Possible ramification over intermediate states m and mp
        coupPnP   = np.zeros((nintmed, nintmed, nvib, nvib)) # encompass Pf_Pf & Pf_Pi & Pi_Pf & Pi_Pi
        coupPnPQ  = np.zeros((nintmed, nintmed, nvib, nvib, nvib)) # encompass Pf_PiQf & Pi_PiQf & Pf_QiPf & Pi_QiPf
        coupPQnP  = np.zeros((nintmed, nintmed, nvib, nvib, nvib)) # encompass PfQi_Pi & PfQi_Pf & QfPi_Pf & QfPi_Pi
        coupPQnPQ = np.zeros((nintmed, nintmed, nvib, nvib, nvib, nvib)) # encompass PfQi_PiQf & PfQi_QiPf & QfPi_QiPf & QfPi_PiQf
        klPnP, klmPnPQ, klmPQnP, klmnPQnPQ = [], [], [], []
        for m in range(nintmed):
            klPnP_tmp2, klmPnPQ_tmp2, klmPQnP_tmp2, klmnPQnPQ_tmp2 = [], [], [], []
            for mp in range(nintmed):
                dEdE = (Ei - Em[m]) * (Ei - Em[mp])
                if process_id[m+1, 0] == 1 and process_id[m+1, 1] == 2: # IC & ISC for intermediate state m
                    if process_id[mp+1, 0] == 1 and process_id[mp+1, 1] == 2: # IC & ISC for intermediate state mp
                        coupPnP[m, mp]   = np.outer(soc0_intmed[m, 1] * nac_intmed[m, 0], nac_intmed[mp, 0] * soc0_intmed[mp, 1]) / dEdE # matrix
                        coupPnPQ[m, mp]  = np.multiply.outer(np.outer(soc0_intmed[m, 1] * nac_intmed[m, 0], nac_intmed[mp, 0]), soc1_intmed[mp, 1]) / dEdE # rank-3 tensor
                        coupPQnP[m, mp]  = np.multiply.outer(np.outer(soc1_intmed[m, 1], nac_intmed[m, 0]), nac_intmed[mp, 0]) * soc0_intmed[mp, 1] / dEdE # rank-3 tensor
                        coupPQnPQ[m, mp] = np.multiply.outer(np.outer(soc1_intmed[m, 1], nac_intmed[m, 0]), np.outer(nac_intmed[mp, 0], soc1_intmed[mp, 1])) / dEdE # rank-4 tensor 
                        klPnP_tmp2, klmPnPQ_tmp2, klmPQnP_tmp2, klmnPQnPQ_tmp2 = store_indices_for_k2(m, mp, 
                                                                                                      klPnP_tmp2, klmPnPQ_tmp2, 
                                                                                                      klmPQnP_tmp2, klmnPQnPQ_tmp2, 
                                                                                                      coupPnP, coupPnPQ, coupPQnP, coupPQnPQ)
                        
                    elif process_id[mp+1, 0] == 2 and process_id[mp+1, 1] == 1: # ISC & IC for intermediate state mp
                        coupPnP[m, mp]   = np.outer(soc0_intmed[m, 1] * nac_intmed[m, 0], soc0_intmed[mp, 0] * nac_intmed[mp, 1]) / dEdE # matrix
                        coupPnPQ[m, mp]  = np.multiply.outer(np.outer(soc0_intmed[m, 1] * nac_intmed[m, 0], soc1_intmed[mp, 0]), nac_intmed[mp, 1]) / dEdE # rank-3 tensor
                        coupPQnP[m, mp]  = np.multiply.outer(np.outer(soc1_intmed[m, 1], nac_intmed[m, 0]), soc0_intmed[mp, 0] * nac_intmed[mp, 1]) / dEdE # rank-3 tensor
                        coupPQnPQ[m, mp] = np.multiply.outer(np.outer(soc1_intmed[m, 1], nac_intmed[m, 0]), np.outer(soc1_intmed[mp, 0], nac_intmed[mp, 1])) / dEdE # rank-4 tensor 
                        klPnP_tmp2, klmPnPQ_tmp2, klmPQnP_tmp2, klmnPQnPQ_tmp2 = store_indices_for_k2(m, mp, 
                                                                                                      klPnP_tmp2, klmPnPQ_tmp2, 
                                                                                                      klmPQnP_tmp2, klmnPQnPQ_tmp2, 
                                                                                                      coupPnP, coupPnPQ, coupPQnP, coupPQnPQ)
       
                elif process_id[m+1, 0] == 2 and process_id[m+1, 1] == 1: # ISC & IC for intermediate state m
                    if process_id[mp+1, 0] == 1 and process_id[mp+1, 1] == 2: # IC & ISC for intermediate state mp
                        coupPnP[m, mp]   = np.outer(nac_intmed[m, 1] * soc0_intmed[m, 0], nac_intmed[mp, 0] * soc0_intmed[mp, 1]) / dEdE # matrix
                        coupPnPQ[m, mp]  = np.multiply.outer(np.outer(nac_intmed[m, 1] * soc0_intmed[m, 0], nac_intmed[mp, 0]), soc1_intmed[mp, 1]) / dEdE # rank-3 tensor
                        coupPQnP[m, mp]  = np.multiply.outer(np.outer(nac_intmed[m, 1], soc1_intmed[m, 0]), nac_intmed[mp, 0] * soc0_intmed[mp, 1]) / dEdE # rank-3 tensor
                        coupPQnPQ[m, mp] = np.multiply.outer(np.outer(nac_intmed[m, 1], soc1_intmed[m, 0]), np.outer(nac_intmed[mp, 0], soc1_intmed[mp, 1])) / dEdE # rank-4 tensor 
                        klPnP_tmp2, klmPnPQ_tmp2, klmPQnP_tmp2, klmnPQnPQ_tmp2 = store_indices_for_k2(m, mp, 
                                                                                                      klPnP_tmp2, klmPnPQ_tmp2, 
                                                                                                      klmPQnP_tmp2, klmnPQnPQ_tmp2, 
                                                                                                      coupPnP, coupPnPQ, coupPQnP, coupPQnPQ)
                        
                    elif process_id[mp+1, 0] == 2 and process_id[mp+1, 1] == 1: # ISC & IC for intermediate state mp
                        coupPnP[m, mp]   = np.outer(nac_intmed[m, 1] * soc0_intmed[m, 0], soc0_intmed[mp, 0] * nac_intmed[mp, 1]) / dEdE # matrix
                        coupPnPQ[m, mp]  = np.multiply.outer(np.outer(nac_intmed[m, 1] * soc0_intmed[m, 0], soc1_intmed[mp, 0]), nac_intmed[mp, 1]) / dEdE # rank-3 tensor
                        coupPQnP[m, mp]  = np.multiply.outer(np.outer(nac_intmed[m, 1], soc1_intmed[m, 0]), soc0_intmed[mp, 0] * nac_intmed[mp, 1]) / dEdE # rank-3 tensor
                        coupPQnPQ[m, mp] = np.multiply.outer(np.outer(nac_intmed[m, 1], soc1_intmed[m, 0]), np.outer(soc1_intmed[mp, 0], nac_intmed[mp, 1])) / dEdE # rank-4 tensor 
                        klPnP_tmp2, klmPnPQ_tmp2, klmPQnP_tmp2, klmnPQnPQ_tmp2 = store_indices_for_k2(m, mp, 
                                                                                                      klPnP_tmp2, klmPnPQ_tmp2, 
                                                                                                      klmPQnP_tmp2, klmnPQnPQ_tmp2, 
                                                                                                      coupPnP, coupPnPQ, coupPQnP, coupPQnPQ)
                
            klPnP.append(klPnP_tmp2)
            klmPnPQ.append(klmPnPQ_tmp2)
            klmPQnP.append(klmPQnP_tmp2)
            klmnPQnPQ.append(klmnPQnPQ_tmp2)

if order > 1:
    del k_nP_tmp, kl_nPQ_tmp, klQnP_tmp, klmQnPQ_tmp
    del klPnP_tmp2, klmPnPQ_tmp2, klmPQnP_tmp2, klmnPQnPQ_tmp2
#    del soc1_im, soc1_mf, nac_im, nac_mf
#    del nac_im_xyz, nac_mf_xyz


"""
Reduce the size of coupling arrays by getting rid of trivial couplings
Then compile them as a dictionary for later calls
"""
class couplings:
    null, _nQ, QnQ = None, None, None
    _nP, QnP, QnPQ = None, None, None
    PnP, PnPQ, PQnP, PQnPQ = None, None, None, None
class indices:   
    _nQ, QnQ = None, None
    _nP, QnP, QnPQ = None, None, None
    PnP, PnPQ, PQnP, PQnPQ = None, None, None, None

coup_dict = couplings()
idx_dict  = indices()

if process_id[0,0] == 1: # If the overall process is IC
     coup_dict.PnP = coupPnP[klPnP[:,0], klPnP[:,1]]
     idx_dict.PnP  = klPnP
#    if order > 1:
#        ... 
elif process_id[0,0] == 2: # If the overall process is ISC
    coup_dict.null = coup0
    if len(k_nQ) > 0:
        coup_dict._nQ  = coup_nQ[k_nQ]
        idx_dict._nQ   = k_nQ 
    if len(klQnQ) > 0:
        coup_dict.QnQ  = coupQnQ[klQnQ[:,0], klQnQ[:,1]]
        idx_dict.QnQ   = klQnQ
    if order > 1:
        coup_nP_list, coup_nPQ_list = [], []
        coupQnP_list, coupQnPQ_list = [], []
        coupPnP_list, coupPnPQ_list = [], []
        coupPQnP_list, coupPQnPQ_list = [], []
        for m in range(nintmed):
            coup_nP_list.append(coup_nP[m][k_nP[m]])
            coup_nPQ_list.append(coup_nPQ[m][kl_nPQ[m][:,0], kl_nPQ[m][:,1]])
            coupQnP_list.append(coupQnP[m][klQnP[m][:,0], klQnP[m][:,1]])
            coupQnPQ_list.append(coupQnPQ[m][klmQnPQ[m][:,0], klmQnPQ[m][:,1], klmQnPQ[m][:,2]])
            
            coupPnP_tmp, coupPnPQ_tmp, coupPQnP_tmp, coupPQnPQ_tmp = [], [], [], []
            for mp in range(nintmed):
                coupPnP_tmp.append(coupPnP[m,mp][klPnP[m][mp][:,0], klPnP[m][mp][:,1]])
                coupPnPQ_tmp.append(coupPnPQ[m,mp][klmPnPQ[m][mp][:,0], klmPnPQ[m][mp][:,1], klmPnPQ[m][mp][:,2]])
                coupPQnP_tmp.append(coupPQnP[m,mp][klmPQnP[m][mp][:,0], klmPQnP[m][mp][:,1], klmPQnP[m][mp][:,2]])
                coupPQnPQ_tmp.append(coupPQnPQ[m,mp][klmnPQnPQ[m][mp][:,0], klmnPQnPQ[m][mp][:,1], klmnPQnPQ[m][mp][:,2], klmnPQnPQ[m][mp][:,3]])
                
            coupPnP_list.append(coupPnP_tmp)
            coupPnPQ_list.append(coupPnPQ_tmp)
            coupPQnP_list.append(coupPQnP_tmp)
            coupPQnPQ_list.append(coupPQnPQ_tmp)
 
        coup_dict._nP   = coup_nP_list        
        coup_dict._nPQ  = coup_nPQ_list          
        coup_dict.QnP   = coupQnP_list          
        coup_dict.QnPQ  = coupQnPQ_list           
        coup_dict.PnP   = coupPnP_list            
        coup_dict.PnPQ  = coupPnPQ_list           
        coup_dict.PQnP  = coupPQnP_list           
        coup_dict.PQnPQ = coupPQnPQ_list           
        idx_dict._nP    = k_nP
        idx_dict._nPQ   = kl_nPQ
        idx_dict.QnP    = klQnP 
        idx_dict.QnPQ   = klmQnPQ
        idx_dict.PnP    = klPnP 
        idx_dict.PnPQ   = klmPnPQ
        idx_dict.PQnP   = klmPQnP
        idx_dict.PQnPQ  = klmnPQnPQ

del coup0, coup_nQ, coupQnQ
del k_nQ, klQnQ
if order > 1:
    del coup_nP, coup_nPQ, coupQnP, coupQnPQ
    del coupPnP, coupPnPQ, coupPQnP, coupPQnPQ
    del coup_nP_list, coup_nPQ_list, coupQnP_list, coupQnPQ_list
    del coupPnP_tmp, coupPnPQ_tmp, coupPQnP_tmp, coupPQnPQ_tmp
    del coupPnP_list, coupPnPQ_list, coupPQnP_list, coupPQnPQ_list
    del k_nP, kl_nPQ, klQnP, klmQnPQ, klPnP, klmPnPQ, klmPQnP, klmnPQnPQ

#%%
'''
Plot the sum of SOC derivatives (soc1_if + soc1_intmed)
'''
import matplotlib.pyplot as plt
sum_of_soc1 = abs(soc1_if)
sum_of_soc1 = sum_of_soc1/np.max(sum_of_soc1)
max_sum_idx = np.where(sum_of_soc1 == np.max(sum_of_soc1))[0][0] # Normal mode that gives the maximum sum of SOCs
plt.vlines(wno_f[6:] * scale, 0, sum_of_soc1, color='gray', linestyle='-')
plt.plot(wno_f[6:] * scale, sum_of_soc1, linestyle='None', marker='o', color='k')
plt.plot(wno_f[6+max_sum_idx] * scale, sum_of_soc1[max_sum_idx], linestyle='None', marker='o', color='r')
plt.ylabel(r'|$\del$SOC($S_1$-$T_1$)/$\del Q_i$ + \del$SOC($S_1$-$T_2$)/$\del Q_i$|')
plt.xlabel(r'Frequency of mode $i$ [cm$^{-1}$]')
plt.show()


"""
Compute rho0

A sequence that evaluates 'prefactor' for all time points in advance
so that discontinuities associated with the square root of complex numbers can be
handled independently as an overhead. 
"""  
#prefac_array = np.zeros(len(t_all), dtype=complex)
#expterm_array = np.zeros(len(t_all), dtype=complex)
rho0 = [get_rho0(tgrid[i], invT, omg_S_d) for i in range(nsec)]
gauss_factor = [np.exp(-0.5 * (width_au * np.array(tgrid[i]))**2) for i in range(nsec)] # Gaussian envelope
exp_factor   = [np.exp(1.0j * (Ei-Ef) * np.array(tgrid[i])) for i in range(nsec)]

for i in range(nsec):
    plt.plot(tgrid[i], np.real(rho0[i]*gauss_factor[i]*exp_factor[i]), label='Re')
    plt.plot(tgrid[i], np.imag(rho0[i]*gauss_factor[i]*exp_factor[i]), label='Im')
    plt.xlim(tgrid[i][0], tgrid[i][-1])
    plt.hlines(0.0, tgrid[i][0], tgrid[i][-1], color='k', linestyles='--')
    plt.legend(loc='upper right')
    plt.show()



"""
Distribute the time points over the designated number of nodes and cpus
"""
tlist = []
for sec in range(nsec):
    ti, tf = tlim[sec], tlim[sec+1]
    tlist_tmp = []
    seg_length = int((tf-ti)/nseg/tstep[sec]) # number of steps in one segment
    tchunk = [tgrid[sec][i*seg_length:(i+1)*seg_length] for i in range(nseg)]
    for ichunk in range(len(tchunk)):
        tchunk_paral = [tchunk[ichunk][i::nodes] for i in range(nodes)] #to distribute over nnodes nodes
        tchunk_paral = [tchunk_paral[node_rank-1][i::ppn] for i in range(ppn)] #to distribute over ppn processors
        tlist_tmp.append(tchunk_paral)
    tlist.append(tlist_tmp)

# tlist = [t_all[i::nodes] for i in range(nodes)] #to distribute over nnodes nodes
# tlist = [tlist[node_rank-1][i::ppn] for i in range(ppn)] #to distribute over ppn processors

#%%
''' Main program '''
if __name__ == '__main__':
    # form the lists of segmented integrals
    if process_id[0,0] == 2: # If the overall process is ISC
        k0_0_segs, k0_nQ_segs, k0QnQ_segs = np.zeros(nseg), np.zeros(nseg), np.zeros(nseg)
        if order > 1:
            k1_nP_segs   = np.zeros((nintmed, nseg), dtype=complex)
            k1_nPQ_segs  = np.zeros((nintmed, nseg), dtype=complex)
            k1QnP_segs   = np.zeros((nintmed, nseg), dtype=complex)
            k1QnPQ_segs  = np.zeros((nintmed, nseg), dtype=complex)
            k2PnP_segs   = np.zeros((nintmed, nintmed, nseg))
            k2PnPQ_segs  = np.zeros((nintmed, nintmed, nseg))
            k2PQnP_segs  = np.zeros((nintmed, nintmed, nseg))
            k2PQnPQ_segs = np.zeros((nintmed, nintmed, nseg))
    elif process_id[0,0] == 1: # If the overall process is IC
        k0PnP_segs = np.zeros(nseg)
#        if order > 1:
#            ... 

    terminate = 0 # routine termination flag

    ##### Parallelization #####
    p = Pool(ppn)
    if process_id[0,0] == 2: # If the overall process is ISC
        k0_0 = 0.0
        if idx_dict._nQ is not None:
            k0_nQ = np.zeros(len(idx_dict._nQ))
        else:
            k0_nQ = 0.0 
        if idx_dict.QnQ is not None:
            k0QnQ = np.zeros(len(idx_dict.QnQ))
        else:
            k0QnQ = 0.0 
        
        if order > 1:            
            k1_nP   = [np.zeros(len(idx_dict._nP[m1]), dtype=complex) for m1 in range(nintmed)]
            k1_nPQ  = [np.zeros(len(idx_dict._nPQ[m1]), dtype=complex) for m1 in range(nintmed)]
            k1QnP   = [np.zeros(len(idx_dict.QnP[m1]), dtype=complex) for m1 in range(nintmed)]
            k1QnPQ  = [np.zeros(len(idx_dict.QnPQ[m1]), dtype=complex) for m1 in range(nintmed)]
            k2PnP   = [[np.zeros(len(idx_dict.PnP[m1][m2])) for m2 in range(nintmed)] for m1 in range(nintmed)]
            k2PnPQ  = [[np.zeros(len(idx_dict.PnPQ[m1][m2])) for m2 in range(nintmed)] for m1 in range(nintmed)]
            k2PQnP  = [[np.zeros(len(idx_dict.PQnP[m1][m2])) for m2 in range(nintmed)] for m1 in range(nintmed)]
            k2PQnPQ = [[np.zeros(len(idx_dict.PQnPQ[m1][m2])) for m2 in range(nintmed)] for m1 in range(nintmed)]
    elif process_id[0,0] == 1: # If the overall process is IC
        k0PnP = np.zeros(len(idx_dict.PnP))
#        if order > 1:
#            ...

    err_to_int = [[[],[],[]] for i in range(nsec)]
    for sec in range(nsec):
        section_time = time.time()
        ti, tf = tlim[sec], tlim[sec+1]
        integrate_CF2 = partial(integrate_CF, order, rho0[sec], width_au, tgrid[sec], ti, tf, tidx_max[sec], tstep[sec], invT, omg_S_d, process_id, idx_dict, coup_dict)
    
        for iseg in range(nseg):
            ''' result[cpus][kinds], where kinds can be out_package, QQ_storage, or intQQ_storage '''
            result = p.map(integrate_CF2, tlist[sec][iseg]) # Parallelization mapping
            
            if process_id[0,0] == 2: # If the overall process is ISC
                k0_0_tmp, k0_nQ_tmp, k0QnQ_tmp = np.zeros_like(k0_0), np.zeros_like(k0_nQ), np.zeros_like(k0QnQ)
                for ippn in range(ppn):
                    k0_0_tmp  += sign_sec[sec] * result[ippn][0][0][0] 
                    k0_nQ_tmp += sign_sec[sec] * result[ippn][0][0][1] 
                    k0QnQ_tmp += sign_sec[sec] * result[ippn][0][0][2]
                k0_0_segs[iseg]  = k0_0_tmp
                k0_nQ_segs[iseg] = np.sum(k0_nQ_tmp)
                k0QnQ_segs[iseg] = np.sum(k0QnQ_tmp)
                k0_0  += k0_0_tmp
                k0_nQ += k0_nQ_tmp
                k0QnQ += k0QnQ_tmp
                if order > 1:
                    for m1 in range(nintmed):
                        k1_nP_tmp, k1_nPQ_tmp = np.zeros_like(k1_nP[m1], dtype=complex), np.zeros_like(k1_nPQ[m1])
                        k1QnPQ_tmp, k1QnP_tmp = np.zeros_like(k1QnPQ[m1]), np.zeros_like(k1QnP[m1]) 
                        for ippn in range(ppn):
                            k1_nP_tmp  += sign_sec[sec] * result[ippn][0][1][0][m1]
                            k1_nPQ_tmp += sign_sec[sec] * result[ippn][0][1][1][m1]
                            k1QnP_tmp  += sign_sec[sec] * result[ippn][0][1][2][m1]
                            k1QnPQ_tmp += sign_sec[sec] * result[ippn][0][1][3][m1] 
                        k1_nP_segs[m1, iseg]  = np.sum(k1_nP_tmp)
                        k1_nPQ_segs[m1, iseg] = np.sum(k1_nPQ_tmp)
                        k1QnP_segs[m1, iseg]  = np.sum(k1QnP_tmp)
                        k1QnPQ_segs[m1, iseg] = np.sum(k1QnPQ_tmp)
                        k1_nP[m1]  += k1_nP_tmp
                        k1_nPQ[m1] += k1_nPQ_tmp
                        k1QnP[m1]  += k1QnP_tmp
                        k1QnPQ[m1] += k1QnPQ_tmp
                    for m1 in range(nintmed):
                        for m2 in range(nintmed):
                            k2PnP_tmp, k2PnPQ_tmp = np.zeros_like(k2PnP[m1][m2]), np.zeros_like(k2PnPQ[m1][m2])
                            k2PQnP_tmp, k2PQnPQ_tmp = np.zeros_like(k2PQnP[m1][m2]), np.zeros_like(k2PQnPQ[m1][m2]) 
                            for ippn in range(ppn):
                                k2PnP_tmp   += sign_sec[sec] * result[ippn][0][2][0][m1][m2]
                                k2PnPQ_tmp  += sign_sec[sec] * result[ippn][0][2][1][m1][m2] 
                                k2PQnP_tmp  += sign_sec[sec] * result[ippn][0][2][2][m1][m2] 
                                k2PQnPQ_tmp += sign_sec[sec] * result[ippn][0][2][3][m1][m2] 
                            k2PnP_segs[m1, m2, iseg]   = np.sum(k2PnP_tmp)
                            k2PnPQ_segs[m1, m2, iseg]  = np.sum(k2PnPQ_tmp)
                            k2PQnP_segs[m1, m2, iseg]  = np.sum(k2PQnP_tmp) 
                            k2PQnPQ_segs[m1, m2, iseg] = np.sum(k2PQnPQ_tmp)
                            k2PnP[m1][m2]   += k2PnP_tmp
                            k2PnPQ[m1][m2]  += k2PnPQ_tmp
                            k2PQnP[m1][m2]  += k2PQnP_tmp
                            k2PQnPQ[m1][m2] += k2PQnPQ_tmp
            elif process_id[0,0] == 1: # If the overall process is IC
                k0PnP_tmp = np.zeros_like(result[0][0][0][0])
                for ippn in range(ppn):
                    k0PnP_tmp += sign_sec[sec] * result[ippn][0][0][0]
                k0PnP_segs[iseg]   = np.sum(k0PnP_tmp)
                k0PnP += k0PnP_tmp
        #        if order > 1:
        #            ... 
     
        segmented_rates = []
        if process_id[0,0] == 2:
            segmented_rates.append(k0_0_segs)
            segmented_rates.append(k0_nQ_segs)
            segmented_rates.append(k0QnQ_segs)
            if order > 1:
                """Exclude k1 CF integrals since their real part is null"""
                # for m1 in range(nintmed):
                #     segmented_rates.append(k1_nP_segs[m1])
                #     segmented_rates.append(k1_nPQ_segs[m1])
                #     segmented_rates.append(k1QnP_segs[m1])
                #     segmented_rates.append(k1QnPQ_segs[m1])
                for m1 in range(nintmed):
                    for m2 in range(nintmed):
                        segmented_rates.append(k2PnP_segs[m1,m2])
                        segmented_rates.append(k2PnPQ_segs[m1,m2])
                        segmented_rates.append(k2PQnP_segs[m1,m2])
                        segmented_rates.append(k2PQnPQ_segs[m1,m2])
        
        '''
        Numerical integration error estimate for k0 CFs
        '''
        CF_labels = ['CF0', 'CF_nQ', 'CFQQ']
        CF = [result[0][1][0], result[0][1][1], result[0][1][2]]
        intgls = [result[0][2][0], result[0][2][1], result[0][2][2]]
        if ppn > 1:
            for i in range(ppn-1):
                CF[0]     += result[i+1][1][0]
                CF[1]     += result[i+1][1][1]
                CF[2]     += result[i+1][1][2]
                intgls[0] += result[i+1][2][0]
                intgls[1] += result[i+1][2][1]
                intgls[2] += result[i+1][2][2]
        
        # Numerical integration error estimate
        if integrator == 'trapezoidal':
            error_estimator = error_estimate_trapezoidal
        elif integrator == 'simpson':
            error_estimator = error_estimate_simpson
        elif integrator == 'boole':
            error_estimator = error_estimate_boole
            
        for cf_idx in range(3):
            err = error_estimator(tf, ti, sec, CF[cf_idx])
            
            # Compute the ratio of errors to integrals
            err = err/intgls[cf_idx]
            err_to_int[sec][cf_idx].append(err)
            print('\nError from the section {:d} of {:s}: {:<.3e}'.format(sec+1, CF_labels[cf_idx], np.max(abs(err))))
            if np.max(abs(err)) > 0.1:
                print('!!!The error of integration is not satisfatory!!!')
                #print('Quitting.')
                #terminate = 1
                #break
            else:
                print('The error of integration is within tolerance.')
                
                
        '''
        Print out time spent on calculating the rate from each section of TVCFs
        '''
        total_section_time = time.time()-section_time
        totald, totalhr, totalmin = 0, 0, 0
        if total_section_time >= 86400.0:
            totald   = int(total_section_time/86400)
            totalhr  = int((total_section_time-totald*86400)/3600)
            totalmin = int((total_section_time-totald*86400-totalhr*3600)/60)
        elif total_section_time > 3600.0 and total_section_time < 86400.0:
            totald   = 0
            totalhr  = int(total_section_time/3600)    
            totalmin = int((total_section_time-totalhr*3600)/60)
        else:
            totald  = 0
            totalhr = 0
            if total_section_time > 60:
                totalmin = int(total_section_time/60)
            else:
                totalmin = 0
        print('Section {:>1d} '.format(sec+1) 
              + 'computation time = {:>3d} D {:>3d} hr {:>2d} min {:>4.1f} sec \n'.format(totald, totalhr, totalmin, total_section_time-totald*86400-totalhr*3600-totalmin*60))
        

    
    #del result
    if process_id[0,0] == 2:
        if order > 1:
            del k1_nP_tmp, k1_nPQ_tmp, k1QnPQ_tmp, k1QnP_tmp
            del k2PnP_tmp, k2PnPQ_tmp, k2PQnP_tmp, k2PQnPQ_tmp
    
    p.close()
    p.join()
    ##### End of parallelization #####         

    if terminate == 0:
        if process_id[0,0] == 2: # If the overall process is ISC
            k0 = k0_0 + np.sum(k0_nQ) + np.sum(k0QnQ)
            overall_rate = k0
            if order > 1:
                tot_k1_nP_interm = np.zeros(nintmed, dtype=complex)
                tot_k1_nPQ_interm = np.zeros(nintmed, dtype=complex)
                tot_k1QnP_interm = np.zeros(nintmed, dtype=complex)
                tot_k1QnPQ_interm = np.zeros(nintmed, dtype=complex)
                tot_k2PnP_interm = np.zeros((nintmed, nintmed))
                tot_k2PnPQ_interm = np.zeros((nintmed, nintmed))
                tot_k2PQnP_interm = np.zeros((nintmed, nintmed))
                tot_k2PQnPQ_interm = np.zeros((nintmed, nintmed))
                for m1 in range(nintmed):
                    tot_k1_nP_interm[m1]  = np.sum(k1_nP[m1])
                    tot_k1_nPQ_interm[m1] = np.sum(k1_nPQ[m1])
                    tot_k1QnP_interm[m1]  = np.sum(k1QnP[m1])
                    tot_k1QnPQ_interm[m1] = np.sum(k1QnPQ[m1])
                    for m2 in range(nintmed):
                        tot_k2PnP_interm[m1,m2]   = np.sum(k2PnP[m1][m2])
                        tot_k2PnPQ_interm[m1,m2]  = np.sum(k2PnPQ[m1][m2])
                        tot_k2PQnP_interm[m1,m2]  = np.sum(k2PQnP[m1][m2])
                        tot_k2PQnPQ_interm[m1,m2] = np.sum(k2PQnPQ[m1][m2])
                k1 = np.sum(tot_k1_nP_interm) + np.sum(tot_k1_nPQ_interm) + np.sum(tot_k1QnP_interm) + np.sum(tot_k1QnPQ_interm)
                #k1 = np.real(k1) # Need only the real part for k1
                k2 = np.sum(tot_k2PnP_interm) + np.sum(tot_k2PnPQ_interm) + np.sum(tot_k2PQnP_interm) + np.sum(tot_k2PQnPQ_interm)
                overall_rate = k0 + k1 + k2
                if np.real(overall_rate) < 0.0:
                    overall_rate_sign = -1
            
            ''' Convergence check '''
            if nseg > 1:
                import matplotlib as mpl
                import matplotlib.pyplot as plt
                import matplotlib.gridspec as gridspec
                from matplotlib.font_manager import FontProperties
                mpl.rcParams.update(mpl.rcParamsDefault)
                mpl.rcParams['mathtext.rm'] = 'Times'
                mpl.rcParams['mathtext.it'] = 'Times:italic'
                mpl.rcParams['mathtext.default'] = 'it'
                mpl.rcParams['mathtext.fontset'] = 'cm'
                nvert  = 2 # number of plots along the vertical axis
                nhoriz = 4 # number of plots along the horizontal axis
    
                # Prepare a grid of plots
                pidx = 0
                plt.figure(figsize=(35, 15))
                gs1 = gridspec.GridSpec(nvert, nhoriz)
                gs1.update(wspace=0.22, hspace=0.10) # set the spacing between axes.
                for j in range(len(segmented_rates)+1):
                    if j != 3:
                        ax1 = plt.subplot(gs1[j])
                        plt.axis('on')
                        #ax1.set_aspect('equal')
                        
                        ### Plot the integrations of k0 and k2 CFs (no k1 since its real part is null) ###
                        conv = []
                        seg_rate = segmented_rates[pidx][0]
                        conv.append(seg_rate)
                        for i in range(nseg-1):
                            seg_rate += segmented_rates[pidx][i+1]
                            conv.append(seg_rate)
                        ax1.plot(np.arange(1,nseg+1,1), np.array(conv)*(autime2s**-1)*overall_rate_sign, linestyle='-', marker='o', color='b')
        
                        # Compute moving average
                        moving_ave = []
                        for i in range(nseg-10):
                            moving_ave.append(np.mean(conv[i:i+10]))
                        ax1.plot([i+10+1 for i in range(nseg-10)], np.array(moving_ave)*(autime2s**-1)*overall_rate_sign, linestyle='', marker='o', color='r')
        
                        #ax1.hlines(0.0, 0, nseg+1, color='k', linestyle='--')
                        plt.xlim(0, nseg)
                        if j in np.arange(len(segmented_rates)-nhoriz+1, len(segmented_rates)+1, 1):
                            plt.xlabel('Time [a.u.]', fontfamily='Times', fontsize=40)
                            ax1.set_xticks(np.arange(0, nseg+1, nseg/5), 
                                           ['{:.0f}'.format(int(((tlim[-1]-tlim[0])/nseg)*i)) for i in np.arange(0, nseg+1, nseg/5)], 
                                           fontsize=32, fontfamily='Times')
                        else:
                            ax1.set_xticks([], labels=None)
                            
                        if j in np.arange(0, len(segmented_rates), nhoriz):
                            plt.ylabel(r'$k$ [$s^{-1}$]', fontfamily='Times', fontsize=40)
                        ax1.ticklabel_format(axis='y', style='sci', scilimits=(0,0), useMathText=True)
                        custom_font = FontProperties(family='Times', style='italic')
                        ax1.yaxis.get_offset_text().set_fontproperties(custom_font)
                        ax1.yaxis.get_offset_text().set_fontsize(32)
                        for tick in ax1.get_yticklabels():
                            tick.set_fontname("Times")
                            tick.set_fontsize(32)
                            
                        pidx += 1
                        
                plt.savefig(out_path + 'conv_check.png', dpi=600)
                plt.show()
            
        elif process_id[0,0] == 1: # If the overall process is IC
            k0 = np.sum(k0PnP)
            overall_rate = k0 
            if np.real(overall_rate) < 0.0:
                overall_rate_sign = -1
            
            ''' Convergence check '''
            # if nseg > 1:
            #     conv = []
            #     total_rate = k0PnP_segs[0]
            #     conv.append(total_rate)
            #     for i in range(nseg-1):
            #         total_rate += k0PnP_segs[i+1]
            #         conv.append(total_rate)
            #     plt.plot(abs(np.array(conv)), linestyle='-', marker='o')
        
            #     # Compute moving average
            #     moving_ave = []
            #     for i in range(nseg-20):
            #         moving_ave.append(np.mean(conv[i:i+20]))
            #     plt.plot([i+20 for i in range(nseg-20)], abs(np.array(moving_ave)), linestyle='', marker='o', color='r')
        
            #     plt.hlines(0.0, 0, nseg, color='k', linestyle='--')
            #     plt.xlim(0, nseg)
            #     plt.xlabel('Total number of segments')
            #     plt.ylabel('k')
            #     plt.show()
            
    #        if order > 1:
    #            ...
    
    
        ##### Writing output files #####   
        ts_format = nsec*'ts{:<.3f}_'
        tm_format = (nsec-1)*'tm{:<.1f}_'
        fname = 'rate_' + ts_format.format(*tstep) + 'ti{:<.1f}_'.format(tlim[0]) + tm_format.format(*tlim[1:-1]) + 'tf{:<.1f}_ccut{:<d}_{:s}.out'.format(tlim[-1],coup_cutoff_1,integrator)
        with open(out_path + fname, 'w+') as f:
            f.write('---------------- \n')
            f.write('Computation info \n')
            f.write('---------------- \n')
            f.write('\n')
            f.write('Number of processors: %d \n' %(ppn))
            f.write('Initial state geometry file: %s \n' %(fgeo_file))
            f.write('Final state geometry file: %s \n' %(fgeo_file))
            f.write('Initial state hessian file: %s \n' %(ifreq_file))
            f.write('Final state hessian file: %s \n' %(ffreq_file))
            f.write('AMU file: %s \n' %(mass_file))
    #        f.write('NACME file: %s \n' %(nac_ini_fin_file))
    #        f.write('SOCME file: %s \n' %(socme_file))
            f.write('Simulation temperature [K]: %d \n' %(simtemp))
            f.write('The width of Gaussian envelope: {:>.3f} cm-1 \n'.format(width))
            f.write('Perturbative order: %d\n' %(order))
            if process_id[0,0] == 1:
                f.write('1st order process: IC \n')
            elif process_id[0,0] == 2:
                f.write('1st order process: ISC \n')
            f.write('\n')
            f.write('Coupling cutoff for 1-index CF: {:d} \n'.format(coup_cutoff_1))
            f.write('Coupling cutoff for 2-index CF: {:d} \n'.format(coup_cutoff_2))
            f.write('Coupling cutoff for 3-index CF: {:d} \n'.format(coup_cutoff_3))
            f.write('Coupling cutoff for 4-index CF: {:d} \n'.format(coup_cutoff_4))
            f.write('\n')
            #f.write('Initial time: {:>.1f} \n'.format(ti))
            #f.write('Final time = {:>.1f} \n'.format(tfinal))
            #f.write('Time step: {:>.2f} \n'.format(tstep))
            totaltime = time.time()-start_time
            totald, totalhr, totalmin = 0, 0, 0
            if totaltime >= 86400.0:
                totald   = int(totaltime/86400)
                totalhr  = int((totaltime-totald*86400)/3600)
                totalmin = int((totaltime-totald*86400-totalhr*3600)/60)
            elif totaltime > 3600.0 and totaltime < 86400.0:
                totald   = 0
                totalhr  = int(totaltime/3600)    
                totalmin = int((totaltime-totalhr*3600)/60)
            else:
                totald  = 0
                totalhr = 0
                if totaltime > 60:
                    totalmin = int(totaltime/60)
                else:
                    totalmin = 0
            f.write('Total computation time = {:>3d} D {:>3d} hr {:>2d} min {:>4.1f} sec \n'.format(totald, totalhr, totalmin, totaltime-totald*86400-totalhr*3600-totalmin*60))
            print('Total computation time = {:>3d} D {:>3d} hr {:>2d} min {:>4.1f} sec \n'.format(totald, totalhr, totalmin, totaltime-totald*86400-totalhr*3600-totalmin*60))
            f.write('Exit time: %s \n' %(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
            f.write('\n')
    
            f.write('\n')
            f.write('------------------ \n')
            f.write('Computation result \n')
            f.write('------------------ \n')
            f.write('\n')
            f.write('Total rate [1/s]: {:>.4e} \n'.format(overall_rate * (autime2s**-1) * overall_rate_sign))
            print('Total rate [1/s]: {:>.4e} \n'.format(overall_rate * (autime2s**-1) * overall_rate_sign))
            f.write('\n')
            f.write('Order-wise analysis \n')
            f.write('------------------- \n')
            f.write('0th-order rate [1/s]: {:>.4e} \n'.format(k0 * (autime2s**-1) * overall_rate_sign))
            f.write('0th-order rate breakdown: \n')
            if process_id[0,0] == 1:
                f.write('k0_rhoPnP : {:>.4e} \n'.format(k0 * (autime2s**-1) * overall_rate_sign))
                f.write('\n')
    #            if order > 1:
    #                ...
            elif process_id[0,0] == 2:
                f.write('k0_rho0      : {:>.4e} \n'.format(k0_0 * (autime2s**-1) * overall_rate_sign))
                f.write('Integration Error-to-Integral ratio for CF0: \n')
                for sec in range(nsec):
                    f.write('Section {:<d} (Time Step = {:<f}): Max = {:>.4e}, Ave = {:>.4e} \n'.format(sec+1, float(tstep[sec]), np.max(np.abs(err_to_int[sec][0])), np.mean(np.abs(err_to_int[sec][0]))))
                f.write('k0_rho_nQf   : {:>.4e} \n'.format(np.sum(k0_nQ) * (autime2s**-1) * overall_rate_sign))
                f.write('Integration Error-to-Integral ratio for Q CF: \n')
                for sec in range(nsec):
                    f.write('Section {:<d} (Time Step = {:<f}): Max = {:>.4e}, Ave = {:>.4e} \n'.format(sec+1, float(tstep[sec]), np.max(np.abs(err_to_int[sec][1])), np.mean(np.abs(err_to_int[sec][1]))))
                f.write('k0_rho_QfnQf : {:>.4e} \n'.format(np.sum(k0QnQ) * (autime2s**-1) * overall_rate_sign))
                f.write('Integration Error-to-Integral ratio for Qf-Qf CF: \n')
                for sec in range(nsec):
                    f.write('Section {:<d} (Time Step = {:<f}): Max = {:>.4e}, Ave = {:>.4e} \n'.format(sec+1, float(tstep[sec]), np.max(np.abs(err_to_int[sec][2])), np.mean(np.abs(err_to_int[sec][2]))))
                f.write('\n')
                if order > 1:
                    f.write('1st-order rate [1/s]: {:>.4e} + {:>.4e}i \n'.format(np.real(k1*(autime2s**-1)*overall_rate_sign), np.imag(k1*(autime2s**-1)*overall_rate_sign)))
                    f.write('1st-order rate breakdown: \n')
                    for m1 in range(nintmed):
                        if process_id[m1+1,0] == 1: # IC-->ISC
                            f.write('k1_rho_nPi     : {:>.4e} + {:>.4e}i \n'.format(np.real(tot_k1_nP_interm[m1]*(autime2s**-1)*overall_rate_sign), np.imag(tot_k1_nP_interm[m1]*(autime2s**-1)*overall_rate_sign)))
                            f.write('k1_rho_nPiQf   : {:>.4e} \n'.format(tot_k1_nPQ_interm[m1]*(autime2s**-1)*overall_rate_sign))
                            f.write('k1_rho_QinPi   : {:>.4e} \n'.format(tot_k1QnP_interm[m1]*(autime2s**-1)*overall_rate_sign))
                            f.write('k1_rho_QinPiQf : {:>.4e} \n'.format(tot_k1QnPQ_interm[m1]*(autime2s**-1)*overall_rate_sign))
                        elif process_id[m1+1,0] == 2: # ISC-->IC
                            f.write('k1_rho_nPf     : {:>.4e} \n'.format(tot_k1_nP_interm[m1]*(autime2s**-1)*overall_rate_sign))
                            f.write('k1_rho_nQiPf   : {:>.4e} \n'.format(tot_k1_nPQ_interm[m1]*(autime2s**-1)*overall_rate_sign))
                            f.write('k1_rho_QinPf   : {:>.4e} \n'.format(tot_k1QnP_interm[m1]*(autime2s**-1)*overall_rate_sign))
                            f.write('k1_rho_QinQiPf : {:>.4e} \n'.format(tot_k1QnPQ_interm[m1]*(autime2s**-1)*overall_rate_sign))
                    f.write('\n')
                    f.write('2nd-order rate [1/s]: {:>.4e} \n'.format(k2 * (autime2s**-1) * overall_rate_sign))
                    f.write('2nd-order rate breakdown: \n')
                    for m1 in range(nintmed):
                        for m2 in range(nintmed):
                            if process_id[m1+1,0] == 1 and process_id[m2+1,0] == 1: # IC-->ISC for both m1 and m2
                                f.write('k2_rho_PinPi     : {:>.4e} \n'.format(tot_k2PnP_interm[m1][m2]*(autime2s**-1)*overall_rate_sign))
                                f.write('k2_rho_PinPiQf   : {:>.4e} \n'.format(tot_k2PnPQ_interm[m1][m2]*(autime2s**-1)*overall_rate_sign))
                                f.write('k2_rho_QfPinPi   : {:>.4e} \n'.format(tot_k2PQnP_interm[m1][m2]*(autime2s**-1)*overall_rate_sign))
                                f.write('k2_rho_QfPinPiQf : {:>.4e} \n'.format(tot_k2PQnPQ_interm[m1][m2]*(autime2s**-1)*overall_rate_sign))
                            elif process_id[m1+1,0] == 1 and process_id[m2+1,0] == 2: # IC-->ISC for m1 and ISC-->IC for m2
                                f.write('k2_rho_PinPf     : {:>.4e} \n'.format(tot_k2PnP_interm[m1][m2]*(autime2s**-1)*overall_rate_sign))
                                f.write('k2_rho_PinQiPf   : {:>.4e} \n'.format(tot_k2PnPQ_interm[m1][m2]*(autime2s**-1)*overall_rate_sign))
                                f.write('k2_rho_QfPinPf   : {:>.4e} \n'.format(tot_k2PQnP_interm[m1][m2]*(autime2s**-1)*overall_rate_sign))
                                f.write('k2_rho_QfPinQiPf : {:>.4e} \n'.format(tot_k2PQnPQ_interm[m1][m2]*(autime2s**-1)*overall_rate_sign))
                            elif process_id[m1+1,0] == 2 and process_id[m2+1,0] == 1: # ISC-->IC for m1 and IC-->ISC for m2
                                f.write('k2_rho_PfnPi     : {:>.4e} \n'.format(tot_k2PnP_interm[m1][m2]*(autime2s**-1)*overall_rate_sign))
                                f.write('k2_rho_PfnPiQf   : {:>.4e} \n'.format(tot_k2PnPQ_interm[m1][m2]*(autime2s**-1)*overall_rate_sign))
                                f.write('k2_rho_PfQinPi   : {:>.4e} \n'.format(tot_k2PQnP_interm[m1][m2]*(autime2s**-1)*overall_rate_sign))
                                f.write('k2_rho_PfQinPiQf : {:>.4e} \n'.format(tot_k2PQnPQ_interm[m1][m2]*(autime2s**-1)*overall_rate_sign))
                            elif process_id[m1+1,0] == 2 and process_id[m2+1,0] == 2: # ISC-->IC for both m1 and m2
                                f.write('k2_rho_PfnPf     : {:>.4e} \n'.format(tot_k2PnP_interm[m1][m2]*(autime2s**-1)*overall_rate_sign))
                                f.write('k2_rho_PfnQiPf   : {:>.4e} \n'.format(tot_k2PnPQ_interm[m1][m2]*(autime2s**-1)*overall_rate_sign))
                                f.write('k2_rho_PfQinPf   : {:>.4e} \n'.format(tot_k2PQnP_interm[m1][m2]*(autime2s**-1)*overall_rate_sign))
                                f.write('k2_rho_PfQinQiPf : {:>.4e} \n'.format(tot_k2PQnPQ_interm[m1][m2]*(autime2s**-1)*overall_rate_sign))
    
    
        if process_id[0,0] == 1: # If 1st order process is IC
            with open(out_path + 'integral_PnP.out', 'w+') as f:
                f.write('Number of entry: {:>8d}\n'.format(len(klPnP)))
                for i in range(len(klPnP)):
                    f.write('{:>6d}{:>6d}{:>15.4e} \n'.format(klPnP[i,0], klPnP[i,1], k0PnP[i]))
    #           if order > 1:
    #               ...
        elif process_id[0,0] == 2: # If 1st order process is ISC
            with open(out_path + 'integral_nQ.out', 'w+') as f:
                f.write('Number of entry: {:>8d}\n'.format(len(idx_dict._nQ)))
                for i in range(len(idx_dict._nQ)):
                    f.write("{:>6d}     {:>15.4e} \n".format(idx_dict._nQ[i], k0_nQ[i]*overall_rate_sign))
            with open(out_path + 'integral_QnQ.out', 'w+') as f:
                f.write('Number of entry:  {:>8d}\n'.format(len(idx_dict.QnQ)))
                for i in range(len(idx_dict.QnQ)):
                    f.write("{:>6d}{:>6d}{:>15.4e} \n".format(idx_dict.QnQ[i,0], idx_dict.QnQ[i,1], k0QnQ[i]*overall_rate_sign))
            if order > 1:
                with open(out_path + 'integral_nP.out', 'w+') as f:
                    for m1 in range(nintmed):
                        f.write('m={:>2d} m2=N/A | Number of entry: {:>8d}\n'.format(m1+1, len(idx_dict._nP[m1])))
                        for i in range(len(idx_dict._nP[m1])):
                            f.write("{:>6d}     {:>15.4e} \n".format(idx_dict._nP[m1][i], k1_nP[m1][i]*overall_rate_sign))
                with open(out_path + 'integral_nPQ.out', 'w+') as f:
                    for m in range(nintmed):
                        f.write('m={:>2d} m2=N/A | Number of entry: {:>8d}\n'.format(m+1, len(idx_dict._nPQ[m])))
                        for i in range(len(idx_dict._nPQ[m])):
                            f.write("{:>6d}{:>6d}{:>15.4e} \n".format(idx_dict._nPQ[m][i,0], idx_dict._nPQ[m][i,1], k1_nPQ[m][i]*overall_rate_sign))
                with open(out_path + 'integral_QnP.out', 'w+') as f:
                    for m in range(nintmed):
                        f.write('m={:>2d} m2=N/A | Number of entry: {:>8d}\n'.format(m+1, len(idx_dict.QnP[m])))
                        for i in range(len(idx_dict.QnP[m])):
                            f.write("{:>6d}{:>6d}{:>15.4e} \n".format(idx_dict.QnP[m][i,0], idx_dict.QnP[m][i,1], k1QnP[m][i]*overall_rate_sign))
                with open(out_path + 'integral_QnPQ.out', 'w+') as f:
                    for m in range(nintmed):
                        f.write('m={:>2d} m2=N/A | Number of entry: {:>10d}\n'.format(m+1, len(idx_dict.QnPQ[m])))
                        for i in range(len(idx_dict.QnPQ[m])):
                            f.write("{:>6d}{:>6d}{:>6d}{:>15.4e} \n".format(*idx_dict.QnPQ[m][i,:], k1QnPQ[m][i]*overall_rate_sign))
                with open(out_path + 'integral_PnP.out', 'w+') as f:
                    for m in range(nintmed):
                        for mp in range(nintmed):
                            f.write('m1={:>2d} m2={:>2d} | Number of entry: {:>8d}\n'.format(m+1, mp+1, len(idx_dict.PnP[m][mp])))
                            for i in range(len(idx_dict.PnP[m][mp])):
                                f.write("{:>6d}{:>6d}{:>15.4e} \n".format(*idx_dict.PnP[m][mp][i,:], k2PnP[m][mp][i]*overall_rate_sign))
                with open(out_path + 'integral_PnPQ.out', 'w+') as f:
                    for m in range(nintmed):
                        for mp in range(nintmed):
                            f.write('m1={:>2d} m2={:>2d} | Number of entry: {:>10d}\n'.format(m+1, mp+1, len(idx_dict.PnPQ[m][mp])))
                            for i in range(len(idx_dict.PnPQ[m][mp])):
                                f.write("{:>6d}{:>6d}{:>6d}{:>15.4e} \n".format(*idx_dict.PnPQ[m][mp][i,:], k2PnPQ[m][mp][i]*overall_rate_sign))
                with open(out_path + 'integral_PQnP.out', 'w+') as f:
                    for m in range(nintmed):
                        for mp in range(nintmed):
                            f.write('m1={:>2d} m2={:>2d} | Number of entry: {:>10d}\n'.format(m+1, mp+1, len(idx_dict.PQnP[m][mp])))
                            for i in range(len(idx_dict.PQnP[m][mp])):
                                f.write("{:>6d}{:>6d}{:>6d}{:>15.4e} \n".format(*idx_dict.PQnP[m][mp][i,:], k2PQnP[m][mp][i]*overall_rate_sign))
                with open(out_path + 'integral_PQnPQ.out', 'w+') as f:
                    for m in range(nintmed):
                        for mp in range(nintmed):
                            f.write('m1={:>2d} m2={:>2d} | Number of entry: {:>12d}\n'.format(m+1, mp+1, len(idx_dict.PQnPQ[m][mp])))
                            for i in range(len(idx_dict.PQnPQ[m][mp])):
                                f.write("{:>6d}{:>6d}{:>6d}{:>6d}{:>15.4e} \n".format(*idx_dict.PQnPQ[m][mp][i,:], k2PQnPQ[m][mp][i]*overall_rate_sign))
                

#%%
''' Main program when using Romberg or tanh_sinh integration '''
# if __name__ == '__main__':

#     # Excersize main integrator
#     if integrator == 'romberg':
#         result = romberg_integration(order, omg_S_d, invT, process_id, idx_dict, coup_dict)
#     elif integrator == 'tanh_sinh':
#         result = tanh_sinh_integration(order, omg_S_d, invT, process_id, idx_dict, coup_dict)
#     if process_id[0,0] == 2: # If the overall process is ISC
#         k0QnQ = result[0][0][0]
#     err_est = result[1]
        
#     del result
#     if process_id[0,0] == 2:
#         if order > 1:
#             del k1_nP_tmp, k1_nPQ_tmp, k1QnPQ_tmp, k1QnP_tmp
#             del k2PnP_tmp, k2PnPQ_tmp, k2PQnP_tmp, k2PQnPQ_tmp    

#     if process_id[0,0] == 2: # If the overall process is ISC
#         k0 = np.sum(k0QnQ)
#         overall_rate = k0
    
#     ##### Writing output files #####    
#     with open(out_path + 'rate_tf{:<d}_ccut{:<d}_{:s}.out'.format(int(tfinal),coup_cutoff_1,integrator), 'w+') as f:
#         f.write('---------------- \n')
#         f.write('Computation info \n')
#         f.write('---------------- \n')
#         f.write('\n')
#         f.write('Number of processors: %d \n' %(ppn))
#         f.write('Initial state geometry file: %s \n' %(fgeo_file))
#         f.write('Final state geometry file: %s \n' %(fgeo_file))
#         f.write('Initial state hessian file: %s \n' %(ifreq_file))
#         f.write('Final state hessian file: %s \n' %(ffreq_file))
#         f.write('AMU file: %s \n' %(mass_file))
# #        f.write('NACME file: %s \n' %(nac_ini_fin_file))
# #        f.write('SOCME file: %s \n' %(socme_file))
#         f.write('Simulation temperature: %d \n' %(simtemp))
#         f.write('Perturbative order: %d\n' %(order))
#         if process_id[0,0] == 1:
#             f.write('1st order process: IC \n')
#         elif process_id[0,0] == 2:
#             f.write('1st order process: ISC \n')
#         f.write('\n')
#         f.write('Coupling cutoff for 1-index CF: {:d} \n'.format(coup_cutoff_1))
#         f.write('Coupling cutoff for 2-index CF: {:d} \n'.format(coup_cutoff_2))
#         f.write('Coupling cutoff for 3-index CF: {:d} \n'.format(coup_cutoff_3))
#         f.write('Coupling cutoff for 4-index CF: {:d} \n'.format(coup_cutoff_4))
#         f.write('\n')
#         #f.write('Initial time: {:>.1f} \n'.format(ti))
#         f.write('Final time = {:>.1f} \n'.format(tfinal))
#         totaltime = time.time()-start_time
#         totald, totalhr, totalmin = 0, 0, 0
#         if totaltime >= 86400.0:
#             totald   = int(totaltime/86400)
#             totalhr  = int((totaltime-totald*86400)/3600)
#             totalmin = int((totaltime-totald*86400-totalhr*3600)/60)
#         elif totaltime > 3600.0 and totaltime < 86400.0:
#             totald   = 0
#             totalhr  = int(totaltime/3600)    
#             totalmin = int((totaltime-totalhr*3600)/60)
#         else:
#             totald  = 0
#             totalhr = 0
#             if totaltime > 60:
#                 totalmin = int(totaltime/60)
#             else:
#                 totalmin = 0
#         f.write('Total computation time = {:>3d} D {:>3d} hr {:>2d} min {:>4.1f} sec \n'.format(totald, totalhr, totalmin, totaltime-totald*86400-totalhr*3600-totalmin*60))
#         f.write('Exit time: %s \n' %(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
#         f.write('\n')

#         f.write('\n')
#         f.write('------------------ \n')
#         f.write('Computation result \n')
#         f.write('------------------ \n')
#         f.write('\n')
#         f.write('Total rate [1/s]: {:>.4e} \n'.format(overall_rate * (autime2s**-1)))
#         f.write('\n')
#         f.write('Order-wise analysis \n')
#         f.write('------------------- \n')
#         f.write('0th-order rate [1/s]: {:>.4e} \n'.format(k0 * (autime2s**-1)))
#         f.write('0th-order rate breakdown: \n')
#         if process_id[0,0] == 1:
#             f.write('k0_rhoPnP : {:>.4e} \n'.format(k0 * (autime2s**-1)))
#             f.write('\n')
# #            if order > 1:
# #                ...
#         elif process_id[0,0] == 2:
# #            f.write('k0_rho0      : {:>.4e} \n'.format(k0_0 * (autime2s**-1)))
# #            f.write('k0_rho_nQf   : {:>.4e} \n'.format(np.sum(k0_nQ)*(autime2s**-1)))
#             f.write('k0_rho_QfnQf : {:>.4e} \n'.format(np.sum(k0QnQ)*(autime2s**-1)))
#             f.write('Integration error estimate: {:>.4e} \n'.format(err_est))
#             f.write('\n')
#             # if order > 1:
#             #     f.write('1st-order rate [1/s]: {:>.4e} \n'.format(k1 * (autime2s**-1)))
#             #     f.write('1st-order rate breakdown: \n')
#             #     for m1 in range(nintmed):
#             #         if process_id[m1+1,0] == 1: # IC-->ISC
#             #             f.write('k1_rho_nPi     : {:>.4e} \n'.format(tot_k1_nP_interm[m1]*(autime2s**-1)))
#             #             f.write('k1_rho_nPiQf   : {:>.4e} \n'.format(tot_k1_nPQ_interm[m1]*(autime2s**-1)))
#             #             f.write('k1_rho_QinPi   : {:>.4e} \n'.format(tot_k1QnP_interm[m1]*(autime2s**-1)))
#             #             f.write('k1_rho_QinPiQf : {:>.4e} \n'.format(tot_k1QnPQ_interm[m1]*(autime2s**-1)))
#             #         elif process_id[m1+1,0] == 2: # ISC-->IC
#             #             f.write('k1_rho_nPf     : {:>.4e} \n'.format(tot_k1_nP_interm[m1]*(autime2s**-1)))
#             #             f.write('k1_rho_nQiPf   : {:>.4e} \n'.format(tot_k1_nPQ_interm[m1]*(autime2s**-1)))
#             #             f.write('k1_rho_QinPf   : {:>.4e} \n'.format(tot_k1QnP_interm[m1]*(autime2s**-1)))
#             #             f.write('k1_rho_QinQiPf : {:>.4e} \n'.format(tot_k1QnPQ_interm[m1]*(autime2s**-1)))
#             #     f.write('\n')
#             #     f.write('2nd-order rate [1/s]: {:>.4e} \n'.format(k2 * (autime2s**-1)))
#             #     f.write('2nd-order rate breakdown: \n')
#             #     for m1 in range(nintmed):
#             #         for m2 in range(nintmed):
#             #             if process_id[m1+1,0] == 1 and process_id[m1+1,0] == 1: # IC-->ISC for both m1 and m2
#             #                 f.write('k2_rho_PinPi     : {:>.4e} \n'.format(tot_k2PnP_interm[m1][m2]*(autime2s**-1)))
#             #                 f.write('k2_rho_PinPiQf   : {:>.4e} \n'.format(tot_k2PnPQ_interm[m1][m2]*(autime2s**-1)))
#             #                 f.write('k2_rho_QfPinPi   : {:>.4e} \n'.format(tot_k2PQnP_interm[m1][m2]*(autime2s**-1)))
#             #                 f.write('k2_rho_QfPinPiQf : {:>.4e} \n'.format(tot_k2PQnPQ_interm[m1][m2]*(autime2s**-1)))
#             #             elif process_id[m1+1,0] == 1 and process_id[m1+1,0] == 2: # IC-->ISC for m1 and ISC-->IC for m2
#             #                 f.write('k2_rho_PinPf     : {:>.4e} \n'.format(tot_k2PnP_interm[m1][m2]*(autime2s**-1)))
#             #                 f.write('k2_rho_PinQiPf   : {:>.4e} \n'.format(tot_k2PnPQ_interm[m1][m2]*(autime2s**-1)))
#             #                 f.write('k2_rho_QfPinPf   : {:>.4e} \n'.format(tot_k2PQnP_interm[m1][m2]*(autime2s**-1)))
#             #                 f.write('k2_rho_QfPinQiPf : {:>.4e} \n'.format(tot_k2PQnPQ_interm[m1][m2]*(autime2s**-1)))
#             #             elif process_id[m1+1,0] == 2 and process_id[m1+1,0] == 1: # ISC-->IC for m1 and IC-->ISC for m2
#             #                 f.write('k2_rho_PfnPi     : {:>.4e} \n'.format(tot_k2PnP_interm[m1][m2]*(autime2s**-1)))
#             #                 f.write('k2_rho_PfnPiQf   : {:>.4e} \n'.format(tot_k2PnPQ_interm[m1][m2]*(autime2s**-1)))
#             #                 f.write('k2_rho_PfQinPi   : {:>.4e} \n'.format(tot_k2PQnP_interm[m1][m2]*(autime2s**-1)))
#             #                 f.write('k2_rho_PfQinPiQf : {:>.4e} \n'.format(tot_k2PQnPQ_interm[m1][m2]*(autime2s**-1)))
#             #             elif process_id[m1+1,0] == 2 and process_id[m1+1,0] == 2: # ISC-->IC for both m1 and m2
#             #                 f.write('k2_rho_PfnPf     : {:>.4e} \n'.format(tot_k2PnP_interm[m1][m2]*(autime2s**-1)))
#             #                 f.write('k2_rho_PfnQiPf   : {:>.4e} \n'.format(tot_k2PnPQ_interm[m1][m2]*(autime2s**-1)))
#             #                 f.write('k2_rho_PfQinPf   : {:>.4e} \n'.format(tot_k2PQnP_interm[m1][m2]*(autime2s**-1)))
#             #                 f.write('k2_rho_PfQinQiPf : {:>.4e} \n'.format(tot_k2PQnPQ_interm[m1][m2]*(autime2s**-1)))

'''End of program'''
