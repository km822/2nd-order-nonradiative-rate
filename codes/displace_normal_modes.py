#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 26 15:16:48 2024

@author: kenmiyazaki
"""

import numpy as np
import os
from sys_param import *
from sim_param import *
from constants import *
from fun import *

ffreq_file = 's1_freq_tpssh_631gd_qchem'
ifreq_file = 't1_freq_tpssh_631gd_qchem'

fhess_file = 's1_hess_tpssh_631gd_qchem'
ihess_file = 't1_hess_tpssh_631gd_qchem'

#fgeo_file = 's1_geo_displaced_along_93_qchem'
fgeo_file = 's1_geo_tpssh_631gd_qchem'
igeo_file = 't1_geo_tpssh_631gd_qchem'

ndisp_modes_i = 1
ndisp_modes_f = 1
disp_size = 0.05 # Displacement ratio in unitless normal coordinates
natom = 54
#scale = 0.960 # B3LYP/6-31G(d) 
scale = 0.959 # TPSSh/6-31G* 



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
if 'qchem' in ihess_file[8:]:
    vec_i = read_hess(ihess_file)

# Convert freq in wavenumber into and a.u.
frqau_i, frqau_f = np.zeros(nvib), np.zeros(nvib)
pad = 6+Nimf
for i in range(nvib):
    frqau_i[i] += wno_i[i+pad] * 2*np.pi * light*100 * autime2s * scale
    frqau_f[i] += wno_f[i+pad] * 2*np.pi * light*100 * autime2s * scale
# frqau_i = np.concatenate(([0.0 for i in range(6)], frqau_i))
# frqau_f = np.concatenate(([0.0 for i in range(6)], frqau_f))

"""
Read AMU of each atom in the molecule
"""
masamu, atm_sym = read_atomic_amu(mass_file)
masamu_ext = np.zeros(nmode)
for iatom in range(natom):
    masamu_ext[iatom*3:(iatom+1)*3] = masamu[iatom]

"""
Calculate mass-weighted Cartesian vectors
"""
mwvec_f, prenorm_mwvec_f = get_mass_weighted_nvec(ffreq_package, vec_f, masamu)
mwvec_i, prenorm_mwvec_i = get_mass_weighted_nvec(ifreq_package, vec_i, masamu)
# if package == 1:
#     mwvec_i = mwvec_i[6:]
#     mwvec_f = mwvec_f[6:]


"""
Compute reduced masses of vibrational modes
"""
# redmas_f, redmas_i = np.zeros(nvib), np.zeros(nvib)
# for i in range(nvib):
#     redmas_i[i] = 1.0/sum((mwvec_i[i]**2)/masamu_ext)
#     redmas_f[i] = 1.0/sum((mwvec_f[i]**2)/masamu_ext)
    
"""
Read Cartesian coordinate of optimized geometries
"""
crd_f, crd_i = read_geo(file_path, fgeo_file, igeo_file)

# Convert into Bohr
crd_i *= ang2bohr
crd_f *= ang2bohr


"""
Calcuate displacement vectors
"""
crddif2 = crd_f - crd_i
crddif1 = crd_i - crd_f
mwdisp2, mwdisp1 = np.zeros(3*natom), np.zeros(3*natom)
for i in range(natom):
    mwdisp2[3*i:3*i+3] += crddif2[i] * (masamu[i]**0.5) #in bohr*amu**0.5
    mwdisp1[3*i:3*i+3] += crddif1[i] * (masamu[i]**0.5) #in bohr*amu**0.5

d2 = np.matmul(mwvec_i, mwdisp2) # 3N dim vector
d1 = np.matmul(mwvec_f, mwdisp1) # 3N dim vector
d2au, d1au = d2 * (amu2au**0.5), d1 * (amu2au**0.5) # 3N dim vectors

# Remove displacement along translational & rotational modes
d2au = np.delete(d2au, (0,1,2,3,4,5)) # 3N-6 dim vector
d1au = np.delete(d1au, (0,1,2,3,4,5)) # 3N-6 dim vector
# Remove the displacement along imaginary modes
for i in range(Nimf): 
    d2au = np.delete(d2au, (0)) # 3N-6 dim vector
    d1au = np.delete(d1au, (0)) # 3N-6 dim vector

# Absolute dimensionless displacement
unitless_d1 = np.sqrt(frqau_f) * abs(d1au) # 3N-6 dim vector
unitless_d2 = np.sqrt(frqau_i) * abs(d2au) # 3N-6 dim vector


"""
Select modes of large dimensionless displacements for the SOC dierivatives
"""
ordered_disp_f, ordered_disp_i = [], []
for i in range(len(unitless_d1)):
    ordered_disp_f.append([unitless_d1[i], i])
ordered_disp_f = sorted(ordered_disp_f, reverse=True) 
for i in range(len(unitless_d2)):
    ordered_disp_i.append([unitless_d2[i], i])
ordered_disp_i = sorted(ordered_disp_i, reverse=True) 



modeidx_i = []
for i in range(ndisp_modes_i):
    modeidx_i.append(ordered_disp_i[i][1]+7+Nimf)
modeidx_i = np.array(modeidx_i)
fmt = 10*'{:d} '
print('Initial state (*** Printed indices begin with 7 for the first vib mode ***)')
for i in range(int(ndisp_modes_i/10)):
    print(fmt.format(*modeidx_i[i*10:(i+1)*10])) # Printed indices begin with 0 for the first vib mode
print('\n')

modeidx_f = []
for i in range(ndisp_modes_f):
    modeidx_f.append(ordered_disp_f[i][1]+7+Nimf)
modeidx_f = np.array(modeidx_f)
fmt = 10*'{:d} '
print('Final state (*** Printed indices begin with 7 for the first vib mode ***)')
for i in range(int(ndisp_modes_f/10)):
    print(fmt.format(*modeidx_f[i*10:(i+1)*10])) # Printed indices begin with 0 for the first vib mode
print('\n')


import zipfile
def zipFiles(zip_location, zip_name, file_name):
    with zipfile.ZipFile(zip_location+zip_name, 'a') as myzip: 
        zipFileName = '' #always take whats after the parentDir for the filename going in the zip
        myzip.write(os.path.join(zip_location, file_name), os.path.join(zipFileName, file_name), compress_type=zipfile.ZIP_DEFLATED) 


"""
Define molecular geometries in unitless normal coordinates
"""
# Convert Mass into atomic unit
masau_ext = masamu_ext * amu2au
masau_mat = np.diag(masau_ext)

# Remove imaginary mode coordinates
if Nimf > 0:
    for i in range(Nimf):
        mwvec_i = np.delete(mwvec_i, (6), axis=0)
        mwvec_f = np.delete(mwvec_f, (6), axis=0)

# # Rotate Cartesian geometries into normal coordinates
# ncgeo_i = np.matmul(mwvec_i, np.matmul(masau_mat**0.5, crd_i.reshape(-1)))
# ncgeo_f = np.matmul(mwvec_f, np.matmul(masau_mat**0.5, crd_f.reshape(-1)))

# # Multiply sqrt(freq) to the above normal coords to make them unitless
# frqau_i = np.insert(frqau_i, 0, (0,0,0,0,0,0))
# frqau_f = np.insert(frqau_f, 0, (0,0,0,0,0,0))
# unitless_ncgeo_i = ncgeo_i * (frqau_i**0.5)
# unitless_ncgeo_f = ncgeo_f * (frqau_f**0.5)


#%%
"""
Displace 
    the FINAL STATE GEOMETRY 
along 
    the FINAL STATE NORMAL MODES
and convert into Cartesian Angstrom
"""
file_path = '/Users/user/Desktop/dabna-1_tpssh/rate_calculations//631g_sp_nac/'
geometry = crd_f
projection = mwvec_f
frequency = frqau_f
displacement = ordered_disp_f
ndisp = ndisp_modes_f
normal_mode_geometry = 's1'

### Final state geometry expressed in final state normal coordinates
normal_coord = np.matmul(projection, np.matmul(masau_mat**0.5, geometry.reshape(-1)))
frequency = np.insert(frequency, 0, (0,0,0,0,0,0))
unitless_normal_coord = normal_coord * (frequency**0.5)

for imode in range(ndisp):
    ilot = int(imode/10) + 1
    mode_idx = displacement[imode][1]
#    mode_idx = imode # When want to take lowest frequency vibrational modes
    mode_idx = 82
    unitless_normal_coord_ph = unitless_normal_coord.copy()
    unitless_normal_coord_mh = unitless_normal_coord.copy()
    unitless_normal_coord_ph[mode_idx+6] += disp_size
    unitless_normal_coord_mh[mode_idx+6] -= disp_size
    
    # Rotate the displaced geometries into Angstrom Cartesian
    normal_coord_ph = unitless_normal_coord_ph[6:] / (frequency[6:]**0.5)
    normal_coord_mh = unitless_normal_coord_mh[6:] / (frequency[6:]**0.5)
    normal_coord_ph = np.insert(normal_coord_ph, 0, normal_coord[:6])
    normal_coord_mh = np.insert(normal_coord_mh, 0, normal_coord[:6])
        
    geometry_ph = np.matmul(np.linalg.inv(masau_mat**0.5), np.matmul(projection.T, normal_coord_ph))
    geometry_mh = np.matmul(np.linalg.inv(masau_mat**0.5), np.matmul(projection.T, normal_coord_mh))
    geometry_ph /= ang2bohr
    geometry_mh /= ang2bohr
    geometry_ph = geometry_ph.reshape((natom, 3))
    geometry_mh = geometry_mh.reshape((natom, 3))
    
    # Mode index '7' labels the first vib mode
    init_state = igeo_file[:2]
    final_state = fgeo_file[:2]
    xyz_to_create = '{:s}_mode{:<d}_+{:<5.3f}.xyz'.format(normal_mode_geometry, mode_idx+7+Nimf, disp_size)
    zip_to_create = '{:s}_flat_hess_disp_{:.2f}_lot{:d}.zip'.format(normal_mode_geometry, disp_size, ilot)
    with open(os.path.join(file_path, xyz_to_create), 'w') as f:
        f.write('{:<d} \n'.format(natom))
        f.write('new \n')
        for iatom in range(natom):
            f.write('{:<4s}{:<14.8f}{:<14.8f}{:<14.8f}\n'.format(atm_sym[iatom], *geometry_ph[iatom]))
#    add_file_to_zip(zip_to_create, xyz_to_create)
    zipFiles(file_path, zip_to_create, xyz_to_create)
    os.remove(os.path.join(file_path, xyz_to_create))
    
    # Mode index '7' labels the first vib mode
    xyz_to_create = '{:s}_mode{:<d}_-{:<5.3f}.xyz'.format(normal_mode_geometry, mode_idx+7+Nimf, disp_size)
    with open(os.path.join(file_path, xyz_to_create), 'w') as f:
        f.write('{:<d} \n'.format(natom))
        f.write('new \n')
        for iatom in range(natom):
            f.write('{:<4s}{:<14.8f}{:<14.8f}{:<14.8f}\n'.format(atm_sym[iatom], *geometry_mh[iatom]))
#    add_file_to_zip(zip_to_create, xyz_to_create)
    zipFiles(file_path, zip_to_create, xyz_to_create)
    os.remove(os.path.join(file_path, xyz_to_create))
            

# for imode in range(ndisp_modes_f):
#     ilot = int(imode/10) + 1
#     mode_idx = ordered_disp_f[imode][1]
# #    mode_idx = imode # When want to take lowest frequency vibrational modes
#     unitless_ncgeo_f_ph = unitless_ncgeo_f.copy()
#     unitless_ncgeo_f_mh = unitless_ncgeo_f.copy()
#     unitless_ncgeo_f_ph[mode_idx+6] += disp_size
#     unitless_ncgeo_f_mh[mode_idx+6] -= disp_size
    
#     # Rotate the displaced geometries into Angstrom Cartesian
#     ncgeo_f_ph = unitless_ncgeo_f_ph[6:] / (frqau_f[6:]**0.5)
#     ncgeo_f_mh = unitless_ncgeo_f_mh[6:] / (frqau_f[6:]**0.5)
#     ncgeo_f_ph = np.insert(ncgeo_f_ph, 0, ncgeo_f[:6])
#     ncgeo_f_mh = np.insert(ncgeo_f_mh, 0, ncgeo_f[:6])
    
#     crd_f_ph = np.matmul(np.linalg.inv(masau_mat)**0.5, np.matmul(mwvec_f.T, ncgeo_f_ph))
#     crd_f_mh = np.matmul(np.linalg.inv(masau_mat)**0.5, np.matmul(mwvec_f.T, ncgeo_f_mh))
#     crd_f_ph /= ang2bohr
#     crd_f_mh /= ang2bohr
#     crd_f_ph = crd_f_ph.reshape((natom,3))
#     crd_f_mh = crd_f_mh.reshape((natom,3))
    
#     # Mode index '7' labels the first vib mode
#     final_state = fgeo_file[:2]
#     xyz_to_create = '{:s}_mode{:<d}_+{:<5.3f}.xyz'.format(final_state, mode_idx+7+Nimf, disp_size)
#     zip_to_create = '{:s}_disp_{:.2f}_lot{:d}.zip'.format(final_state, disp_size, ilot)
#     with open(os.path.join(file_path, xyz_to_create), 'w') as f:
#         f.write('{:<d} \n'.format(natom))
#         f.write('comment \n')
#         for iatom in range(natom):
#             f.write('{:<4s}{:<14.8f}{:<14.8f}{:<14.8f}\n'.format(atm_sym[iatom], *crd_f_ph[iatom]))
# #    add_file_to_zip(zip_to_create, xyz_to_create)
#     zipFiles(file_path, zip_to_create, xyz_to_create)
#     os.remove(os.path.join(file_path, xyz_to_create))
    
#     # Mode index '7' labels the first vib mode
#     xyz_to_create = '{:s}_mode{:<d}_-{:<5.3f}.xyz'.format(final_state, mode_idx+7+Nimf, disp_size)
#     with open(os.path.join(file_path, xyz_to_create), 'w') as f:
#         f.write('{:<d} \n'.format(natom))
#         f.write('comment \n')
#         for iatom in range(natom):
#             f.write('{:<4s}{:<14.8f}{:<14.8f}{:<14.8f}\n'.format(atm_sym[iatom], *crd_f_mh[iatom]))
# #    add_file_to_zip(zip_to_create, xyz_to_create)
#     zipFiles(file_path, zip_to_create, xyz_to_create)
#     os.remove(os.path.join(file_path, xyz_to_create))

#%%
# # Read Hessian matrix
# nmode = 3*natom
# nchunk = int(nmode/6)
# nmod = nmode%6
# if nmod != 0:
#     nchunk += 1

# vec_i = np.zeros((nmode, nmode))
# with open('/Users/user/Desktop/dtc-dbt/' + 't3_hess_orca', 'r') as f:
#     for ichunk in range(nchunk):
#         f.readline()
#         for imode in range(nmode):
#             x = f.readline().split()
#             if ichunk == nchunk-1:
#                 lim = nmod
#             else:
#                 lim = 6
#             for i in range(lim):
#                 vec_i[6*ichunk+i, imode] = float(x[i+1])

# # Sqrt of atomic masses                
# tmp = np.sqrt(masamu) #* np.sqrt(amu2au**0.5)
# sqrt_mass = []
# for i in range(len(tmp)):
#     for j in range(3):
#         sqrt_mass.append(tmp[i])
        
# # multiply Hessian with inverse sqrt of atomic masses
# mass_vec_i = np.zeros_like(vec_i)
# for i in range(len(vec_i)):
#     if i > 5:
#         tmp = vec_i[i] * sqrt_mass
#         mass_vec_i[i] = tmp/np.linalg.norm(tmp)
#     else:
#         mass_vec_i[i] = np.zeros(nmode)
    
# # Check orthonormality
# np.matmul(mass_vec_i[6:], mass_vec_i[6:].T)
# np.matmul(mass_vec_i[6:].T, mass_vec_i[6:])


