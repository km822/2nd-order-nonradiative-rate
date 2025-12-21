import numpy as np

#############################
### Molecular information ###
#############################

'''
DABNA-1 [T1 --> T2 --> S1]

All geometries obtained by TPSSh/6-31G*
Energies obtained by EOM-CCSD/6-31G
SOC obtained by EOM-CCSD/6-31G
   - T1-S1 SOC: centered at S1 geometry, displaced along modes obtained at S1 geometry
   - T2-S1 SOC: centered at T2 geometry, displaced along modes obtained at S1 geometry
NAC obtained by EOM-CCSD/6-31G

Convergence Check: 
    Gaussian width = 0, ts = 15.0, tf = 15000
Rate Calculation:
    Gaussian width = 0, ts = 6.0 & 9.0 & 12.0, tf = 3000 & 7500 & 15000 --> 3.3 x 10^4 [1/s]
'''
# file_path = '/Users/user/Desktop/dabna-1_tpssh/rate_calculations/631g_sp_nac/'
# natom = 54
# adiabE = [-1284.38144131, -1284.36008219, -1284.37020249] # energies of relevant adiabatic states            
# sm = ['T', 'T', 'S'] # multiplicity symbols in the order of initial to intermediate(s) to final state (i, m1, m2, ..., f)

# ffreq_file = 's1_freq_tpssh_631gd_qchem'
# fhess_file = 's1_hess_tpssh_631gd_qchem'
# fgeo_file = 's1_geo_tpssh_631gd_qchem'

# ifreq_file = 't1_freq_tpssh_631gd_qchem'
# ihess_file = 't1_hess_tpssh_631gd_qchem'
# igeo_file = 't1_geo_tpssh_631gd_qchem'

# mass_file = 'amu_qchem'

# soc0_if = 0.047855 # SOC at equil geo [cm^-1]
# soc0_intmed = [[0.0, 0.648161], # SOC at equil geo [cm^-1] involving first intermediate state 
#                 ] # SOC at equil geo [cm^-1] involving second intermediate state 
# NM_disp_factor = 0.010 # dimensionless normal mode displacement
# disp_size = [0.01, 0.05] # dimensionless magnitude of displacements
# nlot = 6 # number of lots of 10 normal modes to include in SOC derivatives
# nmodes_per_lot = 10 # number of normal modes in one lot along which SOC derivatives are computed
# scale = 0.9594 # TPSSh/6-31G*

# '''
# NAC files
# '''
# nac_if_file = ''
# nac_intmed_file = [['nac_t1t2_at_t2_eomccsd_631g', ''],
#                     ]


# '''
# SOC files
# '''
# soc_if_filename_format = 'socS1-T1_s1geo_s1mode_lot{:d}_{:.3f}.out'
# soc_if_files = ['socS1-T1_s1geo_s1mode_lot1_0.010.out', 
#                 'socS1-T1_s1geo_s1mode_lot2_0.010.out',
#                 'socS1-T1_s1geo_s1mode_lot3_0.010.out',
#                 'socS1-T1_s1geo_s1mode_lot4_0.010.out',
#                 'socS1-T1_s1geo_s1mode_lot5_0.010.out',
#                 'socS1-T1_s1geo_s1mode_lot6_0.010.out',
#                 ]

# soc_intmed_filename_format = ['socS1-T2_t2geo_s1mode_lot{:d}_{:.3f}.out']
# soc_intmed_files = [
#                     [['', 'socS1-T2_t2geo_s1mode_lot1_0.010.out'],
#                       ['', 'socS1-T2_t2geo_s1mode_lot2_0.010.out'],
#                       ['', 'socS1-T2_t2geo_s1mode_lot3_0.010.out'],
#                       ['', 'socS1-T2_t2geo_s1mode_lot4_0.010.out'],
#                       ['', 'socS1-T2_t2geo_s1mode_lot5_0.010.out'],
#                       ['', 'socS1-T2_t2geo_s1mode_lot6_0.010.out'],
#                       ],
#                     ] 


#%%
'''
A6AP-Cz (No PCM) [T1 --> T2 --> S1]

All geometries obtained by B3LYP/6-31G*
    - T1, T2, & S1 are all optimized to flat geometries
Energies obtained by EOM-CCSD/6-31G* at the *FLAT* configurations
SOC obtained by EOM-CCSD/6-31G
    - S1 with twist gave a local minimum and enhanced both S1-T1 & S1-T2 SOC.
    - zeroth order (geometry independent) SOCs are obtained at twisted S1
    - Displacements are made along the normal modes of flat S1 projected
      onto the displacement vector between twist S1 and flat T1.
NAC obtained by EOM-CCSD/6-31G* at *TWIST* T1

Convergence Check: 
    Gaussian width = 0, ts = 10.0, tf = 8000
Rate Calculation:
    Gaussian width = 0, ts = 3.0 & 6.0 & 10.0, tf = 900 & 3600 & 8000 --> 2.8 x 10^6  [1/s]
'''
# file_path = '/Users/user/Desktop/a6ap-cz/rate_calculations/no_pcm_flat/631gd_sp_nac/twist_geo_soc/'
# natom = 43
# adiabE = [-1188.91950473, -1188.89564441, -1188.922941] # energies of relevant adiabatic states      
# sm = ['T', 'T', 'S'] # multiplicity symbols in the order of initial to intermediate(s) to final state (i, m1, m2, ..., f)

# ffreq_file = 's1_freq_b3lyp_631gd_qchem'
# fhess_file = 's1_hess_b3lyp_631gd_qchem'
# fgeo_file  = 's1_geo_b3lyp_631gd_qchem'

# ifreq_file = 't1_freq_b3lyp_631gd_qchem'
# ihess_file = 't1_hess_b3lyp_631gd_qchem'
# igeo_file  = 't1_geo_b3lyp_631gd_qchem'

# mass_file = 'amu_qchem'

# soc0_if = 0.052318 # SOC at equil geo [cm^-1]
# soc0_intmed = [
#                 [0.0, 0.996435], # SOC at equil geo [cm^-1] involving first intermediate state 
#                 ] # SOC at equil geo [cm^-1] involving second intermediate state 
# NM_disp_factor = 0.010 # dimensionless normal mode displacement
# disp_size = [0.01] # dimensionless magnitude of displacements
# nlot = 5 # number of lots of 10 normal modes to include in SOC derivatives
# nmodes_per_lot = 10 # number of normal modes in one lot along which SOC derivatives are computed
# scale = 0.960 # B3LYP/6-31G*

# '''
# NAC files
# '''
# nac_if_file = ''
# nac_intmed_file = [['nac_t1t2_at_t1twist_eomccsd_631gd', ''],
#                     ]


# '''
# SOC files
# '''
# soc_if_filename_format = 'socS1-T1_geo_s1twist_hess_s1flat_lot{:d}_{:.3f}.out'
# soc_if_files = ['socS1-T1_geo_s1twist_hess_s1flat_lot1_0.010.out',
#                 'socS1-T1_geo_s1twist_hess_s1flat_lot2_0.010.out',
#                 'socS1-T1_geo_s1twist_hess_s1flat_lot3_0.010.out',
#                 'socS1-T1_geo_s1twist_hess_s1flat_lot4_0.010.out',
#                 'socS1-T1_geo_s1twist_hess_s1flat_lot5_0.010.out',
#                 ]

# soc_intmed_filename_format = ['socS1-T2_geo_s1twist_hess_s1flat_lot{:d}_{:.3f}.out']
# soc_intmed_files = [
#                     [['', 'socS1-T2_geo_s1twist_hess_s1flat_lot1_0.010.out'],
#                       ['', 'socS1-T2_geo_s1twist_hess_s1flat_lot2_0.010.out'],
#                       ['', 'socS1-T2_geo_s1twist_hess_s1flat_lot3_0.010.out'],
#                       ['', 'socS1-T2_geo_s1twist_hess_s1flat_lot4_0.010.out'],
#                       ['', 'socS1-T2_geo_s1twist_hess_s1flat_lot5_0.010.out'],
#                       ],
#                     ]



#%%
'''
A6AP-Cz (No PCM) [T1 --> T2 --> S1]

All geometries obtained by B3LYP/6-31G*
    - T1, T2, & S1 are all optimized to flat geometries
Energies obtained by EOM-CCSD/6-31G* at the *Flat* configurations
T1-S1 and T2-S1 SOCs obtained at S1 equilibrium (flat) geometry (EOM-CCSD/6-31G)
    - Displacements are made along the normal modes of flat S1 
NAC obtained by EOM-CCSD/6-31G* at flat T2

Convergence Check: 
    Gaussian width = 0, ts = 0, tf = 8000
Rate Calculation:
    Gaussian width = 0, ts = 3.0, 6.0, 10.0, tf = 900, 3600, 8000 --> 2.6 x 10^6 [1/s]
'''
file_path = '/Users/user/Desktop/a6ap-cz/rate_calculations/no_pcm_flat/631gd_sp_nac/flat_geo_soc/'
natom = 43
adiabE = [-1188.91950473, -1188.89564441, -1188.922941] # energies of relevant adiabatic states      
sm = ['T', 'T', 'S'] # multiplicity symbols in the order of initial to intermediate(s) to final state (i, m1, m2, ..., f)

ffreq_file = 's1_freq_b3lyp_631gd_qchem'
fhess_file = 's1_hess_b3lyp_631gd_qchem'
fgeo_file  = 's1_geo_b3lyp_631gd_qchem'

ifreq_file = 't1_freq_b3lyp_631gd_qchem'
ihess_file = 't1_hess_b3lyp_631gd_qchem'
igeo_file  = 't1_geo_b3lyp_631gd_qchem'

mass_file = 'amu_qchem'

soc0_if = 0.157338 # SOC at equil geo [cm^-1]
soc0_intmed = [
                [0.0, 0.339033], # SOC at equil geo [cm^-1] involving first intermediate state 
                ] # SOC at equil geo [cm^-1] involving second intermediate state 
NM_disp_factor = 0.010 # dimensionless normal mode displacement
disp_size = [0.01] # dimensionless magnitude of displacements
nlot = 5 # number of lots of 10 normal modes to include in SOC derivatives
nmodes_per_lot = 10 # number of normal modes in one lot along which SOC derivatives are computed
scale = 0.960 # B3LYP/6-31G*

'''
NAC files
'''
nac_if_file = ''
nac_intmed_file = [['nac_t1t2_at_t2_eomccsd_631gd', ''],
                    ]


'''
SOC files
'''
soc_if_filename_format = 'socS1-T1_s1_lot{:d}_{:.3f}.out'
soc_if_files = ['socS1-T1_s1_lot1_0.010.out',
                'socS1-T1_s1_lot2_0.010.out',
                'socS1-T1_s1_lot3_0.010.out',
                'socS1-T1_s1_lot4_0.010.out',
                'socS1-T1_s1_lot5_0.010.out',
                ]

soc_intmed_filename_format = ['socS1-T2_s1_lot{:d}_{:.3f}.out']
soc_intmed_files = [
                    [['', 'socS1-T2_s1_lot1_0.010.out'],
                     ['', 'socS1-T2_s1_lot2_0.010.out'],
                     ['', 'socS1-T2_s1_lot3_0.010.out'],
                     ['', 'socS1-T2_s1_lot4_0.010.out'],
                     ['', 'socS1-T2_s1_lot5_0.010.out'],
                     ],
                    ]



#%%
'''
A6AP-Cz (No PCM) [T1 --> T2 --> S1]

All geometries obtained by B3LYP/6-31G*
    - T1, T2, & S1 are all optimized to flat geometries
Energies obtained by EOM-CCSD/6-31G* at the *FLAT* configurations
SOC obtained by EOM-CCSD/6-31G
    - S1 displaced along 93rd normal mode (87th vibrational mode) enhanced both S1-T1 & S1-T2 SOC.
    - zeroth order (geometry independent) SOCs are obtained at displaced S1
    - Displacements are made along the normal modes of flat S1 projected
      onto the displacement vector between displaced S1 and flat T1.
NAC obtained by EOM-CCSD/6-31G* at S1 that is *DISPLACED* along the 93rd normal mode by -0.7

Convergence Check: 
    Gaussian width = 0, ts = 8.0, tf = 8000
Rate Calculation:
    Gaussian width = 0, ts = 3.0 & 6.0 & 10.0, tf = 900 & 3600 & 8000 --> 3.0 x 10^6 [1/s]
'''
# file_path = '/Users/user/Desktop/a6ap-cz/rate_calculations/no_pcm_flat/631gd_sp_nac/displaced_geo_soc/'
# natom = 43
# adiabE = [-1188.91950473, -1188.89564441, -1188.922941] # energies of relevant adiabatic states      
# sm = ['T', 'T', 'S'] # multiplicity symbols in the order of initial to intermediate(s) to final state (i, m1, m2, ..., f)

# ffreq_file = 's1_freq_b3lyp_631gd_qchem'
# fhess_file = 's1_hess_b3lyp_631gd_qchem'
# fgeo_file  = 's1_geo_b3lyp_631gd_qchem'

# ifreq_file = 't1_freq_b3lyp_631gd_qchem'
# ihess_file = 't1_hess_b3lyp_631gd_qchem'
# igeo_file  = 't1_geo_b3lyp_631gd_qchem'

# mass_file = 'amu_qchem'

# soc0_if = 0.158178 # SOC at equil geo [cm^-1]
# soc0_intmed = [
#                 [0.0, 0.441673], # SOC at equil geo [cm^-1] involving first intermediate state 
#                 ] # SOC at equil geo [cm^-1] involving second intermediate state 
# NM_disp_factor = 0.010 # dimensionless normal mode displacement
# disp_size = [0.01] # dimensionless magnitude of displacements
# nlot = 5 # number of lots of 10 normal modes to include in SOC derivatives
# nmodes_per_lot = 10 # number of normal modes in one lot along which SOC derivatives are computed
# scale = 0.960 # B3LYP/6-31G*

# '''
# NAC files
# '''
# nac_if_file = ''
# nac_intmed_file = [['nac_t1t2_at_s1displaced_along_93_eomccsd_631gd', ''],
#                     ]


# '''
# SOC files
# '''
# soc_if_filename_format = 'socS1-T1_s1_lot{:d}_{:.3f}.out'
# soc_if_files = ['socS1-T1_s1_lot1_0.010.out',
#                 'socS1-T1_s1_lot2_0.010.out',
#                 'socS1-T1_s1_lot3_0.010.out',
#                 'socS1-T1_s1_lot4_0.010.out',
#                 'socS1-T1_s1_lot5_0.010.out',
#                 ]

# soc_intmed_filename_format = ['socS1-T2_s1_lot{:d}_{:.3f}.out']
# soc_intmed_files = [
#                     [['', 'socS1-T2_s1_lot1_0.010.out'],
#                       ['', 'socS1-T2_s1_lot2_0.010.out'],
#                       ['', 'socS1-T2_s1_lot3_0.010.out'],
#                       ['', 'socS1-T2_s1_lot4_0.010.out'],
#                       ['', 'socS1-T2_s1_lot5_0.010.out'],
#                       ],
#                     ]

# # SOC behavior along the 93th normal mode (87th vibrational mode)
# t1s1 = [0.158178, 0.157840, 0.157900, 0.157437, 0.157412, 0.157405, 0.157400, 0.157338, 0.157259, 0.156903, 0.156218, 0.155223, 0.154747, 0.154982, 0.148731]
# t2s1 = [0.441673, 0.430185, 0.407386, 0.390189, 0.377501, 0.364805, 0.351974, 0.339033, 0.325900, 0.313369, 0.300215, 0.285186, 0.254879, 0.148202, 0.118577]
# d = [-0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
# plt.plot(d, t1s1, marker='o', label='T1-S1')
# plt.plot(d, t2s1, marker='o', label='T2-S1')
# plt.xlim(-0.75, 0.75)
# plt.xticks(np.arange(-0.7, 0.8, 0.1), ['{:.1f}'.format(i) for i in np.arange(-0.7,0.8,0.1)])
# plt.xlabel('Displacement along the 93rd normal mode')
# plt.ylabel('SOC')
# plt.legend()
# plt.show()

# SOC behavior along the 108th normal mode (102nd vibrational mode)
import matplotlib.pyplot as plt
disp = [-0.7,     -0.6,     -0.5,     -0.4,     -3.0,     -2.0,     -1.0,     0.0,      1.0,      2.0,      3.0,      0.4,      0.5,      +0.6,     +0.7,   ]
t1s1 = [0.152864, 0.153244, 0.153608, 0.153826, 0.154288, 0.154805, 0.155253, 0.157338, 0.157652, 0.158459, 0.159125, 0.160170, 0.161277, 0.162546, 0.163904]
t2s1 = [0.308282, 0.346115, 0.370478, 0.396274, 0.396580, 0.390438, 0.379035, 0.339033, 0.321167, 0.311663, 0.316276, 0.313731, 0.313733, 0.317873, 0.326398]
d = [-0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
plt.plot(d, t1s1, marker='o', label='T1-S1')
plt.plot(d, t2s1, marker='o', label='T2-S1')
plt.xlim(-0.75, 0.75)
plt.xticks(np.arange(-0.7, 0.8, 0.1), ['{:.1f}'.format(i) for i in np.arange(-0.7,0.8,0.1)])
plt.xlabel('Displacement along the 108th normal mode')
plt.ylabel('SOC')
plt.legend()
plt.show()

#%%
'''
A6AP-Cz (No PCM) [T1 --> T2 --> S1]

All geometries obtained by B3LYP/6-31G*
    - T1, T2, & S1 are all optimized to flat geometries
Energies obtained by EOM-CCSD/6-31G* at the *TWIST* configurations
SOC obtained by EOM-CCSD/6-31G
    - S1 with twist gave a local minimum and enhanced both S1-T1 & S1-T2 SOC.
    - Displacements are made along the normal modes of *FLAT* S1 projected
      onto the displacement vector between *TWIST* S1 and *FLAT* T1.
NAC obtained by EOM-CCSD/6-31G* at *TWIST* T1

Convergence Check: 
    Gaussian width = 0, ts = 10.0, tf = 8000
Rate Calculation:
    Gaussian width = 0, ts = 3.0 & 6.0 & 10.0, tf = 900 & 3600 & 8000 --> 1.1 x 10^5  [1/s]
'''
# file_path = '/Users/user/Desktop/a6ap-cz/rate_calculations/no_pcm_flat/631gd_sp_nac/twist_geo_soc/twist_geo_energy/'
# natom = 43
# adiabE = [-1188.90221926, -1188.88265215, -1188.90057379] # energies of relevant adiabatic states      
# sm = ['T', 'T', 'S'] # multiplicity symbols in the order of initial to intermediate(s) to final state (i, m1, m2, ..., f)

# ffreq_file = 's1_freq_b3lyp_631gd_qchem'
# fhess_file = 's1_hess_b3lyp_631gd_qchem'
# fgeo_file  = 's1_geo_b3lyp_631gd_qchem'

# ifreq_file = 't1_freq_b3lyp_631gd_qchem'
# ihess_file = 't1_hess_b3lyp_631gd_qchem'
# igeo_file  = 't1_geo_b3lyp_631gd_qchem'

# mass_file = 'amu_qchem'

# soc0_if = 0.052318 # SOC at equil geo [cm^-1]
# soc0_intmed = [
#                 [0.0, 0.996435], # SOC at equil geo [cm^-1] involving first intermediate state 
#                 ] # SOC at equil geo [cm^-1] involving second intermediate state 
# NM_disp_factor = 0.010 # dimensionless normal mode displacement
# disp_size = [0.01] # dimensionless magnitude of displacements
# nlot = 5 # number of lots of 10 normal modes to include in SOC derivatives
# nmodes_per_lot = 10 # number of normal modes in one lot along which SOC derivatives are computed
# scale = 0.960 # B3LYP/6-31G*

# '''
# NAC files
# '''
# nac_if_file = ''
# nac_intmed_file = [['nac_t1t2_at_t1twist_eomccsd_631gd', ''],
#                     ]


# '''
# SOC files
# '''
# soc_if_filename_format = 'socS1-T1_geo_s1twist_hess_s1flat_lot{:d}_{:.3f}.out'
# soc_if_files = ['socS1-T1_geo_s1twist_hess_s1flat_lot1_0.010.out',
#                 'socS1-T1_geo_s1twist_hess_s1flat_lot2_0.010.out',
#                 'socS1-T1_geo_s1twist_hess_s1flat_lot3_0.010.out',
#                 'socS1-T1_geo_s1twist_hess_s1flat_lot4_0.010.out',
#                 'socS1-T1_geo_s1twist_hess_s1flat_lot5_0.010.out',
#                 ]

# soc_intmed_filename_format = ['socS1-T2_geo_s1twist_hess_s1flat_lot{:d}_{:.3f}.out']
# soc_intmed_files = [
#                     [['', 'socS1-T2_geo_s1twist_hess_s1flat_lot1_0.010.out'],
#                       ['', 'socS1-T2_geo_s1twist_hess_s1flat_lot2_0.010.out'],
#                       ['', 'socS1-T2_geo_s1twist_hess_s1flat_lot3_0.010.out'],
#                       ['', 'socS1-T2_geo_s1twist_hess_s1flat_lot4_0.010.out'],
#                       ['', 'socS1-T2_geo_s1twist_hess_s1flat_lot5_0.010.out'],
#                       ],
#                     ]


#%%
'''
A6AP-Cz (No PCM) [T1 --> T2 --> S1]

All geometries obtained by B3LYP/6-31G*
    - T1, T2, & S1 are all optimized to flat geometries
Energies obtained by EOM-CCSD/6-31G* at the *Flat* configurations
T1-S1 SOC obtained at T1 geometry, T2-S1 SOC obtained at T2 geometry (EOM-CCSD/6-31G)
    - Displacements are made along the normal modes of flat S1 
NAC obtained by EOM-CCSD/6-31G* at flat T2

Convergence Check: 
    Gaussian width = 0, ts = 0, tf = 8000
Rate Calculation:
    Gaussian width = 0, ts = 3.0, 6.0, 10.0, tf = 900, 3600, 8000 --> 9.3 x 10^4 [1/s]
'''
# file_path = '/Users/user/Desktop/a6ap-cz/rate_calculations/no_pcm_flat/631gd_sp_nac/flat_geo_soc/'
# natom = 43
# adiabE = [-1188.91950473, -1188.89564441, -1188.922941] # energies of relevant adiabatic states      
# sm = ['T', 'T', 'S'] # multiplicity symbols in the order of initial to intermediate(s) to final state (i, m1, m2, ..., f)

# ffreq_file = 's1_freq_b3lyp_631gd_qchem'
# fhess_file = 's1_hess_b3lyp_631gd_qchem'
# fgeo_file  = 's1_geo_b3lyp_631gd_qchem'

# ifreq_file = 't1_freq_b3lyp_631gd_qchem'
# ihess_file = 't1_hess_b3lyp_631gd_qchem'
# igeo_file  = 't1_geo_b3lyp_631gd_qchem'

# mass_file = 'amu_qchem'

# soc0_if = 0.004372 # SOC at equil geo [cm^-1]
# soc0_intmed = [
#                 [0.0, 0.033063], # SOC at equil geo [cm^-1] involving first intermediate state 
#                 ] # SOC at equil geo [cm^-1] involving second intermediate state 
# NM_disp_factor = 0.010 # dimensionless normal mode displacement
# disp_size = [0.01] # dimensionless magnitude of displacements
# nlot = 5 # number of lots of 10 normal modes to include in SOC derivatives
# nmodes_per_lot = 10 # number of normal modes in one lot along which SOC derivatives are computed
# scale = 0.960 # B3LYP/6-31G*

# '''
# NAC files
# '''
# nac_if_file = ''
# nac_intmed_file = [['nac_t1t2_at_t2_eomccsd_631gd', ''],
#                     ]


# '''
# SOC files
# '''
# soc_if_filename_format = 'socS1-T1_t1_lot{:d}_{:.3f}.out'
# soc_if_files = ['socS1-T1_t1_lot1_0.010.out',
#                 'socS1-T1_t1_lot2_0.010.out',
#                 'socS1-T1_t1_lot3_0.010.out',
#                 'socS1-T1_t1_lot4_0.010.out',
#                 'socS1-T1_t1_lot5_0.010.out',
#                 ]

# soc_intmed_filename_format = ['socS1-T2_geo_s1twist_hess_s1flat_lot{:d}_{:.3f}.out']
# soc_intmed_files = [
#                     [['', 'socS1-T2_t2_lot1_0.010.out'],
#                      ['', 'socS1-T2_t2_lot2_0.010.out'],
#                      ['', 'socS1-T2_t2_lot3_0.010.out'],
#                      ['', 'socS1-T2_t2_lot4_0.010.out'],
#                      ['', 'socS1-T2_t2_lot5_0.010.out'],
#                      ],
#                     ]


#%%
'''
A6AP-Cz (No PCM) [T1 --> T2 --> S1]

All geometries obtained by B3LYP/6-31G*
    - T1, T2, & S1 are all optimized to flat geometries
Energies obtained by EOM-CCSD/6-31G* at the *FLAT* configurations
SOC obtained by EOM-CCSD/6-31G
    - S1 with twist gave a local minimum and enhanced both S1-T1 & S1-T2 SOC.
    - Displacements are made along the normal modes of flat S1 projected
      onto the displacement vector between twist S1 and flat T1.
NAC obtained by EOM-CCSD/6-31G* at *FLAT* T1

Convergence Check: 
    Gaussian width = 0, ts = 10.0, tf = 8000
Rate Calculation:
    Gaussian width = 0, ts = 3.0 & 6.0 & 10.0, tf = 900 & 3600 & 8000 --> 5.6 x 10^5  [1/s]
'''
# file_path = '/Users/user/Desktop/a6ap-cz/rate_calculations/no_pcm_flat/631gd_sp_nac/twist_geo_soc/flat_geo_nac/'
# natom = 43
# adiabE = [-1188.91950473, -1188.89564441, -1188.922941] # energies of relevant adiabatic states      
# sm = ['T', 'T', 'S'] # multiplicity symbols in the order of initial to intermediate(s) to final state (i, m1, m2, ..., f)

# ffreq_file = 's1_freq_b3lyp_631gd_qchem'
# fhess_file = 's1_hess_b3lyp_631gd_qchem'
# fgeo_file  = 's1_geo_b3lyp_631gd_qchem'

# ifreq_file = 't1_freq_b3lyp_631gd_qchem'
# ihess_file = 't1_hess_b3lyp_631gd_qchem'
# igeo_file  = 't1_geo_b3lyp_631gd_qchem'

# mass_file = 'amu_qchem'

# soc0_if = 0.052318 # SOC at equil geo [cm^-1]
# soc0_intmed = [
#                 [0.0, 0.996435], # SOC at equil geo [cm^-1] involving first intermediate state 
#                 ] # SOC at equil geo [cm^-1] involving second intermediate state 
# NM_disp_factor = 0.010 # dimensionless normal mode displacement
# disp_size = [0.01] # dimensionless magnitude of displacements
# nlot = 5 # number of lots of 10 normal modes to include in SOC derivatives
# nmodes_per_lot = 10 # number of normal modes in one lot along which SOC derivatives are computed
# scale = 0.960 # B3LYP/6-31G*

# '''
# NAC files
# '''
# nac_if_file = ''
# nac_intmed_file = [['nac_t1t2_at_t2_eomccsd_631gd', ''],
#                     ]


# '''
# SOC files
# '''
# soc_if_filename_format = 'socS1-T1_geo_s1twist_hess_s1flat_lot{:d}_{:.3f}.out'
# soc_if_files = ['socS1-T1_geo_s1twist_hess_s1flat_lot1_0.010.out',
#                 'socS1-T1_geo_s1twist_hess_s1flat_lot2_0.010.out',
#                 'socS1-T1_geo_s1twist_hess_s1flat_lot3_0.010.out',
#                 'socS1-T1_geo_s1twist_hess_s1flat_lot4_0.010.out',
#                 'socS1-T1_geo_s1twist_hess_s1flat_lot5_0.010.out',
#                 ]

# soc_intmed_filename_format = ['socS1-T2_geo_s1twist_hess_s1flat_lot{:d}_{:.3f}.out']
# soc_intmed_files = [
#                     [['', 'socS1-T2_geo_s1twist_hess_s1flat_lot1_0.010.out'],
#                      ['', 'socS1-T2_geo_s1twist_hess_s1flat_lot2_0.010.out'],
#                      ['', 'socS1-T2_geo_s1twist_hess_s1flat_lot3_0.010.out'],
#                      ['', 'socS1-T2_geo_s1twist_hess_s1flat_lot4_0.010.out'],
#                      ['', 'socS1-T2_geo_s1twist_hess_s1flat_lot5_0.010.out'],
#                      ],
#                     ]

#%%
'''
A6AP-Cz (No PCM) [T1 --> T2 --> S1]

All geometries obtained by B3LYP/6-31G*
    - T1, T2, & S1 are all optimized to flat geometries
Energies obtained by EOM-CCSD/6-31G* at the *FLAT* configurations
SOC obtained by EOM-CCSD/6-31G
    - S1 with twist gave a local minimum and enhanced both S1-T1 & S1-T2 SOC.
    - Displacements are made along the normal modes of flat S1 projected
      onto the displacement vector between twist S1 and flat T1.
NAC obtained by EOM-CCSD/6-31G* at *FLAT* T1

Convergence Check: 
    Gaussian width = 0, ts = 10.0, tf = 8000
Rate Calculation:
    Gaussian width = 0, ts = 3.0 & 6.0 & 10.0, tf = 900 & 3600 & 8000 --> 5.6 x 10^5  [1/s]
'''
# file_path = '/Users/user/Desktop/a6ap-cz/rate_calculations/no_pcm_flat/631gd_sp_nac/twist_geo_soc/twist_t1_hess/'
# natom = 43
# adiabE = [-1187.19828866, -1188.89564441, -1188.922941] # energies of relevant adiabatic states      
# sm = ['T', 'T', 'S'] # multiplicity symbols in the order of initial to intermediate(s) to final state (i, m1, m2, ..., f)

# ffreq_file = 's1_freq_b3lyp_631gd_qchem'
# fhess_file = 's1_hess_b3lyp_631gd_qchem'
# fgeo_file  = 's1_geo_b3lyp_631gd_qchem'

# ifreq_file = 't1_freq_b3lyp_631gd_twist_qchem'
# ihess_file = 't1_hess_b3lyp_631gd_twist_qchem'
# igeo_file  = 't1_geo_b3lyp_631gd_twist_qchem'

# mass_file = 'amu_qchem'

# soc0_if = 0.052318 # SOC at equil geo [cm^-1]
# soc0_intmed = [
#                 [0.0, 0.996435], # SOC at equil geo [cm^-1] involving first intermediate state 
#                 ] # SOC at equil geo [cm^-1] involving second intermediate state 
# NM_disp_factor = 0.010 # dimensionless normal mode displacement
# disp_size = [0.01] # dimensionless magnitude of displacements
# nlot = 5 # number of lots of 10 normal modes to include in SOC derivatives
# nmodes_per_lot = 10 # number of normal modes in one lot along which SOC derivatives are computed
# scale = 0.960 # B3LYP/6-31G*

# '''
# NAC files
# '''
# nac_if_file = ''
# nac_intmed_file = [['nac_t1t2_at_t1twist_eomccsd_631gd', ''],
#                     ]


# '''
# SOC files
# '''
# soc_if_filename_format = 'socS1-T1_geo_s1twist_hess_s1flat_lot{:d}_{:.3f}.out'
# soc_if_files = ['socS1-T1_geo_s1twist_hess_s1flat_lot1_0.010.out',
#                 'socS1-T1_geo_s1twist_hess_s1flat_lot2_0.010.out',
#                 'socS1-T1_geo_s1twist_hess_s1flat_lot3_0.010.out',
#                 'socS1-T1_geo_s1twist_hess_s1flat_lot4_0.010.out',
#                 'socS1-T1_geo_s1twist_hess_s1flat_lot5_0.010.out',
#                 ]

# soc_intmed_filename_format = ['socS1-T2_geo_s1twist_hess_s1flat_lot{:d}_{:.3f}.out']
# soc_intmed_files = [
#                     [['', 'socS1-T2_geo_s1twist_hess_s1flat_lot1_0.010.out'],
#                      ['', 'socS1-T2_geo_s1twist_hess_s1flat_lot2_0.010.out'],
#                      ['', 'socS1-T2_geo_s1twist_hess_s1flat_lot3_0.010.out'],
#                      ['', 'socS1-T2_geo_s1twist_hess_s1flat_lot4_0.010.out'],
#                      ['', 'socS1-T2_geo_s1twist_hess_s1flat_lot5_0.010.out'],
#                      ],
#                     ]


