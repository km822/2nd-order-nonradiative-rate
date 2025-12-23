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
file_path = '/examples/' # Location of the directory containing molecular information files
natom = 54
adiabE = [-1284.38144131, -1284.36008219, -1284.37020249] # energies of relevant adiabatic states            
sm = ['T', 'T', 'S'] # multiplicity symbols in the order of initial to intermediate(s) to final state (i, m1, m2, ..., f)

ffreq_file = 's1_freq_tpssh_631gd_qchem'
fhess_file = 's1_hess_tpssh_631gd_qchem'
fgeo_file = 's1_geo_tpssh_631gd_qchem'

ifreq_file = 't1_freq_tpssh_631gd_qchem'
ihess_file = 't1_hess_tpssh_631gd_qchem'
igeo_file = 't1_geo_tpssh_631gd_qchem'

mass_file = 'amu_qchem'

soc0_if = 0.047855 # SOC at equil geo [cm^-1]
soc0_intmed = [[0.0, 0.648161], # SOC at equil geo [cm^-1] involving first intermediate state 
               ] # SOC at equil geo [cm^-1] involving second intermediate state 
NM_disp_factor = 0.010 # dimensionless normal mode displacement
disp_size = [0.01, 0.05] # dimensionless magnitude of displacements
nlot = 6 # number of lots of 10 normal modes along which SOC derivatives are computed
nmodes_per_lot = 10 # number of normal modes in one lot along which SOC derivatives are computed
scale = 0.9594  # Vibrational mode frequency scaling factor for TPSSh/6-31G*

'''
NAC files
'''
# Nonadiabatic couplings
nac_if_file = ''
nac_intmed_file = [['nac_t1t2_at_t2_eomccsd_631g', ''],
                   ]


'''
SOC files
'''
# Initial to final state spin-orbit couplings
soc_if_filename_format = 'socS1-T1_s1geo_s1mode_lot{:d}_{:.3f}.out'
soc_if_files = ['socS1-T1_s1geo_s1mode_lot1_0.010.out', 
                'socS1-T1_s1geo_s1mode_lot2_0.010.out',
                'socS1-T1_s1geo_s1mode_lot3_0.010.out',
                'socS1-T1_s1geo_s1mode_lot4_0.010.out',
                'socS1-T1_s1geo_s1mode_lot5_0.010.out',
                'socS1-T1_s1geo_s1mode_lot6_0.010.out',
                ]

# Spin-orbit couplings involving intermediate states
soc_intmed_filename_format = ['socS1-T2_t2geo_s1mode_lot{:d}_{:.3f}.out']
soc_intmed_files = [
                    [['', 'socS1-T2_t2geo_s1mode_lot1_0.010.out'],
                     ['', 'socS1-T2_t2geo_s1mode_lot2_0.010.out'],
                     ['', 'socS1-T2_t2geo_s1mode_lot3_0.010.out'],
                     ['', 'socS1-T2_t2geo_s1mode_lot4_0.010.out'],
                     ['', 'socS1-T2_t2geo_s1mode_lot5_0.010.out'],
                     ['', 'socS1-T2_t2geo_s1mode_lot6_0.010.out'],
                     ],
                   ] 

