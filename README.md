# 2nd-order-nonradiative-rate
This is a Python code that implements the nonradiative transition rate formulation based on the second-order perturbation theory for molecular systems.
In addition to the initial and final electronic states, the code can accept any number of intermediate electronic states regardless of their spins, provided that necessary nonadiabatic and spin-orbit couplings (NAC and SOC) are provided. Thanks to the second-order treatment, spin-vibronic (SV) effects arising from the conjunction of internal conversion and intersystem crossing are taken into account. Further, we include in the formulation the vibrational spin-orbit (VSO) effects, which account for the first-order dependence of SOCs on nuclear coordinates.

The underlying formulation is known as thermal vibration correlation function (TVCF) formulation and derived analytically assuming harmonic nuclear modes. This implementation takes it further by analytically removing the inherent singularity issues associated with TVCFs, hence achieving stable and robust numerical implementation.

The inputs to the code consist of the temperature, the equilibrium geometries of the initial and final electronic states, normal modes at the equilibrium geometries, the state energies of all electronic states (initial + final + intermediates), appropriate NACs and SOCs, and in case the inclusion of VSO effects is intended, the SOCs evaluated at geometries displaced forward and backward along the normal modes. The code can be RAM intensive since some of the TVCFs are rank-4 tensors dependent on time with each dimension containing as many degrees of freedom as the number of normal modes. It is possible to reduce the memory cost by cutting off couplings smaller than a chosen ratio to the largest coupling and not including SOC derivatives along some normal modes. The trade-off is accuracy so we advise users to use their discretion.

To run the code, users are expected to first edit sys_param.py and sim_param.py. The former needs to be edited so that files containing the molecular information described in the previous paragraph are correctly pointed to. As an example, the examples folder contains the necessary files to compute the reverse intersystem crossing rate of DABNA-1 involving T1, T2, and S1 states. It is important to note that the names of the frequency/Hessian/atomic mass files must follow a convention. Frequency and Hessian files must be named as
```
XX_freq_..._qchem
```
```
XX_hess_..._qchem
```
where XX is electronic state descriptor (s1, s2, t1, t2, etc.) and '...' can be anything. The atomic mass file must be named as
```
amu_qchem
```
The format of these input files must also be strictly consistent. The geometry files are in the standard .xyz format. The frequency/Hessian/atomic mass files are made of a corresponding snippet of Q-Chem output files. For a Hessian matrix, the portion of an output file beginning with the line "Eigenvectors of Proj. Mass-Weighted Hessian Matrix:" is used. Please find out the correct formats to use in the files in the 'examples' folder.

Another input file, sim_param.py, includes simulation parameters such as the temperature, coupling cutoffs, the size of the timestep for TVCF evaluations, and the limits of TVCF integration. The calculation can then be performed by running main.py. 

For more details, see our publication: ... coming soon.


Add .py for SOC displacement and analysis code
Create a folder for hand-written derivations
