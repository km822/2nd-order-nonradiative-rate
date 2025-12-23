#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 18 20:31:00 2024

@author: kenmiyazaki
"""
import numpy as np
from sim_param import simtemp

""" 
Constants & unit conversion factors
"""
pi       = np.pi
planck   = 6.62607015*10**-34 #SI
hbar     = planck/(2*pi) #SI
light    = 2.99792458*10**8 #SI
cau      = 137.03604 #A.U.
eh2j     = 4.359744650*10**-18
eh2ev    = 27.21138602
eh2wno   = 219474.63068 # Eh to linear frequency
eh2hz    = 6.579684*10**15
j2ev     = 6.2415093433*10**18
amu2kg   = 1.66054*10**-27
amu2au   = 1.822888486*10**3
wno2hz   = 2.99792458*10**10
wno2ev   = 1.23981*10**-4
wno2j    = 1.986445857*10**-23
hz2au    = 2.41888*10**-17
ang2bohr = 1.88973
kb       = 1.380649*10**-23 #SI
hbarbykb = hbar/kb
auTMP2K  = eh2j/kb
autime2s = hbar/eh2j #A.U. time to second. Can also be used for Hz to A.U. frequency
hz2ev    = eh2ev * autime2s/(2*pi)
invT     = auTMP2K * (simtemp)**-1
