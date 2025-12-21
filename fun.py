#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 19 13:31:13 2024

@author: kenmiyazaki

Rrepository of functions to
- compute correlation functions
- ... 
"""
import numpy as np
from scipy.linalg import inv 
import mpmath
from tqdm import tqdm
import matplotlib.pyplot as plt
from sim_param import *
from sys_param import *
pi = np.pi
nmode = 3*natom
nvib = nmode-6
tgrid  = [list(np.arange(tlim[i], tlim[i+1], tstep[i])) for i in range(nsec)] # List of every time point btwn initial and final time
tidx_max = [len(tgrid[i]) - 1 for i in range(nsec)]
nintmed = len(sm[1:-1])

# Adiabatic energies
Ei, Ef, Em = adiabE[0], adiabE[-1], adiabE[1:-1]

def read_freq(file_path, freq_file):
    '''
    Read the hessian files for initial and final states to get
    1. frequencies, 2. reduced masses, and 3. normal coordinates
    
    p: the index of electronic structure package (1 = GAMESS, 2 = Q-Chem)
    '''
    nmode = 3*natom
    nvib = nmode-6
    
    if 'gamess' in freq_file[8:]:
        mode_per_row = 5
        nchunk = int(nmode/mode_per_row)
        nleft = nmode - nchunk*mode_per_row
        wno = []
        redmas = np.zeros(nmode)
        vec = np.zeros((nmode, nmode))
        p = 1
    elif 'qchem' in freq_file[8:]:
        mode_per_row = 3
        nchunk = int(nvib/mode_per_row)
        nleft = nvib - nchunk*mode_per_row
        wno = []
        redmas = np.zeros(nmode)
        vec = np.zeros((nvib, nmode))
        p = 2
    elif 'orca' in freq_file[8:]:
        mode_per_row = 6
        nchunk = int(nvib/mode_per_row)
        nleft = nvib - nchunk*mode_per_row
        wno = np.zeros(nvib)
        redmas = []
        vec = np.zeros((nvib, nmode))
        p = 3

    if nleft == 0:
        totchunk = nchunk
    else:
        totchunk = nchunk+1
    
    if p == 1: # GAMESS
        with open(file_path + freq_file, 'r') as f:
            for ichunk in range(1, totchunk+1):
                modini = (ichunk-1) * mode_per_row + 1
                modfnl = modini + mode_per_row - 1
                if (ichunk == totchunk and nleft != 0):
                    modfnl = modini + nleft - 1

                # wavenumber [cm-1]
                f.readline()
                x = f.readline().split()
                for i in range(len(x)):
                    wno.append(x[i])

                f.readline()

                # reduced mass [AMU]
                x = f.readline().split()
                for i in range(0, modfnl-modini+1):
                    redmas[modini-1+i] += float(x[i])

                # hessian coordinate
                [f.readline() for i in range(2)]
                for i in range(nmode):
                    x = f.readline().split()
                    for j in range(modfnl-modini+1):
                        vec[modini-1+j, i] += float(x[j])
                [f.readline() for i in range(11)]
                
        # Identify imaginary frequencies (if any) and remove them from the list of frequencies
        Nimf = len(wno)-nmode
        if Nimf == 0:
            for i in range(0,len(wno)):
                wno[i] = float(wno[i])
        else:
            imf = []
            for i in range(0, 2*Nimf, 2):
                imf.append(float(wno[i]))
                wno = [i for i in wno if "I" not in i]
            for i in range(0, len(wno)):
                wno[i] = float(wno[i])    
            for i in range(Nimf):
                for j in range(6+Nimf-1):
                    wno[j], wno[j+1] = wno[j+1], wno[j]
                    redmas[j], redmas[j+1] = redmas[j+1], redmas[j]
                    vec[[j,j+1]] = vec[[j+1,j]]
        
        wno = np.array(wno)
        redmas = np.array(redmas)
                    
    elif p == 2: # Q-CHEM
        [wno.append(0.0) for i in range(6)]
        with open(file_path + freq_file, 'r') as f:
            for ichunk in range(totchunk):
                modini = ichunk * mode_per_row
                modfnl = modini + mode_per_row - 1
                if (ichunk+1 == totchunk and nleft != 0):
                    modfnl = modini + nleft - 1

                # wavenumber [cm-1]
                f.readline()
                x = f.readline().split()[1:]
                for i in range(len(x)):
                    wno.append(float(x[i]))

                f.readline()

                # reduced mass [AMU]
                x = f.readline().split()
                x = x[2:]
                for i in range(modfnl-modini+1):
                    redmas[6+modini+i] += float(x[i])
                
                [f.readline() for i in range(4)]
                
                # normal mode vectors
                for i in range(natom):
                    x = f.readline().split()[1:]
                    for j in range(len(x)):
                        x[j] = float(x[j])
                    for j in range(modfnl-modini+1):
                        vec[modini+j, i*3:(i+1)*3] += x[j*3:(j+1)*3]
                
                [f.readline() for i in range(2)]
        
        # Identify imaginary frequencies (if any) and remove them from the list of frequencies
        Nimf = sum(1 for i in wno if i < 0)
        if Nimf > 0:
            imf = []
            for i in range(Nimf):
                imf.append(wno[i])
                
        wno = np.array(wno)
        redmas = np.array(redmas)
    
    # elif p == 3: # ORCA
    #     with open(file_path + ffreq_file, 'r') as f:
    #         with open(file_path + ifreq_file, 'r') as g:
    #             [f.readline() for i in range(12)]
    #             [g.readline() for i in range(12)]
                
    #             for i in range(nvib):
    #                 wno_f[i] = float(f.readline().split()[1])
    #                 wno_i[i] = float(g.readline().split()[1])
                
    #             [f.readline() for i in range(10)]
    #             [g.readline() for i in range(10)]
                
    #             # Read translation + rotation vectors
    #             [f.readline() for i in range(nmode+1)]
    #             [g.readline() for i in range(nmode+1)]
                
    #             for ichunk in range(1, totchunk+1):
    #                 modini = (ichunk-1) * mode_per_row + 1
    #                 modfnl = modini + mode_per_row - 1
    #                 if (ichunk == totchunk and nleft != 0):
    #                     modfnl = modini + nleft - 1
                        
    #                 f.readline()
    #                 g.readline()
    #                 for icrd in range(nmode):
    #                     x = f.readline().split()[1:]
    #                     y = g.readline().split()[1:]
    #                     for imode in range(modfnl-modini+1):
    #                         vec_f[modini-1+imode, icrd] += float(x[imode])
    #                         vec_i[modini-1+imode, icrd] += float(y[imode])
                        
    # # Identify imaginary frequencies (if any) and remove them from the list of frequencies
    # if p == 1:
    #     Nimf_f = len(wno_f)-nmode
    #     if Nimf_f == 0:
    #         for i in range(0,len(wno_f)):
    #             wno_f[i] = float(wno_f[i])
    #     else:
    #         imf_f = []
    #         for i in range(0, 2*Nimf_f, 2):
    #             imf_f.append(float(wno_f[i]))
    #             wno_f = [i for i in wno_f if "I" not in i]
    #         for i in range(0, len(wno_f)):
    #             wno_f[i] = float(wno_f[i])    
    #         for i in range(Nimf_f):
    #             for j in range(6+Nimf_f-1):
    #                 wno_f[j], wno_f[j+1] = wno_f[j+1], wno_f[j]
    #                 redmas_f[j], redmas_f[j+1] = redmas_f[j+1], redmas_f[j]
    #                 vec_f[[j,j+1]] = vec_f[[j+1,j]]
        
    #     Nimf_i = len(wno_i)-nmode
    #     if Nimf_i == 0:
    #         for i in range(0,len(wno_i)):
    #             wno_i[i] = float(wno_i[i])
    #     else:
    #         imf_i = []
    #         for i in range(0, 2*Nimf_i, 2):
    #             imf_i.append(float(wno_i[i]))
    #             wno_i = [i for i in wno_i if "I" not in i]
    #         for i in range(0,len(wno_i)):
    #             wno_i[i] = float(wno_i[i])
    #         for i in range(Nimf_i):
    #             for j in range(6+Nimf_i-1):
    #                 wno_i[j], wno_i[j+1] = wno_i[j+1], wno_i[j]
    #                 redmas_i[j], redmas_i[j+1] = redmas_i[j+1], redmas_i[j]
    #                 vec_i[[j,j+1]] = vec_i[[j+1,j]]
                    
    #     wno_f, wno_i = np.array(wno_f), np.array(wno_i)
    #     redmas_f, redmas_i = np.array(redmas_f), np.array(redmas_i)
                    
    # elif p == 2 or p == 3:
    #     Nimf_f = sum(1 for i in wno_f if i < 0)
    #     if Nimf_f > 0:
    #         imf_f = []
    #         for i in range(Nimf_f):
    #             imf_f.append(wno_f[i])
        
    #     Nimf_i = sum(1 for i in wno_i if i < 0)
    #     if Nimf_i > 0:
    #         imf_i = []
    #         for i in range(Nimf_i):
    #             imf_i.append(wno_i[i])
                
    #     if p == 2:
    #         wno_f, wno_i = np.array(wno_f), np.array(wno_i)
    #         redmas_f, redmas_i = np.array(redmas_f), np.array(redmas_i)
    
    return wno, redmas, vec, Nimf, p


def read_hess(hess_file):
    '''
    Read Hessian matrix for Qchem calculation 

    Returns
    -------
    hess : 2D array
        Orthonormalized Hessian matrix
    '''
    row_length = 6
    nchunk = int(nmode/row_length)
    nleft = nmode - nchunk*row_length
    vec = np.zeros((nmode, nmode))
    if nleft == 0:
        totchunk = nchunk
    else:
        totchunk = nchunk+1

    with open(file_path + hess_file, 'r') as f:
        for ichunk in range(totchunk):
            modini = ichunk * row_length + 1
            modfnl = modini + row_length - 1
            if (ichunk == totchunk-1 and nleft != 0):
                modfnl = modini + nleft - 1
                    
            for xyz in range(nmode):            
                x = f.readline().split()
                for j in range(modfnl-modini+1):
                    vec[modini-1+j, xyz] = float(x[j])
            [f.readline() for i in range(2)]
    return vec


def read_atomic_amu(amu_file):
    '''
    Read AMU of each atom in the molecule
    
    amu_file: the name of the file storing atomic masses (string)
    '''
    masamu = np.zeros(natom)
    atm_sym = []
    with open(file_path + amu_file, 'r') as f:
        if 'gamess' in amu_file[4:]:
            for i in range(natom):
                x = f.readline().split()
                masamu[i] += float(x[2])
                atm_sym.append(x[1])
        elif 'qchem' in amu_file[4:]:
            for i in range(natom):
                x = f.readline().split()
                masamu[i] += float(x[6])
                atm_sym.append(x[3])
        elif 'orca' in amu_file[4:]:
            for i in range(natom):
                masamu[i] += float(f.readline())
                # Here add a code for atm_sym
    return masamu, atm_sym


def get_mass_weighted_nvec(p, vec, masamu):
    '''
    Calculate mass-weighted Cartesian vectors that compose orthonormal matrix
    '''
    if p == 1:
        prenorm_mwvec = []
        mwvec = np.zeros((nmode, nmode))
        for i in range(nmode):
            for j in range(1, natom+1):
                atmini = 3*(j-1)+1
                atmfnl = atmini+3
                for xyz in range(atmini, atmfnl):
                    mwvec[i,xyz-1] += vec[i,xyz-1] * (masamu[j-1]**0.5)  
    elif p == 2:
        prenorm_mwvec = []
        mwvec = vec
    elif p == 3:
        prenorm_mwvec = np.zeros((nvib, nmode))
        mwvec = np.zeros((nvib, nmode))
        for i in range(nvib):
            for j in range(1, natom+1):
                atmini = 3*(j-1)+1
                atmfnl = atmini+3
                for xyz in range(atmini, atmfnl):
                    prenorm_mwvec[i,xyz-1] += vec[i,xyz-1] * (masamu[j-1]**0.5)
            norm = np.sqrt(sum(prenorm_mwvec[i]**2))
            mwvec[i] = prenorm_mwvec[i]/norm
    return mwvec, prenorm_mwvec


def read_geo(file_path, fgeo_file, igeo_file):
    '''
    Return the arrays of molecular geometry 
    '''
    with open(file_path + fgeo_file + '.xyz', 'r') as f:
        [f.readline() for i in range(2)]
        crd_f = []
        for line in f:
            x = line.split()[1:]
            crd_f.append([float(j) for j in x])
    with open(file_path + igeo_file + '.xyz', 'r') as g:
        [g.readline() for i in range(2)]
        crd_i = []
        for line in g:
            x = line.split()[1:]
            crd_i.append([float(j) for j in x])
    crd_f = np.array(crd_f)
    crd_i = np.array(crd_i)
    return crd_f, crd_i
        

if integrator == 'simpson':        
    def rule(tidx, tidx_max, tstep, y):
        '''
        Return the value of function weighted by Simpson's rule
    
        Note that all correlation functions to be integrated 
        in this project is symmetric in real domain and anti-symmetric
        in imaginary domain. The integration over indefinite range
        of time, therefore, results in only a finite real value with
        no imaginary part.
    
        tidx: index of time  
        tidx_max: max index of time 
        tstep: step size
        y: quantity of the integrand 
        '''
        if tidx == 0 or tidx == tidx_max:
            return 0.333333 * tstep * y 
        elif tidx % 2 != 0:
            return 1.333333 * tstep * y 
        elif tidx % 2 == 0:
            return 0.666667 * tstep * y
        
elif integrator == 'trapezoidal':        
    def rule(tidx, tidx_max, tstep, y):
        '''
        Return the value of function weighted by trapezoidal rule
    
        Note that all correlation functions to be integrated 
        in this project is symmetric in real domain and anti-symmetric
        in imaginary domain. The integration over indefinite range
        of time, therefore, results in only a finite real value with
        no imaginary part.
    
        tidx: index of time  
        tidx_max: max index of time 
        tstep: step size
        y: quantity of the integrand 
        '''
        if tidx == 0 or tidx == tidx_max:
            return 0.5 * tstep * y 
        else:
            return tstep * y 

elif integrator == 'boole':        
    def rule(tidx, tidx_max, tstep, y):
        '''
        Return the value of function weighted by Boole's rule
    
        Note that all correlation functions to be integrated 
        in this project is symmetric in real domain and anti-symmetric
        in imaginary domain. The integration over indefinite range
        of time, therefore, results in only a finite real value with
        no imaginary part.
    
        tidx: index of time  
        tidx_max: max index of time 
        tstep: step size
        y: quantity of the integrand 
        '''
        if tidx == 0 or tidx == tidx_max:
            return (14/45) * tstep * y 
        elif tidx % 2 != 0:
            return (64/45) * tstep * y 
        elif tidx % 4 == 2:
            return (8/15) * tstep * y 
        elif tidx % 4 == 0:
            return (28/45) * tstep * y 
        
elif integrator == 'romberg':
    def rule(f, ti, tf, beta, omg_S_d, idx_list, max_nrow, acc):
        """
        Calculates the integral of a function using Romberg integration.
        
        Args:
            f       : "integrand_XXXX"
            ti      : lower limit of integration.
            tf      : upper limit of integration.
            beta    : inverse temperature
            omg_S_d : a collection of frequency matrices, Duschinsky matrix, and displacement vector
            idx_list: list of pairs of mode indices
            acc     : desired accuracy.
        
        Returns:
            a 1D array of the approximate value of the integral.
        """
        Rp = np.zeros((max_nrow, len(idx_list)))  # Pointers to previous row
        Rc = np.zeros((max_nrow, len(idx_list)))  # Pointers to current row
        
        h = tf - ti  # Step size
        rho0 = get_rho0([ti,tf], beta, omg_S_d)
        Rp[0] = 0.5 * h * (f(rho0[0],ti,omg_S_d,beta,idx_list) + f(rho0[1],tf,omg_S_d,beta,idx_list))  # First trapezoidal step
        
        for i in range(1, max_nrow):
            h /= 2.
            tgrid = np.arange(ti, tf+h, h)
            rho0 = get_rho0(tgrid, beta, omg_S_d)
            c = 0
            ep = 2**(i-1)
            
            # Trapezoidal rule routine
            for j in range(1, ep+1):
                tm     = (2*j-1)*h
                tm_idx = list(tgrid).index(tm)
                c += f(rho0[tm_idx], tm, omg_S_d, beta, idx_list)
            Rc[0] = h*c + 0.5*Rp[0]  # R(i,0)
            
            # Construct Richardson extrapolation table
            for j in range(1, i + 1):
                n_k = 4**j
                Rc[j] = (n_k * Rc[j - 1] - Rp[j - 1]) / (n_k - 1)  # Compute R(i,j)
          
            # R[i,i] is the best estimate so far
            # Check if a converged result is obtained
            if i > 1 and np.max(abs(Rp[i - 1] - Rc[i])/Rp[i - 1]) < acc:
                print('Accuracy criteria is met at row depth: {:<d}'.format(i))
                print('Error estimate: {:<.3e}'.format(np.max(abs(Rp[i - 1] - Rc[i])/(2**(2*i)-1))))
                return Rc[i], np.max(abs(Rp[i - 1] - Rc[i])/(2**(2*i)-1))
          
            # Rc now becomes Rp in the next cycle
            Rp, Rc = Rc, Rp
            
        print('Row depth: {:<d}'.format(i))
        print('Desired accuracy not achieved. % difference: {:<f}'.format(100*np.max(abs(Rc[i - 1] - Rp[i])/Rc[i - 1])))
        print('Error estimate: {:<.3e}'.format(np.max(abs(Rp[i - 1] - Rc[i])/(2**(2*i)-1))))
        
        # If reach the max_nrow, return our best guess
        return Rp[max_nrow - 1], np.max(abs(Rp[i - 1] - Rc[i])/(2**(2*i)-1))  

elif integrator == 'tanh_sinh':
    def rule(f, tf, beta, omg_S_d, idx_list):
        from mpmath import mp, mpf, ln, log, ceil, floor, sinh, asinh, cosh, tanh
        from sympy import Float
        # number of digits
        mp.dps = 6
    
        # Original integration limits
        a, b = 0, tf

        # we need a < b
        a, b = (a, b) if b > a else (b, a)
        # x = bpa2 + bma2*r
        bpa2, bma2 = (b + a)/2, (b - a)/2

        # epsilon
        eps = mpf('10')**(-mp.dps)
        # convergence threshold
        thr = mpf('10')**(-mp.dps/2)

        # (approx) maximum t that yields the maximum representable r & x
        # values strictly below the upper limits +1 (for r) and b (for x)
        tmax = asinh(ln(2*min(1, bma2)/eps)/pi)
    
        # level
        k = 0
        # maximum allowed level
        maxlevel = int(ceil(log(mp.dps, 2))) + 2
        # ss is the final result
        # 1st append at grid index 0 (r = 0)
        rho0 = get_rho0([a, bpa2, b], beta, omg_S_d)
        ss = f(rho0[1], bpa2, omg_S_d, beta, idx_list)
        # "initial" previous computed result, used in the convergence test
        sshm1 = 2*ss + 1
        # progress thru levels
        while k <= maxlevel:
            h = mpf('2')**-k
            N = int(floor(tmax/h))
            tgrid, wgrid = [], []
            for j in range(-N, N+1):
                tj  = j*h
                csh = pi*sinh(tj)/2
                ch  = cosh(tj)
                r   = tanh(csh)
                w   = ch / cosh(csh)**2
                if j != 0:
                    tgrid.append(bpa2 + bma2*r)
                    wgrid.append(w)
            tgrid = np.array(tgrid) # array of arguments at which f is evaluated
            wgrid = np.array(wgrid) # array of weights for each argument
            rho0 = get_rho0(tgrid, beta, omg_S_d)
            d3 = 0
            for j in range(-N, N+1):
                if j != 0:
                    tj_idx = list(tgrid).index(j*h)
                    ff = f(rho0[tj_idx], tgrid[tj_idx], omg_S_d, beta, idx_list)
                    p = w[tj_idx] * ff
                    if abs(np.max(p)) > d3:
                        d3 = np.max(abs(p))
                    ss += p
                    if j == -N:
                        psave = np.max(abs(p))
                    if j == N and np.max(abs(p)) > psave:
                        psave = np.max(abs(p))
            # converged?
            if np.max(abs(2*sshm1/ss - 1)) < thr: 
                print('Convergence reached at k = {:<d}'.format(k))
                break
            # no, advance to next level
            k += 1
            # store the just computed approximation for the next convergence test
            sshm2 = sshm1
            sshm1 = ss
        # apply constant coefficients
        ss *= bma2*pi*h/2
        
        # Error estimate
        d1 = log(abs(np.mean(ss - sshm1)), 10)
        if k > 1:
            d2 = log(abs(np.mean(ss - sshm2)), 10)
        d3 = log(eps * d3, 10)
        d4 = log(psave, 10)
        err = np.max(d1**2/d2, 2*d2, d3, d4)
        print('Error estimate: {:<f}'.format(err))
    
        # print results
        return Float(ss)


def integrand_QQ(rho0_tidx, t, omg_S_d, beta, idx_list):
    '''
    Evaluate the Q-Q integrand for Romberg integration routine
    Return the pair-wise integrand arranged in 1D array

    rho0_tidx: rho0 at current time t
    omg_S_d  : a collection of frequency matrices, Duschinsky matrix, and displacement vector
    beta     : inverse temperature
    idx_list : list of pairs of mode indices
    
    Return
    rho_kl: pair-wise CF arranged in a form of 1D array 
    '''
    # Complex exponential
    exp_wif = np.exp(1j*(Ei-Ef)*t) # complex exponential of adiabatic energy gap
    
    # Ingredient matrices and vectors
    omg_i, omg_f, S, d = omg_S_d
    dic = get_matvec(t, beta, omg_S_d)
    
    # Q-Q CF
    Omic = inv(dic['S'])@(dic['Ap']@inv(dic['Lp'])@dic['Cp'] - (dic['Am']@inv(dic['L'])@dic['Cm']))@inv(dic['omgf'])
    etap = dic['Cp'] @ dic['eta']
    nonexp = np.zeros(len(idx_list), dtype=complex)
    for j in range(len(idx_list)):
        pair = idx_list[j]
        nonexp[j] += 0.5*Omic[pair[1],pair[0]] + etap[pair[0]] * etap[pair[1]]

    rho_kl = np.real(exp_wif * rho0_tidx * nonexp)
    return(rho_kl)


def integrate0(rho0_tidx, tidx, tmax, tstep, exp_wif, dic):
    '''
    Return the array of integrated function (Rank 0)

    rho0_tidx: rho0 at time t
    t: single-point time
    tidx: index of time
    tmax: index of final integration time point
    tstep: step size
    beta: inverse temperature
    exp_wif: exponential term of adiabatic energy gap
    dic: dictionary of matrices and vectors
    intgl: scalar of integral  
    '''
    rho = np.real(exp_wif * rho0_tidx) 
    intgl = rule(tidx, tmax, tstep, rho)
    return(intgl, rho)


def integrate_nQf(rho0_tidx, tidx, tmax, tstep, idx_list, exp_wif, dic):
    '''
    Return the array of integrated function (Rank 1)

    rho0_tidx: rho0 at time t
    t: single-point time
    tidx: index of time
    tmax: index of final integration time point
    tstep: step size
    beta: inverse temperature
    idx_list: list of mode indices
    exp_wif: exponential term of adiabatic energy gap
    dic: dictionary of matrices and vectors
    intgl: 1D array of integral  
    '''
    nonexp = np.zeros(len(idx_list), dtype=complex)
    for j in range(len(idx_list)):
        sol = idx_list[j]
        nonexp[j] += -dic['eta'][sol]
    
    rho_k = np.real(exp_wif * rho0_tidx * nonexp) # vector
    intgl = rule(tidx, tmax, tstep, rho_k) # vector
    return(intgl, rho_k)


def integrateQfnQf(rho0_tidx, tidx, tmax, tstep, idx_list, exp_wif, dic):
    '''
    Return the array of integrated function (Rank 2)

    rho0_tidx: rho0 at current time 
    tidx: index of time
    tmax: index of final integration time point
    tstep: step size
    idx_list: list of pairs of mode indices
    exp_wif: exponential term of adiabatic energy gap
    dic: dictionary of matrices and vectors
    intgl: pair-wise integrals arranged in a form of 1D array 
    '''
    Omic = inv(dic['S'])@(dic['Ap']@inv(dic['Lp'])@dic['Cp'] - (dic['Am']@inv(dic['L'])@dic['Cm']))@inv(dic['omgf'])
    etap = dic['Cp'] @ dic['eta']
    nonexp = np.zeros(len(idx_list), dtype=complex)
    for j in range(len(idx_list)):
        pair = idx_list[j]
        nonexp[j] += 0.5*Omic[pair[1],pair[0]] + etap[pair[0]] * etap[pair[1]]
    
    rho_kl = np.real(exp_wif * rho0_tidx * nonexp)
    intgl = rule(tidx, tmax, tstep, rho_kl)
    return(intgl, rho_kl)


def integrate_nPi(rho0_tidx, tidx, tmax, tstep, idx_list, exp_wif, dic):
    '''
    Return the array of integrated function (Rank 1)

    rho0_tidx: rho0 at time t
    t: single-point time
    tidx: index of time
    tmax: index of final integration time point
    tstep: step size
    beta: inverse temperature
    idx_list: list of mode indices
    exp_wif: exponential term of adiabatic energy gap
    dic: dictionary of matrices and vectors
    intgl: 1D array of integrals 
    '''
    Appsi = (dic['omgi'] @ dic['A']) @ (dic['d'] - (dic['S'] @ dic['Cp'] @ dic['eta']))
    nonexp = np.zeros(len(idx_list), dtype=complex)
    for j in range(len(idx_list)):
        solo = idx_list[j]
        nonexp[j] += -1j * Appsi[solo]
    
    rho_kl = np.imag(exp_wif * rho0_tidx * nonexp)
    intgl = rule(tidx, tmax, tstep, 1j*rho_kl)
    return(intgl)
    

def integrate_nPf(rho0_tidx, tidx, tmax, tstep, idx_list, exp_wif, dic):
    '''
    Return the array of integrated function (Rank 1)

    rho0_tidx: rho0 at time t
    t: single-point time
    tidx: index of time
    tmax: index of final integration time point
    tstep: step size
    beta: inverse temperature
    idx_list: list of mode indices
    exp_wif: exponential term of adiabatic energy gap
    dic: dictionary of matrices and vectors
    intgl: 1D array of integrals 
    '''
    Appsi = (dic['S'].T @ dic['omgi'] @ dic['A']) @ (dic['d'] - (dic['S'] @ dic['Cp'] @ dic['eta']))
    nonexp = np.zeros(len(idx_list), dtype=complex)
    for j in range(len(idx_list)):
        solo = idx_list[j]
        nonexp[j] += -1j * Appsi[solo]
    
    rho_kl = np.imag(exp_wif * rho0_tidx * nonexp)
    intgl = rule(tidx, tmax, tstep, 1j*rho_kl)
    return(intgl)


def integrate_nPiQf(rho0_tidx, tidx, tmax, tstep, idx_list, exp_wif, dic):
    '''
    Return the array of integrated function (Rank 2)

    rho0_tidx: rho0 at time t    
    t: single-point time
    tidx: index of time
    tmax: index of final integration time point
    tstep: step size
    beta: inverse temperature
    idx_list: list of pairs of mode indices
    exp_wif: exponential term of adiabatic energy gap
    dic: dictionary of matrices and vectors
    intgl: pair-wise integrals arranged in a form of 1D array 
    '''
    tpDeliomgf = dic['omgi']@(dic['Ap']@inv(dic['L'])@dic['Cm'] + (dic['Am']@inv(dic['Lp'])@dic['Cp']))@inv(dic['omgf'])
    etap = dic['Cp'] @ dic['eta']
    Appsi = (dic['omgi'] @ dic['A']) @ (dic['d'] - (dic['S'] @ dic['Cp'] @ dic['eta']))
    nonexp = np.zeros(len(idx_list), dtype=complex)
    for j in range(len(idx_list)):
        pair = idx_list[j]
        nonexp[j] += -0.5j*tpDeliomgf[pair[0],pair[1]] + 1.0j*Appsi[pair[0]]*etap[pair[1]] 
    
    rho_kl = np.imag(exp_wif * rho0_tidx * nonexp)
    intgl = rule(tidx, tmax, tstep, 1j*rho_kl)        
    return(intgl)


def integrate_nQiPf(rho0_tidx, tidx, tmax, tstep, idx_list, exp_wif, dic):
    '''
    Return the array of integrated function (Rank 2)

    rho0_tidx: rho0 at time t    
    t: single-point time
    tidx: index of time
    tmax: index of final integration time point
    tstep: step size
    beta: inverse temperature
    idx_list: list of pairs of mode indices
    exp_wif: exponential term of adiabatic energy gap
    dic: dictionary of matrices and vectors
    intgl: pair-wise integrals arranged in a form of 1D array 
    '''
    Zpterm = dic['omgf']@(dic['Cm']@inv(dic['J'])@dic['Ap'] + dic['Cp']@inv(dic['Jp'])@dic['Am'])@inv(dic['omgi'])
    psi = dic['d'] - (dic['S'] @ dic['Cp'] @ dic['eta'])
    oceta = dic['omgf'] @ dic['Cm'] @ dic['eta']
    nonexp = np.zeros(len(idx_list), dtype=complex)
    for j in range(len(idx_list)):
        pair = idx_list[j]
        nonexp[j] += 0.5j*Zpterm[pair[1],pair[0]] - 1j*psi[pair[0]]*oceta[pair[1]] 
    
    rho_kl = np.imag(exp_wif * rho0_tidx * nonexp)
    intgl = rule(tidx, tmax, tstep, 1j*rho_kl)
    return(intgl)


def integrateQinPi(rho0_tidx, tidx, tmax, tstep, idx_list, exp_wif, dic):
    '''
    Return the array of integrated function (Rank 2)

    rho0_tidx: rho0 at time t    
    t: single-point time
    tidx: index of time
    tmax: index of final integration time point
    tstep: step size
    beta: inverse temperature
    idx_list: list of pairs of mode indices
    exp_wif: exponential term of adiabatic energy gap
    dic: dictionary of matrices and vectors
    intgl: pair-wise integrals arranged in a form of 1D array 
    '''
    tLterm = dic['omgi']@(dic['Ap']@inv(dic['L'])@dic['Cm'] - dic['Am']@inv(dic['Lp'])@dic['Cp'])@inv(dic['omgf'])@dic['S'].T
    psi = dic['d'] - (dic['S'] @ dic['Cp'] @ dic['eta'])
    oapsi = dic['omgi'] @ dic['A'] @ psi
    nonexp = np.zeros(len(idx_list), dtype=complex)
    for j in range(len(idx_list)):
        pair = idx_list[j]
        nonexp[j] += 0.5j*tLterm[pair[1],pair[0]] - 1j*psi[pair[0]]*oapsi[pair[1]]
    
    rho_kl = np.imag(exp_wif * rho0_tidx * nonexp)
    intgl = rule(tidx, tmax, tstep, 1j*rho_kl)
    return(intgl)


def integrateQinPf(rho0_tidx, tidx, tmax, tstep, idx_list, exp_wif, dic):
    '''
    Return the array of integrated function (Rank 2)

    rho0_tidx: rho0 at time t    
    t: single-point time
    tidx: index of time
    tmax: index of final integration time point
    tstep: step size
    beta: inverse temperature
    idx_list: list of pairs of mode indices
    exp_wif: exponential term of adiabatic energy gap
    dic: dictionary of matrices and vectors
    intgl: pair-wise integrals arranged in a form of 1D array 
    '''
    Zterm = dic['omgf']@(dic['Cm']@inv(dic['J'])@dic['Ap'] - dic['Cp']@inv(dic['Jp'])@dic['Am'])@inv(dic['omgi'])
    psi = dic['d'] - (dic['S'] @ dic['Cp'] @ dic['eta'])
    oceta = dic['omgf'] @ dic['Cm'] @ dic['eta']
    nonexp = np.zeros(len(idx_list), dtype=complex)
    for j in range(len(idx_list)):
        pair = idx_list[j]
        nonexp[j] += 0.5j*Zterm[pair[1],pair[0]] - 1j*psi[pair[0]]*oceta[pair[1]]
    
    rho_kl = np.imag(exp_wif * rho0_tidx * nonexp)
    intgl = rule(tidx, tmax, tstep, 1j*rho_kl)
    return(intgl)


def integrateQinPiQf(rho0_tidx, tidx, tmax, tstep, idx_list, exp_wif, dic):
    '''
    Return the array of integrated function (Rank 3)

    rho0_tidx: rho0 at time t
    t: single-point time
    tidx: index of time
    tmax: index of final integration time point
    tstep: step size
    beta: inverse temperature
    idx_list: list of triads of mode indices
    exp_wif: exponential term of adiabatic energy gap
    dic: dictionary of matrices and vectors
    intgl: triad-wise integrals arranged in a form of 1D array 
    '''
    tLterm = dic['omgi']@(dic['Ap']@inv(dic['L'])@dic['Cm'] - dic['Am']@inv(dic['Lp'])@dic['Cp'])@inv(dic['omgf'])@dic['S'].T
    ceta = dic['Cp'] @ dic['eta'] 
    tpLterm = dic['omgi']@(dic['Ap']@inv(dic['L'])@dic['Cm'] + dic['Am']@inv(dic['Lp'])@dic['Cp'])@inv(dic['omgf'])
    psi = dic['d'] - (dic['S'] @ dic['Cp'] @ dic['eta'])
    pLterm = (dic['Ap']@inv(dic['Lp'])@dic['Cp'] - dic['Am']@inv(dic['L'])@dic['Cm'])@inv(dic['omgf'])
    oapsi = dic['omgi'] @ dic['A'] @ psi
    nonexp = np.zeros(len(idx_list), dtype=complex)
    for j in range(len(idx_list)):
        tri = idx_list[j]
        nonexp[j] += -0.5j*tLterm[tri[1],tri[0]] * ceta[tri[2]]
        nonexp[j] += -0.5j*psi[tri[0]] * tpLterm[tri[1],tri[2]]
        nonexp[j] += -0.5j*pLterm[tri[0],tri[2]] * oapsi[tri[2]]
        nonexp[j] += 1j*psi[tri[0]] * oapsi[tri[1]] * ceta[tri[2]]
    
    rho_klm = np.imag(exp_wif * rho0_tidx * nonexp)
    intgl = rule(tidx, tmax, tstep, 1j*rho_klm)
    return(intgl)


def integrateQinQiPf(rho0_tidx, tidx, tmax, tstep, idx_list, exp_wif, dic):
    '''
    Return the array of integrated function (Rank 3)

    rho0_tidx: rho0 at time t
    t: single-point time
    tidx: index of time
    tmax: index of final integration time point
    tstep: step size
    beta: inverse temperature
    idx_list: list of triads of mode indices
    exp_wif: exponential term of adiabatic energy gap
    dic: dictionary of matrices and vectors
    intgl: triad-wise integrals arranged in a form of 1D array 
    '''
    tpZterm = dic['S']@(dic['Cp']@inv(dic['J'])@dic['Ap'] - dic['Cm']@inv(dic['Jp'])@dic['Am'])@inv(dic['omgi'])
    oceta = dic['omgf'] @ dic['Cm'] @ dic['eta'] 
    Zterm = dic['omgf']@(dic['Cm']@inv(dic['J'])@dic['Ap'] - dic['Cp']@inv(dic['Jp'])@dic['Am'])@inv(dic['omgi'])
    psi = dic['d'] - (dic['S'] @ dic['Cp'] @ dic['eta'])
    pZterm = dic['omgf']@(dic['Cm']@inv(dic['J'])@dic['Ap'] + dic['Cp']@inv(dic['Jp'])@dic['Am'])@inv(dic['omgi'])
    nonexp = np.zeros(len(idx_list), dtype=complex)
    for j in range(len(idx_list)):
        tri = idx_list[j]
        nonexp[j] += -0.5j*tpZterm[tri[1],tri[0]] * oceta[tri[2]]
        nonexp[j] += 0.5j*Zterm[tri[2],tri[0]] * psi[tri[1]] 
        nonexp[j] += 0.5j*psi[tri[0]] * pZterm[tri[2],tri[1]]
        nonexp[j] += -1j*psi[tri[0]] * psi[tri[1]] * oceta[tri[2]]
    
    rho_klm = np.imag(exp_wif * rho0_tidx * nonexp)
    intgl = rule(tidx, tmax, tstep, 1j*rho_klm)
    return(intgl)


def integratePinPi(rho0_tidx, tidx, tmax, tstep, idx_list, exp_wif, dic):
    '''
    Return the array of integrated function (Rank 2)

    rho0_tidx: rho0 at time t
    t: single-point time
    tidx: index of time
    tmax: index of final integration time point
    tstep: step size
    beta: inverse temperature
    idx_list: list of pairs of mode indices
    exp_wif: exponential term of adiabatic energy gap
    dic: dictionary of matrices and vectors
    intgl: pair-wise integrals arranged in a form of 1D array 
    '''
    Lpterm = dic['omgi'] @ dic['Am'] @ inv(dic['Lp']) @ dic['Cp'] @ inv(dic['omgf']) @ dic['S'].T - np.identity(nvib)
    term1 = Lpterm @ dic['omgi'] @ dic['A']
    Lterm  = dic['omgi'] @ dic['Ap'] @ inv(dic['L']) @ dic['Cm'] @ inv(dic['omgf']) @ dic['S'].T - np.identity(nvib)
    term2 = Lterm @ dic['omgi'] @ inv(dic['A'])
    oapsi = dic['omgi'] @ dic['A'] @ (dic['d'] - (dic['S'] @ dic['Cp'] @ dic['eta']))
    nonexp = np.zeros(len(idx_list), dtype=complex)
    for j in range(len(idx_list)):
        pair = idx_list[j]
        nonexp[j] += 0.5*(term1 - term2)[pair[1],pair[0]] + oapsi[pair[0]] * oapsi[pair[1]]
                   
    rho_kl = np.real(exp_wif * rho0_tidx * nonexp)
    intgl = rule(tidx, tmax, tstep, rho_kl) 
    return(intgl)


def integratePinPf(rho0_tidx, tidx, tmax, tstep, idx_list, exp_wif, dic):
    '''
    Return the array of integrated function (Rank 2)

    rho0_tidx: rho0 at time t
    t: single-point time
    tidx: index of time
    tmax: index of final integration time point
    tstep: step size
    beta: inverse temperature
    idx_list: list of pairs of mode indices
    exp_wif: exponential term of adiabatic energy gap
    dic: dictionary of matrices and vectors
    intgl: pair-wise integrals arranged in a form of 1D array 
    '''
    Lterm = dic['omgi'] @ (dic['Ap'] @ inv(dic['L']) @ dic['Cp'] - (dic['Am'] @ inv(dic['Lp']) @ dic['Cm']))
    oceta = dic['omgf'] @ dic['Cm'] @ dic['eta']
    oapsi = dic['omgi'] @ dic['A'] @ (dic['d'] - (dic['S'] @ dic['Cp'] @ dic['eta']))
    nonexp = np.zeros(len(idx_list), dtype=complex)
    for j in range(len(idx_list)):
        pair = idx_list[j]
        nonexp[j] += 0.5*Lterm[pair[0],pair[1]] + oapsi[pair[0]] * oceta[pair[1]]
                   
    rho_kl = np.real(exp_wif * rho0_tidx * nonexp)
    intgl = rule(tidx, tmax, tstep, rho_kl) 
    return(intgl)


def integratePfnPi(rho0_tidx, tidx, tmax, tstep, idx_list, exp_wif, dic):
    '''
    Return the array of integrated function (Rank 2)

    rho0_tidx: rho0 at time t
    t: single-point time
    tidx: index of time
    tmax: index of final integration time point
    tstep: step size
    beta: inverse temperature
    idx_list: list of pairs of mode indices
    exp_wif: exponential term of adiabatic energy gap
    dic: dictionary of matrices and vectors
    intgl: pair-wise integrals arranged in a form of 1D array  
    '''
    Lterm = dic['omgi'] @ (dic['Ap'] @ inv(dic['L']) @ dic['Cp'] - (dic['Am'] @ inv(dic['Lp']) @ dic['Cm']))
    oceta = dic['omgf'] @ dic['Cm'] @ dic['eta']
    oapsi = dic['omgi'] @ dic['A'] @ (dic['d'] - (dic['S'] @ dic['Cp'] @ dic['eta']))
    nonexp = np.zeros(len(idx_list), dtype=complex)
    for j in range(len(idx_list)):
        pair = idx_list[j]
        nonexp[j] += 0.5*Lterm[pair[1],pair[0]] + oapsi[pair[1]] * oceta[pair[0]]
                   
    rho_kl = np.real(exp_wif * rho0_tidx * nonexp)
    intgl = rule(tidx, tmax, tstep, rho_kl) 
    return(intgl)


def integratePfnPf(rho0_tidx, tidx, tmax, tstep, idx_list, exp_wif, dic):
    '''
    Return the array of integrated function (Rank 2)

    rho0_tidx: rho0 at time t
    t: single-point time
    tidx: index of time
    tmax: index of final integration time point
    tstep: step size
    beta: inverse temperature
    idx_list: list of pairs of mode indices
    exp_wif: exponential term of adiabatic energy gap
    dic: dictionary of matrices and vectors
    intgl: pair-wise integrals arranged in a form of 1D array 
    '''
    Lterm = dic['S'].T @ dic['omgi'] @ (dic['Ap'] @ inv(dic['L']) @ dic['Cp'] - (dic['Am'] @ inv(dic['Lp']) @ dic['Cm']))
    oceta = dic['omgf'] @ dic['Cm'] @ dic['eta']
    soapsi = dic['S'].T @ dic['omgi'] @ dic['A'] @ (dic['d'] - (dic['S'] @ dic['Cp'] @ dic['eta']))
    nonexp = np.zeros(len(idx_list), dtype=complex)
    for j in range(len(idx_list)):
        pair = idx_list[j]
        nonexp[j] += 0.5*Lterm[pair[1],pair[0]] + oceta[pair[0]] * soapsi[pair[1]]
                   
    rho_kl = np.real(exp_wif * rho0_tidx * nonexp)
    intgl = rule(tidx, tmax, tstep, rho_kl) 
    return(intgl)


def integratePfnPiQf(rho0_tidx, tidx, tmax, tstep, idx_list, exp_wif, dic):
    '''
    Return the array of integrated function (Rank 3)

    rho0_tidx: rho0 at time t
    t: single-point time
    tidx: index of time
    tmax: index of final integration time point
    tstep: step size
    beta: inverse temperature
    idx_list: list of triads of mode indices
    exp_wif: exponential term of adiabatic energy gap
    dic: dictionary of matrices and vectors
    intgl: triad-wise integrals arranged in a form of 1D array 
    ''' 
    Lterm = dic['omgi'] @ (dic['Ap'] @ inv(dic['L']) @ dic['Cp'] - dic['Am'] @ inv(dic['Lp']) @ dic['Cm'])
    ceta = dic['Cp'] @ dic['eta']
    hLterm = inv(dic['S']) @ (dic['Ap'] @ inv(dic['Lp']) @ dic['Cm'] - dic['Am'] @ inv(dic['L']) @ dic['Cp'])
    oapsi = dic['omgi'] @ dic['A'] @ (dic['d'] - (dic['S'] @ dic['Cp'] @ dic['eta']))
    oceta = dic['omgf'] @ dic['Cm'] @ dic['eta']
    tpLterm = dic['omgi']@(dic['Ap']@inv(dic['L'])@dic['Cm'] + dic['Am']@inv(dic['Lp'])@dic['Cp'])@inv(dic['omgf'])
    nonexp = np.zeros(len(idx_list), dtype=complex)
    for j in range(len(idx_list)):
        tri = idx_list[j]
        nonexp[j] += -0.5*Lterm[tri[1],tri[0]] * ceta[tri[2]]
        nonexp[j] += -0.5*hLterm[tri[0],tri[2]] * oapsi[tri[1]]
        nonexp[j] += oceta[tri[0]] * (0.5*tpLterm[tri[1],tri[2]] - oapsi[tri[1]] * ceta[tri[2]])

    rho_klm = np.real(exp_wif * rho0_tidx * nonexp)
    intgl = rule(tidx, tmax, tstep, rho_klm)
    return(intgl)


def integratePinPiQf(rho0_tidx, tidx, tmax, tstep, idx_list, exp_wif, dic):
    '''
    Return the array of integrated function (Rank 3)

    rho0_tidx: rho0 at time t
    t: single-point time
    tidx: index of time
    tmax: index of final integration time point
    tstep: step size
    beta: inverse temperature
    idx_list: list of triads of mode indices
    exp_wif: exponential term of adiabatic energy gap
    dic: dictionary of matrices and vectors
    intgl: triad-wise integrals arranged in a form of 1D array 
    ''' 
    tLterm = dic['omgi'] @ (dic['Ap'] @ inv(dic['L']) @ dic['Cm'] - dic['Am'] @ inv(dic['Lp']) @ dic['Cp']) @ inv(dic['omgf'])
    oapsi = dic['omgi'] @ dic['A'] @ (dic['d'] - (dic['S'] @ dic['Cp'] @ dic['eta']))
    tpLterm = dic['omgi'] @ (dic['Ap'] @ inv(dic['L']) @ dic['Cm'] + dic['Am'] @ inv(dic['Lp']) @ dic['Cp']) @ inv(dic['omgf'])
    ceta = dic['Cp'] @ dic['eta']
    Lpterm = dic['omgi'] @ dic['Am'] @ inv(dic['Lp']) @ dic['Cp'] @ inv(dic['omgf']) @ dic['S'].T - np.identity(nvib)
    term1 = Lpterm @ dic['omgi'] @ dic['A']
    Lterm  = dic['omgi'] @ dic['Ap'] @ inv(dic['L']) @ dic['Cm'] @ inv(dic['omgf']) @ dic['S'].T - np.identity(nvib)
    term2 = Lterm @ dic['omgi'] @ inv(dic['A'])
    nonexp = np.zeros(len(idx_list), dtype=complex)
    for j in range(len(idx_list)):
        tri = idx_list[j]
        nonexp[j] += -0.5*tLterm[tri[0],tri[2]] * oapsi[tri[1]]
        nonexp[j] += 0.5*oapsi[tri[0]] * tpLterm[tri[1],tri[2]]
        nonexp[j] += -0.5*(term1 - term2)[tri[0],tri[1]] * ceta[tri[2]]
        nonexp[j] += -1.0*oapsi[tri[0]] * oapsi[tri[1]] * ceta[tri[2]]

    rho_klm = np.real(exp_wif * rho0_tidx * nonexp)
    intgl = rule(tidx, tmax, tstep, rho_klm)
    return(intgl)


def integratePinQiPf(rho0_tidx, tidx, tmax, tstep, idx_list, exp_wif, dic):
    '''
    Return the array of integrated function (Rank 3)

    rho0_tidx: rho0 at time t
    t: single-point time
    tidx: index of time
    tmax: index of final integration time point
    tstep: step size
    beta: inverse temperature
    idx_list: list of triads of mode indices
    exp_wif: exponential term of adiabatic energy gap
    dic: dictionary of matrices and vectors
    intgl: triad-wise integrals arranged in a form of 1D array 
    ''' 
    tZterm = (dic['Cp'] @ inv(dic['J']) @ dic['Am'] - dic['Cm'] @ inv(dic['Jp']) @ dic['Ap']) @ dic['S']
    oceta = dic['omgf'] @ dic['Cm'] @ dic['eta']
    pZterm = dic['omgf'] @ (dic['Cm'] @ inv(dic['J']) @ dic['Ap'] + dic['Cp'] @ inv(dic['Jp']) @ dic['Am']) @ inv(dic['omgi'])
    psi = dic['d'] - (dic['S'] @ dic['Cp'] @ dic['eta'])
    oapsi = dic['omgi'] @ dic['A'] @ psi
    ppZterm = dic['omgf'] @ (dic['Cm'] @ inv(dic['J']) @ dic['Am'] - dic['Cp'] @ inv(dic['Jp']) @ dic['Ap'])
    nonexp = np.zeros(len(idx_list), dtype=complex)
    for j in range(len(idx_list)):
        tri = idx_list[j]
        nonexp[j] += 0.5*tZterm[tri[0],tri[1]] * oceta[tri[2]]
        nonexp[j] += -0.5*ppZterm[tri[2],tri[0]] * psi[tri[1]]
        nonexp[j] += -0.5*oapsi[tri[0]] * pZterm[tri[2],tri[1]]
        nonexp[j] += oapsi[tri[0]] * psi[tri[1]] * oceta[tri[2]]

    rho_klm = np.real(exp_wif * rho0_tidx * nonexp)
    intgl = rule(tidx, tmax, tstep, rho_klm)
    return(intgl)


def integratePfnQiPf(rho0_tidx, tidx, tmax, tstep, idx_list, exp_wif, dic):
    '''
    Return the array of integrated function (Rank 3)

    rho0_tidx: rho0 at time t
    t: single-point time
    tidx: index of time
    tmax: index of final integration time point
    tstep: step size
    beta: inverse temperature
    idx_list: list of triads of mode indices
    exp_wif: exponential term of adiabatic energy gap
    dic: dictionary of matrices and vectors
    intgl: triad-wise integrals arranged in a form of 1D array 
    ''' 
    Zterm = dic['omgf']@(dic['Cm']@inv(dic['J'])@dic['Ap'] - dic['Cp']@inv(dic['Jp'])@dic['Am'])@inv(dic['omgi'])
    psip = dic['d'] + (dic['S'] @ dic['Cp'] @ dic['eta'])
    psi = dic['d'] - (dic['S'] @ dic['Cp'] @ dic['eta'])
    soapsip = dic['S'].T @ dic['omgi'] @ dic['A'] @ psip
    Lterm = dic['S'].T @ dic['omgi'] @ (dic['Ap'] @ inv(dic['L']) @ dic['Cp'] - dic['Am'] @ inv(dic['Lp']) @ dic['Cm'])
    oceta = dic['omgf'] @ dic['Cm'] @ dic['eta']
    tpLterm = dic['S'].T @ dic['omgi'] @ (dic['Am'] @ inv(dic['Lp']) @ dic['Cp'] + dic['Ap'] @ inv(dic['L']) @ dic['Cm']) @ inv(dic['omgf']) @ dic['S'].T
    nonexp = np.zeros(len(idx_list), dtype=complex)
    for j in range(len(idx_list)):
        tri = idx_list[j]
        nonexp[j] += 0.5*Zterm[tri[0],tri[1]] * soapsip[tri[2]]
        nonexp[j] += 0.5*Lterm[tri[2],tri[0]] * psi[tri[1]]
        nonexp[j] += 0.5*oceta[tri[0]] * tpLterm[tri[2],tri[1]]
        nonexp[j] += -oceta[tri[0]] * psi[tri[1]] * soapsip[tri[2]]

    rho_klm = np.real(exp_wif * rho0_tidx * nonexp)
    intgl = rule(tidx, tmax, tstep, rho_klm)
    return(intgl)


def integrateQfPinPi(rho0_tidx, tidx, tmax, tstep, idx_list, exp_wif, dic):
    '''
    Return the array of integrated function (Rank 3)

    rho0_tidx: rho0 at time t
    t: single-point time
    tidx: index of time
    tmax: index of final integration time point
    tstep: step size
    beta: inverse temperature
    idx_list: list of triads of mode indices
    exp_wif: exponential term of adiabatic energy gap
    dic: dictionary of matrices and vectors
    intgl: triad-wise integrals arranged in a form of 1D array 
    ''' 
    ceta = dic['Cp'] @ dic['eta']
    Lpterm = dic['omgi'] @ dic['Am'] @ inv(dic['Lp']) @ dic['Cp'] @ inv(dic['omgf']) @ dic['S'].T - np.identity(nvib)
    term1 = Lpterm @ dic['omgi'] @ dic['A']
    Lterm  = dic['omgi'] @ dic['Ap'] @ inv(dic['L']) @ dic['Cm'] @ inv(dic['omgf']) @ dic['S'].T - np.identity(nvib)
    term2 = Lterm @ dic['omgi'] @ inv(dic['A'])
    tpLterm = dic['omgi'] @ (dic['Ap'] @ inv(dic['L']) @ dic['Cm'] + dic['Am'] @ inv(dic['Lp']) @ dic['Cp']) @ inv(dic['omgf'])
    oapsi = dic['omgi'] @ dic['A'] @ (dic['d'] - (dic['S'] @ dic['Cp'] @ dic['eta']))
    tLterm = dic['omgi'] @ (dic['Ap'] @ inv(dic['L']) @ dic['Cm'] - dic['Am'] @ inv(dic['Lp']) @ dic['Cp']) @ inv(dic['omgf'])
    nonexp = np.zeros(len(idx_list), dtype=complex)
    for j in range(len(idx_list)):
        tri = idx_list[j]
        nonexp[j] += -0.5*ceta[tri[0]] * (term1 - term2)[tri[2],tri[1]]
        nonexp[j] += 0.5*tpLterm[tri[1],tri[0]] * oapsi[tri[2]]
        nonexp[j] += -0.5*tLterm[tri[2],tri[0]] * oapsi[tri[1]]
        nonexp[j] += -1.0*ceta[tri[0]] * oapsi[tri[1]] * oapsi[tri[2]]

    rho_klm = np.real(exp_wif * rho0_tidx * nonexp)
    intgl = rule(tidx, tmax, tstep, rho_klm)
    return(intgl)


def integrateQfPinPf(rho0_tidx, tidx, tmax, tstep, idx_list, exp_wif, dic):
    '''
    Return the array of integrated function (Rank 3)

    rho0_tidx: rho0 at time t
    t: single-point time
    tidx: index of time
    tmax: index of final integration time point
    tstep: step size
    beta: inverse temperature
    idx_list: list of triads of mode indices
    exp_wif: exponential term of adiabatic energy gap
    dic: dictionary of matrices and vectors
    intgl: triad-wise integrals arranged in a form of 1D array 
    ''' 
    tppZterm = dic['Cp'] @ inv(dic['J']) @ dic['Am'] + dic['Cm'] @ inv(dic['Jp']) @ dic['Ap']
    oceta = dic['omgf'] @ dic['Cm'] @ dic['eta']
    hLterm = inv(dic['S']) @ (dic['Ap'] @ inv(dic['Lp']) @ dic['Cm'] - dic['Am'] @ inv(dic['L']) @ dic['Cp'])
    oapsi = dic['omgi'] @ dic['A'] @ (dic['d'] - (dic['S'] @ dic['Cp'] @ dic['eta']))
    ceta = dic['Cp'] @ dic['eta']
    ppZterm = dic['omgf'] @ (dic['Cm'] @ inv(dic['J']) @ dic['Am'] - dic['Cp'] @ inv(dic['Jp']) @ dic['Ap'])
    nonexp = np.zeros(len(idx_list), dtype=complex)
    for j in range(len(idx_list)):
        tri = idx_list[j]
        nonexp[j] += 0.5*tppZterm[tri[0],tri[1]] * oceta[tri[2]]
        nonexp[j] += -0.5*hLterm[tri[0],tri[2]] * oapsi[tri[1]]
        nonexp[j] += 0.5*ceta[tri[0]] * ppZterm[tri[2],tri[1]] 
        nonexp[j] += -1.0*ceta[tri[0]] * oapsi[tri[1]] * oceta[tri[2]]

    rho_klm = np.real(exp_wif * rho0_tidx * nonexp)
    intgl = rule(tidx, tmax, tstep, rho_klm)
    return(intgl)


def integratePfQinPi(rho0_tidx, tidx, tmax, tstep, idx_list, exp_wif, dic):
    '''
    Return the array of integrated function (Rank 3)

    rho0_tidx: rho0 at time t
    t: single-point time
    tidx: index of time
    tmax: index of final integration time point
    tstep: step size
    beta: inverse temperature
    idx_list: list of triads of mode indices
    exp_wif: exponential term of adiabatic energy gap
    dic: dictionary of matrices and vectors
    intgl: triad-wise integrals arranged in a form of 1D array 
    ''' 
    hpLterm = dic['Ap'] @ inv(dic['Lp']) @ dic['Cm'] + dic['Am'] @ inv(dic['L']) @ dic['Cp']
    psi = dic['d'] - (dic['S'] @ dic['Cp'] @ dic['eta'])
    oapsi = dic['omgi'] @ dic['A'] @ psi
    Lterm = dic['omgi'] @ (dic['Ap'] @ inv(dic['L']) @ dic['Cp'] - dic['Am'] @ inv(dic['Lp']) @ dic['Cm'])
    oceta = dic['omgf'] @ dic['Cm'] @ dic['eta']
    tLterm = dic['omgi'] @ (dic['Ap'] @ inv(dic['L']) @ dic['Cm'] - dic['Am'] @ inv(dic['Lp']) @ dic['Cp']) @ inv(dic['omgf']) @ dic['S'].T
    nonexp = np.zeros(len(idx_list), dtype=complex)
    for j in range(len(idx_list)):
        tri = idx_list[j]
        nonexp[j] += -0.5*hpLterm[tri[1],tri[0]] * oapsi[tri[2]]
        nonexp[j] += 0.5*Lterm[tri[2],tri[0]] * psi[tri[1]]
        nonexp[j] += -0.5*oceta[tri[0]] * tLterm[tri[2],tri[1]] 
        nonexp[j] += oceta[tri[0]] * psi[tri[1]] * oapsi[tri[2]]

    rho_klm = np.real(exp_wif * rho0_tidx * nonexp)
    intgl = rule(tidx, tmax, tstep, rho_klm)
    return(intgl)


def integratePfQinPf(rho0_tidx, tidx, tmax, tstep, idx_list, exp_wif, dic):
    '''
    Return the array of integrated function (Rank 3)

    rho0_tidx: rho0 at time t
    t: single-point time
    tidx: index of time
    tmax: index of final integration time point
    tstep: step size
    beta: inverse temperature
    idx_list: list of triads of mode indices
    exp_wif: exponential term of adiabatic energy gap
    dic: dictionary of matrices and vectors
    intgl: triad-wise integrals arranged in a form of 1D array 
    ''' 
    pZterm = dic['omgf'] @ (dic['Cm'] @ inv(dic['J']) @ dic['Ap'] + dic['Cp'] @ inv(dic['Jp']) @ dic['Am']) @ inv(dic['omgi'])
    psi = dic['d'] - (dic['S'] @ dic['Cp'] @ dic['eta'])
    soapsi = dic['S'].T @ dic['omgi'] @ dic['A'] @ psi
    Lterm = dic['S'].T @ dic['omgi'] @ (dic['Ap'] @ inv(dic['L']) @ dic['Cp'] - dic['Am'] @ inv(dic['Lp']) @ dic['Cm'])
    oceta = dic['omgf'] @ dic['Cm'] @ dic['eta']
    tLterm = dic['S'].T @ dic['omgi'] @ (dic['Ap'] @ inv(dic['L']) @ dic['Cm'] - dic['Am'] @ inv(dic['Lp']) @ dic['Cp']) @ inv(dic['omgf']) @ dic['S'].T
    nonexp = np.zeros(len(idx_list), dtype=complex)
    for j in range(len(idx_list)):
        tri = idx_list[j]
        nonexp[j] += -0.5*pZterm[tri[0],tri[1]] * soapsi[tri[2]]
        nonexp[j] += 0.5*Lterm[tri[2],tri[0]] * psi[tri[1]]
        nonexp[j] += -0.5*oceta[tri[0]] * tLterm[tri[2],tri[1]] 
        nonexp[j] += oceta[tri[0]] * psi[tri[1]] * soapsi[tri[2]]

    rho_klm = np.real(exp_wif * rho0_tidx * nonexp)
    intgl = rule(tidx, tmax, tstep, rho_klm)
    return(intgl)


def integrateQfPinPiQf(rho0_tidx, tidx, tmax, tstep, idx_list, exp_wif, dic):
    '''
    Return the array of integrated function (Rank 4)

    rho0_tidx: rho0 at time t
    t: single-point time
    tidx: index of time
    tmax: index of final integration time point
    tstep: step size
    beta: inverse temperature
    idx_list: list of triads of mode indices
    exp_wif: exponential term of adiabatic energy gap
    dic: dictionary of matrices and vectors
    intgl: tetrad-wise integrals arranged in a form of 1D array 
    ''' 
    tpLterm = dic['omgi'] @ (dic['Ap'] @ inv(dic['L']) @ dic['Cm'] + dic['Am'] @ inv(dic['Lp']) @ dic['Cp']) @ inv(dic['omgf'])
    psi = dic['d'] - (dic['S'] @ dic['Cp'] @ dic['eta'])
    oapsi = dic['omgi'] @ dic['A'] @ psi
    ceta = dic['Cp'] @ dic['eta']
    tLterm = dic['omgi'] @ (dic['Ap'] @ inv(dic['L']) @ dic['Cm'] - dic['Am'] @ inv(dic['Lp']) @ dic['Cp']) @ inv(dic['omgf'])
    long_tLterm = tLterm @ dic['S'].T @ dic['omgi'] @ dic['A']
    OAA = dic['omgi'] @ (dic['A'] - inv(dic['A']))
    pLterm = inv(dic['S']) @ (dic['Ap'] @ inv(dic['Lp']) @ dic['Cp'] - dic['Am'] @ inv(dic['L']) @ dic['Cm']) @ inv(dic['omgf'])
    nonexp = np.zeros(len(idx_list), dtype=complex)
    for j in range(len(idx_list)):
        tet = idx_list[j]
        nonexp[j] += (0.75*tpLterm[tet[1],tet[0]] - 0.5*ceta[tet[0]] * oapsi[tet[1]]) * tpLterm[tet[2],tet[3]]
        nonexp[j] += -0.5*tpLterm[tet[1],tet[0]] * oapsi[tet[2]] * ceta[tet[3]]
        nonexp[j] += 0.5*tLterm[tet[2],tet[0]] * oapsi[tet[1]] * ceta[tet[3]]
        nonexp[j] += 0.5*ceta[tet[0]] * tLterm[tet[1],tet[3]] * oapsi[tet[2]]
        nonexp[j] += -0.5*ceta[tet[0]] * (long_tLterm + OAA)[tet[2],tet[1]] * ceta[tet[3]]
        nonexp[j] += -0.5*pLterm[tet[0],tet[3]] * (0.5*OAA[tet[1],tet[2]] - oapsi[tet[1]]*oapsi[tet[2]])
        nonexp[j] += ceta[tet[0]] * oapsi[tet[1]] * psi[tet[2]] * ceta[tet[3]]

    rho_klmn = np.real(exp_wif * rho0_tidx * nonexp)
    intgl = rule(tidx, tmax, tstep, rho_klmn)
    return(intgl)


def integrateQfPinQiPf(rho0_tidx, tidx, tmax, tstep, idx_list, exp_wif, dic):
    '''
    Return the array of integrated function (Rank 4)

    rho0_tidx: rho0 at time t
    t: single-point time
    tidx: index of time
    tmax: index of final integration time point
    tstep: step size
    beta: inverse temperature
    idx_list: list of triads of mode indices
    exp_wif: exponential term of adiabatic energy gap
    dic: dictionary of matrices and vectors
    intgl: tetrad-wise integrals arranged in a form of 1D array 
    ''' 
    pLterm = (dic['Ap'] @ inv(dic['Lp']) @ dic['Cp'] - dic['Am'] @ inv(dic['L']) @ dic['Cm']) @ inv(dic['omgf'])
    Lterm = dic['omgi'] @ (dic['Ap'] @ inv(dic['L']) @ dic['Cp'] - dic['Am'] @ inv(dic['Lp']) @ dic['Cm'])
    psi = dic['d'] - (dic['S'] @ dic['Cp'] @ dic['eta'])
    oapsi = dic['omgi'] @ dic['A'] @ psi
    oceta = dic['omgf'] @ dic['Cm'] @ dic['eta']
    ceta = dic['Cp'] @ dic['eta']
    hLterm = inv(dic['S']) @ (dic['Ap'] @ inv(dic['Lp']) @ dic['Cm'] - dic['Am'] @ inv(dic['L']) @ dic['Cp'])
    tpLterm = dic['omgi'] @ (dic['Ap'] @ inv(dic['L']) @ dic['Cm'] + dic['Am'] @ inv(dic['Lp']) @ dic['Cp']) @ inv(dic['omgf'])
    tLterm = dic['omgi'] @ (dic['Ap'] @ inv(dic['L']) @ dic['Cm'] - dic['Am'] @ inv(dic['Lp']) @ dic['Cp']) @ inv(dic['omgf']) @ dic['S'].T
    hpLterm = dic['Ap'] @ inv(dic['Lp']) @ dic['Cm'] + dic['Am'] @ inv(dic['L']) @ dic['Cp']
    nonexp = np.zeros(len(idx_list), dtype=complex)
    for j in range(len(idx_list)):
        tet = idx_list[j]
        nonexp[j] += pLterm[tet[2],tet[0]] * (0.75*Lterm[tet[1],tet[3]] + 0.5*oapsi[tet[1]]*oceta[tet[3]])
        nonexp[j] += -0.5*ceta[tet[0]] * Lterm[tet[1],tet[3]] * psi[tet[2]]
        nonexp[j] += -0.5*hLterm[tet[0],tet[3]] * oapsi[tet[1]] * psi[tet[2]]
        nonexp[j] += 0.5*tpLterm[tet[1],tet[0]] * psi[tet[2]] * oceta[tet[3]]
        nonexp[j] += 0.5*ceta[tet[0]] * tLterm[tet[1],tet[2]] * oceta[tet[3]]        
        nonexp[j] += 0.5*ceta[tet[0]] * oapsi[tet[1]] * hpLterm[tet[2],tet[3]]
        nonexp[j] += -ceta[tet[0]] * oapsi[tet[1]] * psi[tet[2]] * oceta[tet[3]]

    rho_klmn = np.real(exp_wif * rho0_tidx * nonexp)
    intgl = rule(tidx, tmax, tstep, rho_klmn)
    return(intgl)


def integratePfQinPiQf(rho0_tidx, tidx, tmax, tstep, idx_list, exp_wif, dic):
    '''
    Return the array of integrated function (Rank 4)

    rho0_tidx: rho0 at time t
    t: single-point time
    tidx: index of time
    tmax: index of final integration time point
    tstep: step size
    beta: inverse temperature
    idx_list: list of triads of mode indices
    exp_wif: exponential term of adiabatic energy gap
    dic: dictionary of matrices and vectors
    intgl: tetrad-wise integrals arranged in a form of 1D array 
    ''' 
    hpLterm = dic['Ap'] @ inv(dic['Lp']) @ dic['Cm'] + dic['Am'] @ inv(dic['L']) @ dic['Cp']
    psi = dic['d'] - (dic['S'] @ dic['Cp'] @ dic['eta'])
    oapsi = dic['omgi'] @ dic['A'] @ psi
    ceta = dic['Cp'] @ dic['eta']
    Lterm = dic['omgi'] @ (dic['Ap'] @ inv(dic['L']) @ dic['Cp'] - dic['Am'] @ inv(dic['Lp']) @ dic['Cm'])
    tpZterm = (dic['Cp'] @ inv(dic['J']) @ dic['Ap'] - dic['Cm'] @ inv(dic['Jp']) @ dic['Am']) @ inv(dic['omgi'])
    hLterm = inv(dic['S']) @ (dic['Ap'] @ inv(dic['Lp']) @ dic['Cm'] - dic['Am'] @ inv(dic['L']) @ dic['Cp'])
    oceta = dic['omgf'] @ dic['Cm'] @ dic['eta']
    tLterm = dic['omgi'] @ (dic['Ap'] @ inv(dic['L']) @ dic['Cm'] - dic['Am'] @ inv(dic['Lp']) @ dic['Cp']) @ inv(dic['omgf']) @ dic['S'].T
    tpLterm = dic['omgi'] @ (dic['Ap'] @ inv(dic['L']) @ dic['Cm'] + dic['Am'] @ inv(dic['Lp']) @ dic['Cp']) @ inv(dic['omgf'])
    nonexp = np.zeros(len(idx_list), dtype=complex)
    for j in range(len(idx_list)):
        tet = idx_list[j]
        nonexp[j] += 0.5*hpLterm[tet[1],tet[0]] * oapsi[tet[2]] * ceta[tet[3]]
        nonexp[j] += Lterm[tet[2],tet[0]] * (0.75*tpZterm[tet[3],tet[1]] - 0.5*psi[tet[1]] * ceta[tet[3]])
        nonexp[j] += -0.5*hLterm[tet[3],tet[0]] * psi[tet[1]] * oapsi[tet[2]]
        nonexp[j] += 0.5*oceta[tet[0]] * tLterm[tet[2],tet[1]] * ceta[tet[3]]
        nonexp[j] += 0.5*oceta[tet[0]] * tpZterm[tet[3],tet[1]] * oapsi[tet[2]]
        nonexp[j] += 0.5*oceta[tet[0]] * psi[tet[1]] * tpLterm[tet[2],tet[3]]
        nonexp[j] += -oceta[tet[0]] * psi[tet[1]] * oapsi[tet[2]] * ceta[tet[3]]

    rho_klmn = np.real(exp_wif * rho0_tidx * nonexp)
    intgl = rule(tidx, tmax, tstep, rho_klmn)
    return(intgl)


def integratePfQinQiPf(rho0_tidx, tidx, tmax, tstep, idx_list, exp_wif, dic):
    '''
    Return the array of integrated function (Rank 4)

    rho0_tidx: rho0 at time t
    t: single-point time
    tidx: index of time
    tmax: index of final integration time point
    tstep: step size
    beta: inverse temperature
    idx_list: list of triads of mode indices
    exp_wif: exponential term of adiabatic energy gap
    dic: dictionary of matrices and vectors
    intgl: tetrad-wise integrals arranged in a form of 1D array 
    ''' 
    hpLterm = dic['Ap'] @ inv(dic['Lp']) @ dic['Cm'] + dic['Am'] @ inv(dic['L']) @ dic['Cp']
    psi = dic['d'] - (dic['S'] @ dic['Cp'] @ dic['eta'])
    soapsi = dic['S'].T @ dic['omgi'] @ dic['A'] @ psi
    hLterm = dic['Ap'] @ inv(dic['Lp']) @ dic['Cm'] - dic['Am'] @ inv(dic['L']) @ dic['Cp']
    Lterm = dic['S'].T @ dic['omgi'] @ (dic['Ap'] @ inv(dic['L']) @ dic['Cp'] - dic['Am'] @ inv(dic['Lp']) @ dic['Cm'])
    oceta = dic['omgf'] @ dic['Cm'] @ dic['eta']
    pLterm = (dic['Ap'] @ inv(dic['Lp']) @ dic['Cp'] - dic['Am'] @ inv(dic['L']) @ dic['Cm']) @ inv(dic['omgf']) @ dic['S'].T
    tLterm = dic['S'].T @ dic['omgi'] @ (dic['Ap'] @ inv(dic['L']) @ dic['Cm'] - dic['Am'] @ inv(dic['Lp']) @ dic['Cp']) @ inv(dic['omgf']) @ dic['S'].T
    tpLterm = dic['S'].T @ dic['omgi'] @ (dic['Ap'] @ inv(dic['L']) @ dic['Cm'] + dic['Am'] @ inv(dic['Lp']) @ dic['Cp']) @ inv(dic['omgf']) @ dic['S'].T
    nonexp = np.zeros(len(idx_list), dtype=complex)
    for j in range(len(idx_list)):
        tet = idx_list[j]
        nonexp[j] += -0.5*hpLterm[tet[1],tet[0]] * psi[tet[2]] * soapsi[tet[3]]
        nonexp[j] += -0.5*hLterm[tet[2],tet[0]] * psi[tet[1]] * soapsi[tet[3]]
        nonexp[j] += Lterm[tet[3],tet[0]] * (0.75*pLterm[tet[2],tet[1]] + 0.5*psi[tet[1]]*psi[tet[2]])
        nonexp[j] += 0.5*oceta[tet[0]] * pLterm[tet[2],tet[1]] * soapsi[tet[3]]
        nonexp[j] += -0.5*oceta[tet[0]] * tLterm[tet[3],tet[1]] * psi[tet[2]]
        nonexp[j] += 0.5*oceta[tet[0]] * psi[tet[1]] * tpLterm[tet[3],tet[2]]
        nonexp[j] += oceta[tet[0]] * psi[tet[1]] * psi[tet[2]] * soapsi[tet[3]]

    rho_klmn = np.real(exp_wif * rho0_tidx * nonexp)
    intgl = rule(tidx, tmax, tstep, rho_klmn)
    return(intgl)


# def integrate_nPQ(rho0_tidx, t, tidx, beta, idx_list, exp_wif, dic):
#     '''
#     Return the array of integrated function (Rank 2)

#     rho0_tidx: rho0 at time t    
#     t: single-point time
#     tidx: index of time
#     beta: inverse temperature
#     idx_list: list of pairs of mode indices
#     exp_wif: exponential term of adiabatic energy gap
#     dic: dictionary of matrices and vectors
#     intgl: 2D array of integral 
#     '''
#     tDeliomgf = dic['S'].T@dic['omgi']@(dic['Ap']@inv(dic['L'])@dic['Cm'] - (dic['Am']@inv(dic['Lp'])@dic['Cp']))@inv(dic['omgf'])
#     etap = dic['Cp'] @ dic['eta']
#     Appsi = (dic['S'].T @ dic['omgi'] @ dic['A']) @ (dic['d'] - (dic['S'] @ dic['Cp'] @ dic['eta']))
#     tmax = len(t_all)-1
#     nonexp = np.zeros(len(idx_list), dtype=complex)
#     for j in range(len(idx_list)):
#         pair = idx_list[j]
#         nonexp[j] += 0.5j*tDeliomgf[pair[0],pair[1]] + 1.0j*etap[pair[1]] * Appsi[pair[0]]
    
#     rho_kl = np.real(exp_wif * rho0_tidx * nonexp)
#     intgl = simpson(tidx, tmax, rho_kl)
#     return(intgl)


# def integrateQnP(rho0_tidx, t, tidx, beta, idx_list, exp_wif, dic):
#     '''
#     Return the array of integrated function (Rank 2)

#     rho0_tidx: rho0 at time t    
#     t: single-point time
#     tidx: index of time
#     beta: inverse temperature
#     idx_list: list of pairs of mode indices
#     exp_wif: exponential term of adiabatic energy gap
#     dic: dictionary of matrices and vectors
#     intgl: 2D array of integral 
#     '''
#     tDeliomgf = dic['S'].T@dic['omgi']@(dic['Ap']@inv(dic['L'])@dic['Cm'] - (dic['Am']@inv(dic['Lp'])@dic['Cp']))@inv(dic['omgf'])
#     etap = dic['Cp'] @ dic['eta']
#     Appsi = (dic['S'].T @ dic['omgi'] @ dic['A']) @ (dic['d'] - (dic['S'] @ dic['Cp'] @ dic['eta']))
#     tmax = len(t_all)-1
#     nonexp = np.zeros(len(idx_list), dtype=complex)
#     for j in range(len(idx_list)):
#         pair = idx_list[j]
#         nonexp[j] += 0.5j*tDeliomgf[pair[1],pair[0]] + 1.0j*etap[pair[0]] * Appsi[pair[1]]
    
#     rho_kl = np.real(exp_wif * rho0_tidx * nonexp)
#     intgl = simpson(tidx, tmax, rho_kl)
#     return(intgl)


# def integrateQnPQ(rho0_tidx, t, tidx, beta, idx_list, exp_wif, dic):
#     '''
#     Return the array of integrated function (Rank 3)

#     rho0_tidx: rho0 at time t
#     t: single-point time
#     tidx: index of time
#     beta: inverse temperature
#     idx_list: list of triads of mode indices
#     exp_wif: exponential term of adiabatic energy gap
#     dic: dictionary of matrices and vectors
#     intgl: 3D array of integral  
#     '''
#     Omic = inv(dic['S'])@(dic['Ap']@inv(dic['Lp'])@dic['Cp'] - (dic['Am']@inv(dic['L'])@dic['Cm']))@inv(dic['omgf'])
#     Appsi = (dic['S'].T @ dic['omgi'] @ dic['A']) @ (dic['d'] - (dic['S'] @ dic['Cp'] @ dic['eta']))
#     etap = dic['Cp'] @ dic['eta'] 
#     tDeliomgf = dic['S'].T@dic['omgi']@(dic['Ap']@inv(dic['L'])@dic['Cm'] - (dic['Am']@inv(dic['Lp'])@dic['Cp']))@inv(dic['omgf'])
#     tmax = len(t_all)-1
#     nonexp = np.zeros(len(idx_list), dtype=complex)
#     for j in range(len(idx_list)):
#         tri = idx_list[j]
#         nonexp[j] += -1.0j*(0.5*Omic[tri[2],tri[0]] + etap[tri[0]]*etap[tri[2]]) * Appsi[tri[1]]
#         nonexp[j] += -0.5j*(tDeliomgf[tri[1],tri[0]] * etap[tri[2]] + tDeliomgf[tri[1],tri[2]] * etap[tri[0]])
    
#     rho_klm = np.real(exp_wif * rho0_tidx * nonexp)
#     intgl = simpson(tidx, tmax, rho_klm)
#     return(intgl)


# def integratePQnP(rho0_tidx, t, tidx, beta, idx_list, exp_wif, dic):
#     '''
#     Return the array of integrated function (Rank 3)

#     rho0_tidx: rho0 at time t
#     t: single-point time
#     tidx: index of time
#     beta: inverse temperature
#     idx_list: list of triads of mode indices
#     exp_wif: exponential term of adiabatic energy gap
#     dic: dictionary of matrices and vectors
#     intgl: 3D array of integral  
#     '''
#     Zpterm = dic['omgf'] @ (dic['Cm']@inv(dic['Am']@dic['S']@dic['Cp'] + (dic['Ap']@inv(dic['omgi'])@inv(dic['S'].T)@dic['omgf']@dic['Cm'])) @ dic['Ap'] \
#              + (dic['Cp']@inv(dic['Ap']@dic['S']@dic['Cm'] + (dic['Am']@inv(dic['omgi'])@inv(dic['S'].T)@dic['omgf']@dic['Cp']))@dic['Am'])) @ inv(dic['omgi']) @ inv(dic['S'].T)
#     eta2p  = dic['omgf'] @ dic['Cm'] @ dic['eta']
#     etap   = dic['Cp'] @ dic['eta'] 
#     Appsi  = (dic['S'].T @ dic['omgi'] @ dic['A']) @ (dic['d'] - (dic['S'] @ dic['Cp'] @ dic['eta']))
#     tDeliomgf = dic['S'].T@dic['omgi']@(dic['Ap']@inv(dic['L'])@dic['Cm'] - (dic['Am']@inv(dic['Lp'])@dic['Cp']))@inv(dic['omgf'])
#     Del    = dic['S'].T @ dic['omgi'] @ (dic['Ap'] @ inv(dic['L']) @ dic['Cp'] - (dic['Am'] @ inv(dic['Lp']) @ dic['Cm']))
#     tmax = len(t_all)-1
#     nonexp = np.zeros(len(idx_list), dtype=complex)
#     for j in range(len(idx_list)):
#         tri = idx_list[j]
#         nonexp[j] += (-0.5*Zpterm[tri[0],tri[1]] - eta2p[tri[0]]*etap[tri[1]]) * Appsi[tri[2]]
#         nonexp[j] += -0.5*tDeliomgf[tri[2],tri[1]] * eta2p[tri[0]]
#         nonexp[j] += -0.5*Del[tri[2],tri[0]] * etap[tri[1]]

#     rho_klm = np.real(exp_wif * rho0_tidx * nonexp)
#     intgl = simpson(tidx, tmax, rho_klm)
#     return(intgl)


# def integratePQnPQ(rho0_tidx, t, tidx, beta, idx_list, exp_wif, dic):
#     '''
#     Return the array of integrated function (Rank 4)

#     rho0_tidx: rho0 at time t
#     t       : single-point time
#     tidx    : index of time
#     beta    : inverse temperature
#     idx_list: list of tetrads of mode indices
#     exp_wif : exponential term of adiabatic energy gap
#     dic     : dictionary of matrices and vectors
#     intgl   : 4D array of integral  

#     Note: 'rho_klmn' is a full-dimensional 4D array. On the other hand,
#           'intgl' is only those for pre-selected indices of significant
#           couplings.
#     '''
#     Omic   = inv(dic['S'])@(dic['Ap']@inv(dic['Lp'])@dic['Cp'] - (dic['Am']@inv(dic['L'])@dic['Cm']))@inv(dic['omgf'])
#     Del    = dic['S'].T @ dic['omgi'] @ (dic['Ap'] @ inv(dic['L']) @ dic['Cp'] - (dic['Am'] @ inv(dic['Lp']) @ dic['Cm']))
#     eta2p  = dic['omgf'] @ dic['Cm'] @ dic['eta']
#     etap   = dic['Cp'] @ dic['eta'] 
#     Appsi  = (dic['S'].T @ dic['omgi'] @ dic['A']) @ (dic['d'] - (dic['S'] @ dic['Cp'] @ dic['eta']))
#     Appsip = (dic['S'].T @ dic['omgi'] @ dic['A']) @ (dic['d'] - 2.0*(dic['S'] @ dic['Cp'] @ dic['eta']))
#     Chi    = inv(dic['S']) @ dic['Ap'] @ inv(dic['Lp']) @ dic['Cm']
#     Chip   = inv(dic['S']) @ dic['Am'] @ inv(dic['L']) @ dic['Cp']
#     A2pd   = dic['S'].T @ dic['omgi'] @ inv(dic['A']) @ dic['d']
#     ApSCe  = dic['S'].T @ dic['omgi'] @ dic['A'] @ dic['S'] @ dic['Cp'] @ dic['eta']
#     A2pSCe = dic['S'].T @ dic['omgi'] @ inv(dic['A']) @ dic['S'] @ dic['Cp'] @ dic['eta']
#     Chi2p  = inv(dic['S']) @ dic['Am'] @ inv(dic['L']) @ dic['Cm'] @ dic['C']
#     tmax   = len(t_all)-1
#     nonexp = np.zeros(len(idx_list), dtype=complex)
#     for j in range(len(idx_list)):
#         tet = idx_list[j]
#         nonexp[j] += Del[tet[2],tet[0]] * (0.75*Omic[tet[3],tet[1]] + 0.5*etap[tet[3]]*etap[tet[1]])
#         nonexp[j] += Appsi[tet[2]] * (0.5*eta2p[tet[0]]*Omic[tet[3],tet[1]] + 4.0*eta2p[tet[0]]*etap[tet[1]]*etap[tet[3]])
#         nonexp[j] += 2.0*Appsip[tet[2]] * (etap[tet[3]]*Chi[tet[1],tet[0]] + etap[tet[1]]*Chi[tet[3],tet[0]])
#         nonexp[j] += 2.0*(etap[tet[3]]*Chip[tet[1],tet[0]] - etap[tet[1]]*Chip[tet[3],tet[0]]) * (A2pd - ApSCe)[tet[2]]
#         nonexp[j] += -2.0*A2pSCe[tet[2]] * (etap[tet[3]]*Chi2p[tet[1],tet[0]] - etap[tet[1]]*Chi2p[tet[3],tet[0]])
        
#     rho_klmn = np.real(exp_wif * rho0_tidx * nonexp)
#     intgl = simpson(tidx, tmax, rho_klmn)
#     return(intgl)


def get_matvec(t, beta, omg_S_d):
    """
    Routine to construct a library of matrices and vectors that are 
    the ingredients of TVCFs
    
    t      : current time
    beta   : inverse temperature
    omg_S_d: a collection of frequency matrices, Duschinsky matrix, and displacement vector
    
    Returns
    matvec : library of obtained matrices and vectors
    """
    ''' Define matrices '''
    omg_i, omg_f, S, d = omg_S_d
    C  = 1j * np.tan(0.5 * np.diag(omg_f) * t)
    expG = np.exp(-1j * np.diag(omg_f) * t) # ground state freq exponential
    Cp = 1 + expG 
    Cm = 1 - expG

    A  = 1j * np.tan(0.5 * (-1j * np.diag(omg_i) * beta - np.diag(omg_i) * t))
    expb = np.exp(beta * np.diag(omg_i)) # thermal energy exponential
    expE = np.exp(1j * np.diag(omg_i) * t) # excited state freq exponential
    Ap = (expb + expE) / (expb - 1)
    Am = (expb - expE) / (expb - 1)    

    C, A   = np.diag(C), np.diag(A)
    Cp, Cm = np.diag(Cp), np.diag(Cm)
    Ap, Am = np.diag(Ap), np.diag(Am)

    J  = Am @ S @ Cp + (Ap @ inv(omg_i) @ inv(S.T) @ omg_f @ Cm) # J
    Jp  = Ap @ S @ Cm + (Am @ inv(omg_i) @ inv(S.T) @ omg_f @ Cp) # J prime
    L  = Cp @ inv(S) @ Am + (Cm @ inv(omg_f) @ S.T @ omg_i @ Ap) # L
    Lp  = Cm @ inv(S) @ Ap + (Cp @ inv(omg_f) @ S.T @ omg_i @ Am) # L prime

    ''' Define vectors '''
    eta  = inv(J) @ Am @ d # vector eta

    ''' Define a dictionary of matrices and vectors '''
    matvec = {'S': S, 'omgi': omg_i, 'omgf': omg_f, 'A': A, 'C': C,
              'Am': Am, 'Ap': Ap, 'Cm': Cm, 'Cp': Cp, 'd': d, 'eta': eta,
              'L': L, 'Lp': Lp, 'J': J, 'Jp': Jp}
    
    return matvec


def get_rho0(tgrid, beta, omg_S_d):
    """
    Routine to compute smooth rho0 with proper phase treatment
    
    tgrid  : 1D array of time points
    beta   : inverse temperature
    omg_S_d: a collection of frequency matrices, Duschinsky matrix, and displacement vector
    
    Returns
    rho0 : 1D array of rho0 as a function of time
    """
    
    omgi, omgf, S, d = omg_S_d
    
    ''' Compute thermal vibrational partition function '''
    zpe = np.sum(0.5 * np.diag(omgi))
    PF = 1.0
    for i in range(nvib):
        PF *= (1 - np.exp(-beta * omgi[i,i]))**-1
    PF *= np.exp(-beta * zpe)   
    
    out = np.zeros(len(tgrid), dtype=complex)
    nsign = 0 # sign change counter to fix the sign of complex sqrt in prefactor calculation
    for t in tgrid:
        tidx = list(tgrid).index(t)
    
        ''' Define matrices and vectors '''
        matvec = get_matvec(t, beta, omg_S_d)
        expb = np.exp(beta * np.diag(omgi)) # thermal energy exponential
        expE = np.exp(1j * np.diag(omgi) * t) # excited state freq exponential
        zeta = omgi @ matvec['A'] @ d  # vector zeta
        psi  = d - (S @ matvec['Cp'] @ matvec['eta']) # vector psi 
    
        ''' 
        Beginning of prefactor calculation
        Log scale is used to avoid numerical breakdowns due to extremely small numbers
        '''
        plog = 0.0
        plog += 2*mpmath.log(PF**-1)
    
        ''' det(Ap + Am) '''
        minus = 0 # count the number of minus signs
        for i in range(nvib):
            ptmp  = 0.0
            ptmp += 2 * expb[i]/(expb[i]-1)
            if ptmp.real > 0:
                ptmp = np.log(ptmp)
                minus += 0
            elif ptmp.real < 0:
                ptmp = np.log(-1 * ptmp)
                minus += 1
            plog += ptmp
    
        ''' det(Ap - Am) '''
        for i in range(nvib):
            ptmp  = 0.0
            ptmp += 2 * expE[i]/(expb[i]-1)
            if ptmp.real > 0:
                ptmp = np.log(ptmp)
                minus += 0
            elif ptmp.real < 0:
                ptmp = np.log(-1 * ptmp)
                minus += 1
            plog += ptmp
    
        ''' det(Cp - Cm) '''
        pp = np.linalg.det(matvec['Cp'] - matvec['Cm'])
    
        ''' det(JL)^{-1} '''
        JLsign, JLdet = np.linalg.slogdet(matvec['J'] @ matvec['L'])
    
        ''' Construct prefactor '''
        pp *= (np.linalg.det(S.T @ S))**-1
        prefac = (-1)**minus * mpmath.exp(plog) * pp * (JLsign * np.exp(JLdet))**-1
        if tidx == 0:
            pf0 = pf_hist = prefac
            smooth_pf = pf0**0.5
        else:
            """ Fix the discontinuity in prefactor """
            # only works when prefac is before square-rooted
            if prefac.real < 0 and prefac.imag * pf_hist.imag < 0:
                nsign += 1
            smooth_pf = (-1)**nsign * prefac**0.5
            
        if t == tgrid[0]:
            if np.real(smooth_pf) < 0:
                signflip = -1 # Pick a positive root at t = 0 (!!! an arbitrary choice !!!)
            else:
                signflip = 1
        smooth_pf *= signflip
        smooth_pf *= (-1/np.sqrt(2))**(nvib)
        pf_hist = prefac
    
        ''' 
        Exponential term 
        '''
        expterm = np.exp(-zeta @ psi)
        #expterm_array[tidx] = expterm
    
        ''' 
        rho0
        '''
        out[tidx] = smooth_pf * expterm
        
    return out


def integrate_CF(order, rho0, width, tgrid, ta, tb, maxidx, tstep, beta, omg_S_d, proc_id, idx_dict, coup_dict, tlist):
    """
    This is the to-be-parallelized function to be iterated over time to 
    1. compute all of the relevant matrices and vectors
    2. construct all CFs, integrate them, and store them according to the
       normal mode indices.
    3. weight the integrals by a proper coupling term           
 
    order    : the perturbative order of simulation
    rho0     : the pre-evaluated Frank-Condon correlation function
    width    : The width of Gaussian envelope
    tgrid    : list of time points at which CFs are evaluated
    ta       : Lower limit of integration
    tb       : Upper limit of integration
    maxidx   : index of final integration time point
    tstep    : step size
    beta     : inverse temperature
    omg_S_d  : a collection of frequency matrices, Duschinsky matrix, and displacement vector
    proc_id  : type identifications of each of the 0th, 1st, and 2nd
               processes
    idx_dict : dictionary of classified normal modes indices necessary
               for relevant CF computation
    coup_dict: dictionary of various compound coupling terms with 
               above-threshold magnitudes 
    tlist    : time points of evaluation selected for parallelization
    """
    
    '''
    Unpack the dictionary of vibrational mode indices with 
    significant couplings
    '''

    if proc_id[0,0] == 2: # If the overall process is ISC
        if idx_dict._nQ is not None:
            intgl_nQ = np.zeros(len(idx_dict._nQ), dtype=float)
            int_nQ_storage, CF_nQ_storage = np.zeros(len(idx_dict._nQ)), np.zeros((int((tb-ta)/tstep), len(idx_dict._nQ)))
        if idx_dict.QnQ is not None:
            intglQnQ = np.zeros(len(idx_dict.QnQ), dtype=float)
            intQQ_storage, CFQQ_storage = np.zeros(len(idx_dict.QnQ)), np.zeros((int((tb-ta)/tstep), len(idx_dict.QnQ)))
        if order > 1:
            intgl_nP   = [np.zeros(len(idx_dict._nP[m1]), dtype=complex) for m1 in range(nintmed)]
            intgl_nPQ  = [np.zeros(len(idx_dict._nPQ[m1]), dtype=complex) for m1 in range(nintmed)]
            intglQnP   = [np.zeros(len(idx_dict.QnP[m1]), dtype=complex) for m1 in range(nintmed)]
            intglQnPQ  = [np.zeros(len(idx_dict.QnPQ[m1]), dtype=complex) for m1 in range(nintmed)]
            intglPnP   = [[np.zeros(len(idx_dict.PnP[m1][m2])) for m2 in range(nintmed)] for m1 in range(nintmed)]
            intglPnPQ  = [[np.zeros(len(idx_dict.PnPQ[m1][m2])) for m2 in range(nintmed)] for m1 in range(nintmed)]
            intglPQnP  = [[np.zeros(len(idx_dict.PQnP[m1][m2])) for m2 in range(nintmed)] for m1 in range(nintmed)]
            intglPQnPQ = [[np.zeros(len(idx_dict.PQnPQ[m1][m2])) for m2 in range(nintmed)] for m1 in range(nintmed)]
    elif proc_id[0,0] == 1: # If the overall process is IC
        intglPnP = np.zeros(len(idx_dict.PnP))
#        if order > 1:
#            ...

    intgl0 = 0.0 
    int0_storage, CF0_storage = 0.0, np.zeros(int((tb-ta)/tstep))
    # int_nP_storage = [np.zeros(len(idx_dict._nP[m1]), dtype=complex) for m1 in range(nintmed)]
    # CF_nP_storage  = [np.zeros((int((tb-ta)/tstep), len(idx_dict._nP[m1])), dtype=complex) for m1 in range(nintmed)]
    k0_intgl_storage, k0_CF_storage = [], []
    for t in tqdm(tlist):
        tidx = tgrid.index(t)
        exp_wif  = np.exp(1j*(Ei-Ef)*t) # complex exponential of adiabatic energy gap
        exp_wif *= np.exp(-0.5 * (width * t)**2) # Multiply the Gaussian envelope to loosen the energy conservation criteria

        ''' Define matrices and vectors for correlation functions '''
        matvec = get_matvec(t, beta, omg_S_d)
    
        ''' 
        Compute the integrals of correlation functions 
    
        Each of them is 2Xed for t != 0 since the real part of integral 
        is symmetric about t=0
        '''
        mirror_factor = 2
        if tidx == 0:
            mirror_factor = 1 # no reflection for t = 0
            
        """
        k0 rate
        """
        if proc_id[0,0] == 2: # If the overall process is ISC
            tmp0      = integrate0(rho0[tidx], tidx, maxidx, tstep, exp_wif, matvec)
            intgl0   += mirror_factor * tmp0[0]
            int0_storage += tmp0[0]
            CF0_storage[tidx] = tmp0[1]
            
            if idx_dict._nQ is not None:
                tmp_nQ    = integrate_nQf(rho0[tidx], tidx, maxidx, tstep, idx_dict._nQ, exp_wif, matvec) 
                intgl_nQ += mirror_factor * 2*tmp_nQ[0]
                int_nQ_storage += tmp_nQ[0]
                CF_nQ_storage[tidx] = tmp_nQ[1]
            
            if idx_dict.QnQ is not None:
                tmpQfnQf  = integrateQfnQf(rho0[tidx], tidx, maxidx, tstep, idx_dict.QnQ, exp_wif, matvec)
                intglQnQ += mirror_factor * tmpQfnQf[0]
                intQQ_storage += tmpQfnQf[0]
                CFQQ_storage[tidx] = tmpQfnQf[1]
                
            if order > 1:
                for m1 in range(nintmed):
                    """
                    k1 rate
                    """
                    if proc_id[m1+1,0] == 1: # IC-->ISC
                        # tmp_nP        = integrate_nPi(rho0[tidx], tidx, maxidx, tstep, idx_dict._nP[m1], exp_wif, matvec)
                        # intgl_nP[m1] += mirror_factor * 2*tmp_nP[0]
                        # int_nP_storage[m1] += tmp_nP[0]
                        # CF_nP_storage[m1][tidx] = tmp_nP[1]
                        intgl_nP[m1]  += mirror_factor * 2*integrate_nPi(rho0[tidx], tidx, maxidx, tstep, idx_dict._nP[m1], exp_wif, matvec)
                        intgl_nPQ[m1] += mirror_factor * 2*integrate_nPiQf(rho0[tidx], tidx, maxidx, tstep, idx_dict._nPQ[m1], exp_wif, matvec)
                        intglQnP[m1]  += mirror_factor * 2*integrateQinPi(rho0[tidx], tidx, maxidx, tstep, idx_dict.QnP[m1], exp_wif, matvec)
                        intglQnPQ[m1] += mirror_factor * 2*integrateQinPiQf(rho0[tidx], tidx, maxidx, tstep, idx_dict.QnPQ[m1], exp_wif, matvec)
                    elif proc_id[m1+1,0] == 2: # ISC-->IC
                        intgl_nP[m1]  += mirror_factor * 2*integrate_nPf(rho0[tidx], tidx, maxidx, tstep, idx_dict._nP[m1], exp_wif, matvec)
                        intgl_nPQ[m1] += mirror_factor * 2*integrate_nQiPf(rho0[tidx], tidx, maxidx, tstep, idx_dict._nPQ[m1], exp_wif, matvec)
                        intglQnP[m1]  += mirror_factor * 2*integrateQinPf(rho0[tidx], tidx, maxidx, tstep, idx_dict.QnP[m1], exp_wif, matvec)
                        intglQnPQ[m1] += mirror_factor * 2*integrateQinQiPf(rho0[tidx], tidx, maxidx, tstep, idx_dict.QnPQ[m1], exp_wif, matvec)
                    for m2 in range(nintmed):
                        """
                        k2 rate
                        """
                        if proc_id[m1+1,0] == 1 and proc_id[m2+1,0] == 1: # IC-->ISC for both m1 and m2
                            intglPnP[m1][m2]   += mirror_factor * integratePinPi(rho0[tidx], tidx, maxidx, tstep, idx_dict.PnP[m1][m2], exp_wif, matvec)
                            intglPnPQ[m1][m2]  += mirror_factor * integratePinPiQf(rho0[tidx], tidx, maxidx, tstep, idx_dict.PnPQ[m1][m2], exp_wif, matvec)
                            intglPQnP[m1][m2]  += mirror_factor * integrateQfPinPi(rho0[tidx], tidx, maxidx, tstep, idx_dict.PQnP[m1][m2], exp_wif, matvec)
                            intglPQnPQ[m1][m2] += mirror_factor * integrateQfPinPiQf(rho0[tidx], tidx, maxidx, tstep, idx_dict.PQnPQ[m1][m2], exp_wif, matvec)
                        elif proc_id[m1+1,0] == 1 and proc_id[m2+1,0] == 2: # IC-->ISC for m1 and ISC-->IC for m2
                            intglPnP[m1][m2]   += mirror_factor * integratePinPf(rho0[tidx], tidx, maxidx, tstep, idx_dict.PnP[m1][m2], exp_wif, matvec)
                            intglPnPQ[m1][m2]  += mirror_factor * integratePinQiPf(rho0[tidx], tidx, maxidx, tstep, idx_dict.PnPQ[m1][m2], exp_wif, matvec)
                            intglPQnP[m1][m2]  += mirror_factor * integrateQfPinPf(rho0[tidx], tidx, maxidx, tstep, idx_dict.PQnP[m1][m2], exp_wif, matvec)
                            intglPQnPQ[m1][m2] += mirror_factor * integrateQfPinQiPf(rho0[tidx], tidx, maxidx, tstep, idx_dict.PQnPQ[m1][m2], exp_wif, matvec)
                        elif proc_id[m1+1,0] == 2 and proc_id[m2+1,0] == 1: # ISC-->IC for m1 and IC-->ISC for m2
                            intglPnP[m1][m2]   += mirror_factor * integratePfnPi(rho0[tidx], tidx, maxidx, tstep, idx_dict.PnP[m1][m2], exp_wif, matvec)
                            intglPnPQ[m1][m2]  += mirror_factor * integratePfnPiQf(rho0[tidx], tidx, maxidx, tstep, idx_dict.PnPQ[m1][m2], exp_wif, matvec)
                            intglPQnP[m1][m2]  += mirror_factor * integratePfQinPi(rho0[tidx], tidx, maxidx, tstep, idx_dict.PQnP[m1][m2], exp_wif, matvec)
                            intglPQnPQ[m1][m2] += mirror_factor * integratePfQinPiQf(rho0[tidx], tidx, maxidx, tstep, idx_dict.PQnPQ[m1][m2], exp_wif, matvec)
                        elif proc_id[m1+1,0] == 2 and proc_id[m2+1,0] == 2: # ISC-->IC for both m1 and m2
                            intglPnP[m1][m2]   += mirror_factor * integratePfnPf(rho0[tidx], tidx, maxidx, tstep, idx_dict.PnP[m1][m2], exp_wif, matvec)
                            intglPnPQ[m1][m2]  += mirror_factor * integratePfnQiPf(rho0[tidx], tidx, maxidx, tstep, idx_dict.PnPQ[m1][m2], exp_wif, matvec)
                            intglPQnP[m1][m2]  += mirror_factor * integratePfQinPf(rho0[tidx], tidx, maxidx, tstep, idx_dict.PQnP[m1][m2], exp_wif, matvec)
                            intglPQnPQ[m1][m2] += mirror_factor * integratePfQinQiPf(rho0[tidx], tidx, maxidx, tstep, idx_dict.PQnPQ[m1][m2], exp_wif, matvec)
        elif proc_id[0,0] == 1: # If the overall process is IC
            intglPnP += integratePfnPf(rho0[tidx], tidx, maxidx, tstep, idx_dict.PnP, exp_wif, matvec)
#                if order > 1:
#                    ...

    k0_intgl_storage.append(int0_storage)
    k0_CF_storage.append(CF0_storage)
    if idx_dict._nQ is not None:
        k0_intgl_storage.append(int_nQ_storage)
        k0_CF_storage.append(CF_nQ_storage)
    if idx_dict.QnQ is not None:
        k0_intgl_storage.append(intQQ_storage)
        k0_CF_storage.append(CFQQ_storage)
    del matvec

    '''
    Once the time propagation of CFs is done, weight them by coupling terms
    '''
    out_package = []
    if proc_id[0,0] == 2: # If the overall process is ISC
        ''' Multiply the integrals by coupling terms '''
        intgl0   *= coup_dict.null
        if idx_dict._nQ is not None:
            intgl_nQ *= coup_dict._nQ
        else:
            intgl_nQ = 0.0
        if idx_dict.QnQ is not None:
            intglQnQ *= coup_dict.QnQ
        else:
            intglQnQ = 0.0

        out_package.append([intgl0, intgl_nQ, intglQnQ])

        if order > 1:
            ''' Multiply the integrals by coupling terms '''
            for m1 in range(nintmed):
                intgl_nP[m1]   *= coup_dict._nP[m1]
                intgl_nPQ[m1]  *= coup_dict._nPQ[m1]
                intglQnP[m1]   *= coup_dict.QnP[m1]
                intglQnPQ[m1]  *= coup_dict.QnPQ[m1]
                for m2 in range(nintmed):
                    intglPnP[m1][m2]   *= coup_dict.PnP[m1][m2]
                    intglPnPQ[m1][m2]  *= coup_dict.PnPQ[m1][m2]
                    intglPQnP[m1][m2]  *= coup_dict.PQnP[m1][m2]
                    intglPQnPQ[m1][m2] *= coup_dict.PQnPQ[m1][m2]
 
            out_package.append([intgl_nP, intgl_nPQ, intglQnP, intglQnPQ])
            out_package.append([intglPnP, intglPnPQ, intglPQnP, intglPQnPQ])
         
    elif proc_id[0,0] == 1: # If the overall process is IC
        ''' Multiply the integrals by a proper coupling term '''
        intglPnP *= coup_dict.PnP
        out_package.append([intglPnP])

#        if order > 1:
#            ...
    
    return out_package, k0_CF_storage, k0_intgl_storage#, int_nP_storage, CF_nP_storage


def error_estimate_trapezoidal(tf, ti, sec_idx, cf):
    if isinstance(cf[0], float):
        cf_dim = 1
    else:
        cf_dim = len(cf[0])
        
    f2 = np.zeros((int((tf-ti)/tstep[sec_idx])-2, cf_dim))
    for tidx in range(int((tf-ti)/tstep[sec_idx])-2):
        t = tidx + 1
        f2[tidx] = tstep[sec_idx]**-2 * (cf[t-1] - 2*cf[t] + cf[t+1])
    # Identify the max of absolute value of 2nd-derivative
    f2_max = np.max(abs(f2.T), axis=1)
    # error bound for trapezoidal rule
    err = ((tf-ti)*(tstep[sec_idx]**2)/12) * f2_max
    return err


def error_estimate_simpson(tf, ti, sec_idx, cf):
    if isinstance(cf[0], float):
        cf_dim = 1
    else:
        cf_dim = len(cf[0])
        
    f4 = np.zeros((int((tf-ti)/tstep[sec_idx])-4, len(cf[0])))
    for tidx in range(int((tf-ti)/tstep[sec_idx])-4):
        t = tidx + 2
        f4[tidx] = tstep[sec_idx]**-4 * (cf[t-2] -4*cf[t-1] +6*cf[t] -4*cf[t+1] +cf[t+2])
    # Identify the max of absolute value of 4th-derivative
    f4_max = np.max(abs(f4.T), axis=1)
    # error bound for Composite Simpson's 1/3 rule
    err = ((tf-ti)*(tstep[sec_idx]**4)/180) * f4_max
    return err


def error_estimate_boole(tf, ti, sec_idx, cf):
    if isinstance(cf[0], float):
        cf_dim = 1
    else:
        cf_dim = len(cf[0])
        
    # 6-th derivative
    f6 = np.zeros((int((tf-ti)/tstep[sec_idx])-6, cf_dim))
    for tidx in range(int((tf-ti)/tstep[sec_idx])-6):
        t = tidx + 3
        f6[tidx] = tstep[sec_idx]**-6 * (cf[t-3] -6*cf[t-2] + 15*cf[t-1]
                                    -20*cf[t] +15*cf[t+1] -6*cf[t+2] +cf[t+3])
     # 8-th derivative
    f8 = np.zeros((int((tf-ti)/tstep[sec_idx])-8, cf_dim))
    for tidx in range(int((tf-ti)/tstep[sec_idx])-8):
        t = tidx + 4
        f8[tidx] = tstep[sec_idx]**-8 * (cf[t-4] -8*cf[t-3] +28*cf[t-2] -56*cf[t-1]
                                    +70*cf[t] -56*cf[t+1] +28*cf[t+2] -8*cf[t+3] +cf[t+4])
    # Identify the max of absolute value of derivatives
    f6_max = np.max(abs(f6.T), axis=1)
    f8_max = np.max(abs(f8.T), axis=1)
    # error bound for Boole's rule
    err = (2*(tf-ti)*(tstep[sec_idx]**6)/945)*f6_max + ((tf-ti)*(tstep[sec_idx]**8)/450)*f8_max
    return err


def romberg_integration(order, omg_S_d, beta, proc_id, idx_dict, coup_dict):
    """
    This is the to-be-parallelized function to be iterated over time to 
    1. compute all of the relevant matrices and vectors
    2. construct all CFs, integrate them, and store them according to the
        normal mode indices.
    3. weight the integrals by a proper coupling term           
 
    order    : the perturbative order of simulation
    omg_S_d  : a collection of frequency matrices, Duschinsky matrix, and displacement vector
    beta     : inverse temperature
    proc_id  : type identifications of each of the 0th, 1st, and 2nd
                processes
    idx_dict : dictionary of classified normal modes indices necessary
                for relevant CF computation
    coup_dict: dictionary of various compound coupling terms with 
                above-threshold magnitudes 
    """
    
    '''
    Unpack the dictionary of vibrational mode indices with 
    significant couplings
    '''
    if proc_id[0,0] == 2: # If the overall process is ISC
        intglQnQ  = np.zeros(len(idx_dict.QnQ))
#        if order > 1:
#            ...
    elif proc_id[0,0] == 1: # If the overall process is IC
        intglPnP = np.zeros(len(idx_dict.PnP))
#        if order > 1:
#            ...
    
    ''' 
    Compute the integrals of correlation functions 
    
    Each of them is 2Xed for t != 0 since the real part of integral 
    is symmetric about t=0
    '''
#    intQQ_storage = np.zeros(len(idx_dict.QnQ))
#    QQ_storage = np.zeros((int(tfinal/tstep), len(idx_dict.QnQ)))
    mirror_factor = 2
        
    # 0th order rate
    if proc_id[0,0] == 2: # If the overall process is ISC
        intgl, er = rule(integrand_QQ, tinit, tfinal, beta, omg_S_d, idx_dict.QnQ, max_nrow, tol)
        intglQnQ += mirror_factor * intgl
        
#     elif proc_id[0,0] == 1: # If the overall process is IC
#         intglPnP += integratePfnPf(rho0[tidx], tidx, idx_dict.PnP, exp_wif, matvec)
#            if order > 1:
#                ...

    '''
    Once the time propagation of CFs is done, weight them by coupling terms
    '''
    out_package = []
    if proc_id[0,0] == 2: # If the overall process is ISC
        ''' Multiply the integrals by coupling terms '''
        intglQnQ *= coup_dict.QnQ

        out_package.append([intglQnQ])
         
    elif proc_id[0,0] == 1: # If the overall process is IC
        ''' Multiply the integrals by a proper coupling term '''
        intglPnP *= coup_dict.PnP
        out_package.append([intglPnP])
#        if order > 1:
#            ...
    
    return out_package, er
    

def tanh_sinh_integration(order, omg_S_d, beta, proc_id, idx_dict, coup_dict):
    """
    This is the to-be-parallelized function to be iterated over time to 
    1. compute all of the relevant matrices and vectors
    2. construct all CFs, integrate them, and store them according to the
        normal mode indices.
    3. weight the integrals by a proper coupling term           
 
    order    : the perturbative order of simulation
    omg_S_d  : a collection of frequency matrices, Duschinsky matrix, and displacement vector
    beta     : inverse temperature
    proc_id  : type identifications of each of the 0th, 1st, and 2nd
                processes
    idx_dict : dictionary of classified normal modes indices necessary
                for relevant CF computation
    coup_dict: dictionary of various compound coupling terms with 
                above-threshold magnitudes 
    """
    
    '''
    Unpack the dictionary of vibrational mode indices with 
    significant couplings
    '''
    if proc_id[0,0] == 2: # If the overall process is ISC
        intglQnQ  = np.zeros(len(idx_dict.QnQ))
#        if order > 1:
#            ...
    elif proc_id[0,0] == 1: # If the overall process is IC
        intglPnP = np.zeros(len(idx_dict.PnP))
#        if order > 1:
#            ...
    
    ''' 
    Compute the integrals of correlation functions 
    
    Each of them is 2Xed for t != 0 since the real part of integral 
    is symmetric about t=0
    '''
#    intQQ_storage = np.zeros(len(idx_dict.QnQ))
#    QQ_storage = np.zeros((int(tfinal/tstep), len(idx_dict.QnQ)))
    mirror_factor = 2
        
    # 0th order rate
    if proc_id[0,0] == 2: # If the overall process is ISC
        intglQnQ += mirror_factor * rule(integrand_QQ, tfinal, beta, omg_S_d, idx_dict.QnQ)
        
#     elif proc_id[0,0] == 1: # If the overall process is IC
#         intglPnP += integratePfnPf(rho0[tidx], tidx, idx_dict.PnP, exp_wif, matvec)
#            if order > 1:
#                ...

    '''
    Once the time propagation of CFs is done, weight them by coupling terms
    '''
    out_package = []
    if proc_id[0,0] == 2: # If the overall process is ISC
        ''' Multiply the integrals by coupling terms '''
        intglQnQ *= coup_dict.QnQ

        out_package.append([intglQnQ])
         
    elif proc_id[0,0] == 1: # If the overall process is IC
        ''' Multiply the integrals by a proper coupling term '''
        intglPnP *= coup_dict.PnP
        out_package.append([intglPnP])
#        if order > 1:
#            ...
    
    return out_package
    


