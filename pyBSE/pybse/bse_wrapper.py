import numpy as np
import sys
sys.path.append("../")
import bse
import os

def evolve(M1,
           M2,
           P_orb,
           ecc,
           v_kick_1,
           theta_kick_1,
           phi_kick_1,
           v_kick_2,
           theta_kick_2,
           phi_kick_2,
           time,
           metallicity,
           verbose_output,
           idum = None,
           neta = 0.5,
           bwind = 0.0,
           hewind = 0.5,
           alpha1 = 1.0,
           lambda_ce = 0.5,
           ceflag = 0,
           tflag = 1,
           ifflag = 0,
           wdflag = 1,
           GRflag = 0,
           bhflag = 0,
           nsflag = 3,
           mxns = 2.5,
           pts1 = 0.05,
           pts2 = 0.01,
           pts3 = 0.01,
           sigma = 190.0,
           beta = 0.125,
           xi = 1.0,
           acc2 = 1.5,
           epsnov = 0.001,
           eddfac = 1.0,
           gamma = -1.0):

    """ A wrapper for BSE

    GRflag : int
        Flag to turn on gravitational wave radiation. Default = 0 (off)


    """

    if idum is None:
        random_data = os.urandom(4)
        idum = int.from_bytes(random_data, byteorder="big")



    M1_out, M2_out, a_out, ecc_out, v_sys_out, mdot1_out, \
    mdot2_out, t_SN1, t_SN2, r1_out, r2_out, teff1_out, \
    teff2_out, lum1_out, lum2_out, k1_out, k2_out = \
    bse.evolv_wrapper(1, M1, M2, P_orb, ecc, v_kick_1,
                      theta_kick_1, phi_kick_1, v_kick_2,
                      theta_kick_2, phi_kick_2, time, metallicity,
                      verbose_output, idum,
                      neta, bwind, hewind,
                      alpha1, lambda_ce, ceflag,
                      tflag, ifflag, wdflag,
                      GRflag,
                      bhflag, nsflag, mxns,
                      pts1, pts2, pts3, sigma,
                      beta, xi, acc2, epsnov,
                      eddfac, gamma)

    print(M1_out, M2_out, a_out, ecc_out, v_sys_out, mdot1_out, \
          mdot2_out, t_SN1, t_SN2, r1_out, r2_out, teff1_out, \
          teff2_out, lum1_out, lum2_out, k1_out, k2_out)


    dtype = [('M1', 'f8'), ('M2', 'f8'), ('a', 'f8'), ('ecc', 'f8'), ('v_sys', 'f8'),
             ('mdot1', 'f8'), ('mdot2', 'f8'), ('t_SN1', 'f8'), ('t_SN2', 'f8'),
             ('R1', 'f8'), ('R2', 'f8'), ('Teff1', 'f8'), ('Teff2', 'f8'),
             ('L1', 'f8'), ('L2', 'f8'), ('k1','i8'), ('k2','i8')]

    output = np.array([M1_out, M2_out, a_out, ecc_out, v_sys_out,
                       mdot1_out, mdot2_out, t_SN1, t_SN2, r1_out,
                       r2_out, teff1_out, teff2_out, lum1_out,
                       lum2_out, int(k1_out), int(k2_out)], dtype=dtype)

    return output
