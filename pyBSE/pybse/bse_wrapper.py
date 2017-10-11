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
           neta = 1.0,
           bwind = 0.0,
           hewind = 1.0,
           alpha1 = 1.0,
           lambda_ce = 0.5,
           ceflag = 0,
           tflag = 1,
           ifflag = 0,
           wdflag = 1,
           GRflag = 0,
           bhflag = 0,
           nsflag = 1,
           mxns = 3.0,
           pts1 = 0.05,
           pts2 = 0.01,
           pts3 = 0.01,
           sigma = 190.0,
           beta = 0.125,
           xi = 1.0,
           acc2 = 1.5,
           epsnov = 0.001,
           eddfac = 10.0,
           gamma = -1.0):

    """ A wrapper for BSE

    GRflag : int
        Flag to turn on gravitational wave radiation. Default = 0 (off)


    """

    if idum is None:
        random_data = os.urandom(4)
        idum = int.from_bytes(random_data, byteorder="big")


    M1_out, M2_out, a_out, ecc_out, v_sys_out, mdot_out, \
    t_SN1, k1_out, k2_out = \
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

    return M1_out, M2_out, a_out, ecc_out, v_sys_out, mdot_out, \
                t_SN1, k1_out, k2_out
