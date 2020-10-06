import numpy as np
import os

from cosmic.sample.initialbinarytable import InitialBinaryTable
from cosmic.sample.sampler import independent
from cosmic.evolve import Evolve

from cosmic import _evolvebin

BPP_COLUMNS = ['tphys', 'mass_1', 'mass_2', 'kstar_1', 'kstar_2', 'sep', 'porb', 'ecc',
               'RRLO_1', 'RRLO_2', 'evol_type', 'aj_1', 'aj_2', 'tms_1', 'tms_2',
               'massc_1', 'massc_2', 'rad_1', 'rad_2', 'mass0_1', 'mass0_2', 'lum_1',
               'lum_2', 'teff_1', 'teff_2', 'radc_1', 'radc_2', 'menv_1', 'menv_2',
               'renv_1', 'renv_2', 'omega_spin_1', 'omega_spin_2', 'B_1', 'B_2',
               'bacc_1', 'bacc_2', 'tacc_1', 'tacc_2', 'epoch_1', 'epoch_2',
               'bhspin_1', 'bhspin_2']

bpp_dtype = [('tphys','f8'),
             ('mass_1','f8'), ('mass_2','f8'),
             ('kstar_1','f8'), ('kstar_2','f8'),
             ('sep','f8'), ('porb','f8'), ('ecc','f8'),
             ('RROL_1','f8'), ('RROL_2','f8'),
             ('evol_type','f8'),
             ('aj_1','f8'), ('aj_2','f8'),
             ('tms_1','f8'), ('tms_2','f8'),
             ('massc_1','f8'), ('massc_2','f8'),
             ('rad_1','f8'), ('rad_2','f8'),
             ('mass0_1','f8'), ('mass0_2','f8'),
             ('lum_1','f8'), ('lum_2','f8'),
             ('teff_1','f8'), ('teff_2','f8'),
             ('radc_1','f8'), ('radc_2','f8'),
             ('menv_1','f8'), ('menv_2','f8'),
             ('renv_1','f8'), ('renv_2','f8'),
             ('omega_spin_1','f8'), ('omega_spin_2','f8'),
             ('B_1','f8'), ('B_2','f8'),
             ('bacc_1','f8'), ('bacc_2','f8'),
             ('tacc_1','f8'), ('tacc_2','f8'),
             ('epoch_1','f8'), ('epoch_2','f8'),
             ('bhspin_1','f8'), ('bhspin_2','f8')]



bcm_dtype = [('tphys','f8'),
             ('kstar_1','f8'), ('mass0_1','f8'), ('mass_1','f8'), ('lum_1','f8'), ('rad_1','f8'),
             ('teff_1','f8'), ('massc_1','f8'), ('radc_1','f8'), ('menv_1','f8'), ('renv_1','f8'),
             ('epoch_1','f8'),('omega_spin_1','f8'), ('deltam_1','f8'), ('RROL_1','f8'),
             ('kstar_2','f8'), ('mass0_2','f8'), ('mass_2','f8'),('lum_2','f8'), ('rad_2','f8'),
             ('teff_2','f8'), ('massc_2','f8'), ('radc_2','f8'), ('menv_2','f8'),
             ('renv_2','f8'), ('epoch_2','f8'), ('omega_spin_2','f8'), ('deltam_2','f8'), ('RROL_2','f8'),
             ('porb','f8'), ('sep','f8'), ('ecc','f8'), ('B_1','f8'), ('B_2','f8'),
             ('SN_1','f8'), ('SN_2','f8'), ('bin_state','f8'), ('merger_type','f8')]

KICK_COLUMNS = ['star', 'disrupted', 'natal_kick', 'phi', 'theta', 'eccentric_anomaly',
                'delta_vsysx_1', 'delta_vsysy_1', 'delta_vsysz_1', 'vsys_1_total',
                'delta_vsysx_2', 'delta_vsysy_2', 'delta_vsysz_2', 'vsys_2_total',
                'delta_theta_total', 'omega', 'randomseed', 'bin_num']

kick_dtype = [('star','f8'), ('disrupted','f8'), ('natal_kick','f8'),
              ('phi','f8'), ('theta','f8'), ('eccentric_anomaly','f8'),
              ('delta_vsysx_1','f8'), ('delta_vsysy_1','f8'), ('delta_vsysz_1','f8'),
              ('vsys_1_total','f8'),
              ('delta_vsysx_2','f8'), ('delta_vsysy_2','f8'), ('delta_vsysz_2','f8'),
              ('vsys_2_total','f8'),
              ('delta_theta_total','f8'), ('omega','f8'), ('randomseed','f8')]


def evolve(M1,
           M2,
           P_orb,
           ecc,
           v_kick_1,
           theta_kick_1,
           phi_kick_1,
           omega_kick_1,
           v_kick_2,
           theta_kick_2,
           phi_kick_2,
           omega_kick_2,
           time,
           metallicity,
           return_evolution = False,
           verbose_output = False,
           idum = None,
           dtp = 0.1,
           neta = 0.5,
           bwind = 0.0,
           hewind = 0.5,
           alpha1 = 1.0,
           lambda_ce = 0.5,
           ceflag = 0,
           cekickflag = 2,
           cemergeflag = 0,
           cehestarflag = 0,
           sigmadiv = -20,
           tflag = 1,
           windflag = 3,
           ifflag = 0,
           wdflag = 1,
           GRflag = 0,
           bhflag = 1,
           bhspinflag = 0,
           bhspinmag = 0.0,
           remnantflag = 3,
           mxns = 2.5,
           rejuvflag = 0,
           rejuv_fac = 1.0,
           pts1 = 0.005,
           pts2 = 0.01,
           pts3 = 0.02,
           sigma = 265.0,
           polar_kick_angle = 90.0,
           qcrit_array = [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],
           bhsigmafrac = 1.0,
           pisn = 45.0,
           ecsn = 2.5,
           ecsn_mlow = 1.4,
           aic = 1,
           ussn = 0,
           eddlimflag = 0,
           fprimc_array = [2.0/21.0,2.0/21.0,2.0/21.0,2.0/21.0,2.0/21.0,2.0/21.0,2.0/21.0,2.0/21.0,2.0/21.0,2.0/21.0,2.0/21.0,2.0/21.0,2.0/21.0,2.0/21.0,2.0/21.0,2.0/21.0],
           beta = 0.125,
           xi = 1.0,
           acc2 = 1.5,
           epsnov = 0.001,
           eddfac = 1.0,
           gamma = -2.0,
           qcflag = 4,
           ck = -1000,
           bconst = -3000,
           htpmb = 1,
           ST_cr = 1,
           ST_tide = 0,
           bdecayfac = 1,
           rembar_massloss = 0.5,
           kickflag = 0,
           f_acc = 1,
           don_lim = 0,
           acc_lim = -2,
           zsun = 0.014):

    if idum is None:
        random_data = os.urandom(4)
        idum = int.from_bytes(random_data, byteorder="big")


    # This is a stand in - the eccentric anomaly needs to be passed instead of pi/3 below
    # Unfortunately, this needs to be updated within cosmic to be the mean anomaly.
    natal_kick_array = [[v_kick_1, (np.pi/2-theta_kick_1) * 180/np.pi, phi_kick_1 * 180/np.pi, omega_kick_1, 0],
                        [v_kick_2, (np.pi/2-theta_kick_2) * 180/np.pi, phi_kick_2 * 180/np.pi, omega_kick_2, 0]]
    # natal_kick_array = [[-100.0, -100.0, -100.0, -100.0, 0.0], [-100.0, -100.0, -100.0, -100.0, 0.0]]
    kick_info = np.zeros((2,len(KICK_COLUMNS)-1))
    # kick_info[0,2] = v_kick_1
    # kick_info[0,3] = np.pi/2-theta_kick_1
    # kick_info[0,4] = phi_kick_1
    # kick_info[1,2] = v_kick_2
    # kick_info[1,3] = np.pi/2-theta_kick_2
    # kick_info[1,4] = phi_kick_2

    _evolvebin.windvars.neta = neta
    _evolvebin.windvars.bwind = bwind
    _evolvebin.windvars.hewind = hewind
    _evolvebin.cevars.alpha1 = alpha1
    _evolvebin.cevars.lambdaf = lambda_ce
    _evolvebin.ceflags.ceflag = ceflag
    _evolvebin.flags.tflag = tflag
    _evolvebin.flags.ifflag = ifflag
    _evolvebin.flags.wdflag = wdflag
    _evolvebin.snvars.pisn = pisn
    _evolvebin.flags.bhflag = bhflag
    # _evolvebin.flags.nsflag = nsflag
    _evolvebin.flags.remnantflag = remnantflag
    _evolvebin.ceflags.cekickflag = cekickflag
    _evolvebin.ceflags.cemergeflag = cemergeflag
    _evolvebin.ceflags.cehestarflag = cehestarflag
    _evolvebin.snvars.mxns = mxns
    _evolvebin.points.pts1 = pts1
    _evolvebin.points.pts2 = pts2
    _evolvebin.points.pts3 = pts3
    _evolvebin.snvars.ecsn = ecsn
    _evolvebin.snvars.ecsn_mlow = ecsn_mlow
    _evolvebin.flags.aic = aic
    _evolvebin.ceflags.ussn = ussn
    _evolvebin.snvars.sigma = sigma
    _evolvebin.snvars.sigmadiv = sigmadiv
    _evolvebin.snvars.bhsigmafrac = bhsigmafrac
    _evolvebin.snvars.polar_kick_angle = polar_kick_angle
    _evolvebin.snvars.natal_kick_array = natal_kick_array
    _evolvebin.cevars.qcrit_array = qcrit_array
    _evolvebin.windvars.beta = beta
    _evolvebin.windvars.xi = xi
    _evolvebin.windvars.acc2 = acc2
    _evolvebin.windvars.epsnov = epsnov
    _evolvebin.windvars.eddfac = eddfac
    _evolvebin.windvars.gamma = gamma
    _evolvebin.flags.bdecayfac = bdecayfac
    _evolvebin.flags.grflag = GRflag
    _evolvebin.magvars.bconst = bconst
    _evolvebin.magvars.ck = ck
    _evolvebin.flags.windflag = windflag
    _evolvebin.flags.qcflag = qcflag
    _evolvebin.flags.eddlimflag = eddlimflag
    _evolvebin.tidalvars.fprimc_array = fprimc_array
    _evolvebin.rand1.idum1 = idum
    _evolvebin.flags.bhspinflag = bhspinflag
    _evolvebin.snvars.bhspinmag = bhspinmag
    _evolvebin.mixvars.rejuv_fac = rejuv_fac
    _evolvebin.flags.rejuvflag = rejuvflag
    _evolvebin.flags.htpmb = htpmb
    _evolvebin.flags.st_cr = ST_cr
    _evolvebin.flags.st_tide = ST_tide
    _evolvebin.snvars.rembar_massloss = rembar_massloss
    _evolvebin.metvars.zsun = zsun
    _evolvebin.snvars.kickflag = kickflag
    _evolvebin.cmcpass.using_cmc = 0
    _evolvebin.windvars.f_acc = f_acc
    _evolvebin.windvars.don_lim = don_lim
    _evolvebin.windvars.acc_lim = acc_lim

    kstar_1 = 1
    kstar_2 = 1

    [bpp_index, bcm_index, kick_info_out] \
                       = _evolvebin.evolv2([kstar_1, kstar_2],
                                           [M1, M2],
                                           P_orb,
                                           ecc,
                                           metallicity,
                                           time,
                                           time,
                                           # dtp,
                                           [M1, M2],
                                           [0.0, 0.0],
                                           [0.0, 0.0],
                                           [0.0, 0.0],
                                           [0.0, 0.0],
                                           [0.0, 0.0],
                                           [0.0, 0.0],
                                           [0.0, 0.0],
                                           [0.0, 0.0],
                                           [0.0, 0.0],
                                           [0.0, 0.0],
                                           [0.0, 0.0],
                                           [0.0, 0.0],
                                           [0.0, 0.0],
                                           0.0,
                                           np.zeros(20),
                                           kick_info)

    bcm = _evolvebin.binary.bcm[:bcm_index].copy()
    bpp = _evolvebin.binary.bpp[:bpp_index].copy()
    _evolvebin.binary.bpp[:bpp_index] = np.zeros(bpp.shape)
    _evolvebin.binary.bcm[:bcm_index] = np.zeros(bcm.shape)
    bcm = bcm.view(dtype=bcm_dtype)
    bpp = bpp.view(dtype=bpp_dtype)

    # If we want the entire evolution
    if return_evolution:
        return bcm, bpp

    kick = kick_info_out.ravel().view(dtype=kick_dtype)

    dtype = [('M1', 'f8'), ('M2', 'f8'), ('a', 'f8'), ('ecc', 'f8'), ('v_sys', 'f8'),
             ('mdot1', 'f8'), ('mdot2', 'f8'), ('t_SN1', 'f8'), ('t_SN2', 'f8'),
             ('R1', 'f8'), ('R2', 'f8'), ('Teff1', 'f8'), ('Teff2', 'f8'),
             ('L1', 'f8'), ('L2', 'f8'), ('k1','i8'), ('k2','i8')]

    output = np.zeros(1, dtype=dtype)

    if len(bcm) == 0: return output[0]

    V_sys_1 = np.sqrt(kick[0]['delta_vsysx_1']**2 + kick[0]['delta_vsysy_1']**2 + kick[0]['delta_vsysz_1']**2)
    V_sys_2 = np.sqrt(kick[1]['delta_vsysx_2']**2 + kick[1]['delta_vsysy_2']**2 + kick[1]['delta_vsysz_2']**2)


    output[0]['M1'] = bcm[-1]['mass_1']
    output[0]['M2'] = bcm[-1]['mass_2']
    output[0]['a'] = bcm[-1]['sep']
    output[0]['ecc'] = bcm[-1]['ecc']
    output[0]['v_sys'] = np.sqrt(V_sys_1**2 + V_sys_2**2)
    output[0]['mdot1'] = bcm[-1]['deltam_1']
    output[0]['mdot2'] = bcm[-1]['deltam_2']
    try:
        output[0]['t_SN1'] = bpp[bpp['evol_type'] == 15][0]['tphys']
    except:
        output[0]['t_SN1'] = 0.0
    try:
        output[0]['t_SN2'] = bpp[bpp['evol_type'] == 16][0]['tphys']
    except:
        output[0]['t_SN2'] = 0.0
    output[0]['R1'] = bcm[-1]['rad_1']
    output[0]['R2'] = bcm[-1]['rad_2']
    output[0]['Teff1'] = bcm[-1]['teff_1']
    output[0]['Teff2'] = bcm[-1]['teff_2']
    output[0]['L1'] = bcm[-1]['lum_1']
    output[0]['L2'] = bcm[-1]['lum_2']
    output[0]['k1'] = bcm[-1]['kstar_1']
    output[0]['k2'] = bcm[-1]['kstar_2']

    return output[0]
