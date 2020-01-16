import numpy as np
import os

from cosmic.sample.initialbinarytable import InitialBinaryTable
from cosmic.sample.sampler import independent
from cosmic.evolve import Evolve

from cosmic import _evolvebin



BPP_COLUMNS = ['tphys', 'mass_1', 'mass_2', 'kstar_1', 'kstar_2' ,
               'sep', 'porb', 'ecc', 'RROL_1', 'RROL_2', 'evol_type',
               'Vsys_1', 'Vsys_2', 'SNkick', 'SNtheta',
               'aj_1', 'aj_2', 'tms_1', 'tms_2',
               'massc_1', 'massc_2', 'rad_1', 'rad_2',
               'bin_num']

bpp_dtype = [('tphys','f8'), ('mass_1','f8'), ('mass_2','f8'), ('kstar_1','f8'), ('kstar_2','f8'),
             ('sep','f8'), ('porb','f8'), ('ecc','f8'), ('RROL_1','f8'), ('RROL_2','f8'), ('evol_type','f8'),
             ('Vsys_1','f8'), ('Vsys_2','f8'), ('SNkick','f8'), ('SNtheta','f8'),
             ('aj_1','f8'), ('aj_2','f8'), ('tms_1','f8'), ('tms_2','f8'),
             ('massc_1','f8'), ('massc_2','f8'), ('rad_1','f8'), ('rad_2','f8')]



bcm_dtype = [('tphys','f8'), ('kstar_1','f8'), ('mass0_1','f8'), ('mass_1','f8'), ('lumin_1','f8'), ('rad_1','f8'),
             ('teff_1','f8'), ('massc_1','f8'), ('radc_1','f8'), ('menv_1','f8'), ('renv_1','f8'), ('epoch_1','f8'),
             ('ospin_1','f8'), ('deltam_1','f8'), ('RROL_1','f8'), ('kstar_2','f8'), ('mass0_2','f8'), ('mass_2','f8'),
             ('lumin_2','f8'), ('rad_2','f8'), ('teff_2','f8'), ('massc_2','f8'), ('radc_2','f8'), ('menv_2','f8'),
             ('renv_2','f8'), ('epoch_2','f8'), ('ospin_2','f8'), ('deltam_2','f8'), ('RROL_2','f8'),
             ('porb','f8'), ('sep','f8'), ('ecc','f8'), ('B_0_1','f8'), ('B_0_2','f8'),
             ('SNkick_1','f8'), ('SNkick_2','f8'), ('Vsys_final','f8'), ('SNtheta_final','f8'),
             ('SN_1','f8'), ('SN_2','f8'), ('bin_state','f8'), ('merger_type','f8')]



    # BSEDict = {
    #
    #            'natal_kick_array' : [v_kick_1,v_kick_2,np.pi/2-theta_kick_1,np.pi/2-theta_kick_2,phi_kick_1,phi_kick_2],
    #            'qcrit_array' : [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],
    #            'fprimc_array' : [2.0/21.0,2.0/21.0,2.0/21.0,2.0/21.0,2.0/21.0,2.0/21.0,2.0/21.0,2.0/21.0,2.0/21.0,2.0/21.0,2.0/21.0,2.0/21.0,2.0/21.0,2.0/21.0,2.0/21.0,2.0/21.0],





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
           nsflag = 3,
           mxns = 2.5,
           rejuvflag = 0.0,
           rejuv_fac = 1.0,
           pts1 = 0.001,
           pts2 = 0.01,
           pts3 = 0.02,
           sigma = 265.0,
           polar_kick_angle = 90.0,
           bhsigmafrac = 1.0,
           pisn = 45.0,
           ecsn = 2.5,
           ecsn_mlow = 1.4,
           aic = 1,
           ussn = 0,
           eddlimflag = 0,
           beta = 0.125,
           xi = 1.0,
           acc2 = 1.5,
           epsnov = 0.001,
           eddfac = 1.0,
           gamma = -1.0,
           qcflag = 2,
           ck = -1000,
           bconst = -3000,
           htpmb = 1,
           ST_cr = 1,
           ST_tide = 0,
           bdecayfac = 1):


    if idum is None:
        random_data = os.urandom(4)
        idum = int.from_bytes(random_data, byteorder="big")



    natal_kick_array = [v_kick_1,v_kick_2,np.pi/2-theta_kick_1,np.pi/2-theta_kick_2,phi_kick_1,phi_kick_2]
    qcrit_array = [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
    fprimc_array = [2.0/21.0,2.0/21.0,2.0/21.0,2.0/21.0,2.0/21.0,2.0/21.0,2.0/21.0,2.0/21.0,2.0/21.0,2.0/21.0,2.0/21.0,2.0/21.0,2.0/21.0,2.0/21.0,2.0/21.0,2.0/21.0]

# initial_conditions = initialbinarytable[INITIAL_CONDITIONS_PASS_COLUMNS].values

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
    _evolvebin.flags.nsflag = nsflag
    _evolvebin.ceflags.cekickflag = cekickflag
    _evolvebin.ceflags.cemergeflag = cemergeflag
    _evolvebin.ceflags.cehestarflag = cehestarflag
    _evolvebin.windvars.mxns = mxns
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
    _evolvebin.magvars.bconst = bconst
    _evolvebin.magvars.ck = ck
    _evolvebin.flags.windflag = windflag
    _evolvebin.flags.qcflag = qcflag
    _evolvebin.windvars.eddlimflag = eddlimflag
    _evolvebin.tidalvars.fprimc_array = fprimc_array
    _evolvebin.rand1.idum1 = idum
    _evolvebin.flags.bhspinflag = bhspinflag
    _evolvebin.snvars.bhspinmag = bhspinmag
    _evolvebin.mixvars.rejuv_fac = rejuv_fac
    _evolvebin.flags.rejuvflag = rejuvflag
    _evolvebin.flags.htpmb = htpmb
    _evolvebin.flags.st_cr = ST_cr
    _evolvebin.flags.st_tide = ST_tide
    _evolvebin.cmcpass.using_cmc = 0


    kstar_1 = 1
    kstar_2 = 1

    [tmp_bpp, tmp_bcm] = _evolvebin.evolv2([kstar_1, kstar_2],
                                           [M1, M2],
                                           P_orb,
                                           ecc,
                                           metallicity,
                                           time,
                                           dtp,
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
                                           np.zeros(20))





    bcm = tmp_bcm.ravel().view(dtype=bcm_dtype)
    bpp = tmp_bpp.ravel().view(dtype=bpp_dtype)



    idx = np.argmax(bcm['tphys'])
    bcm = bcm[:idx]
    idx = np.argmax(bpp['tphys'])
    bpp = bpp[:idx]


    # print(tmp[0])
    # print(tmp[10]['tphys'])
    #
    # bpp = np.array(bpp, dtype=bpp_dtype)
    # bcm = np.array(bcm, dtype=bcm_dtype)

    # print(bcm.shape)
    # print(bcm[0].shape)
    # print(bcm[0][0])
    # print(bcm[0]['tphys'])
    # print(bcm[0])


    dtype = [('M1', 'f8'), ('M2', 'f8'), ('a', 'f8'), ('ecc', 'f8'), ('v_sys', 'f8'),
             ('mdot1', 'f8'), ('mdot2', 'f8'), ('t_SN1', 'f8'), ('t_SN2', 'f8'),
             ('R1', 'f8'), ('R2', 'f8'), ('Teff1', 'f8'), ('Teff2', 'f8'),
             ('L1', 'f8'), ('L2', 'f8'), ('k1','i8'), ('k2','i8')]

    output = np.zeros(1, dtype=dtype)

    if len(bcm) == 0: return output[0]


    output[0]['M1'] = bcm[-1]['mass_1']
    output[0]['M2'] = bcm[-1]['mass_2']
    output[0]['a'] = bcm[-1]['sep']
    output[0]['ecc'] = bcm[-1]['ecc']
    output[0]['v_sys'] = bcm[-1]['Vsys_final']
    output[0]['mdot1'] = bcm[-1]['deltam_1']
    output[0]['mdot2'] = bcm[-1]['deltam_2']
    try:
        output[0]['t_SN1'] = bpp[bpp['evol_type'] == 15][0]['tphys']
    except:
        output[0]['t_SN1'] = 0.0
    try:
        output[0]['t_SN2'] = bpp[bpp['evol_type'] == 15][0]['tphys']
    except:
        output[0]['t_SN2'] = 0.0
    output[0]['R1'] = bcm[-1]['rad_1']
    output[0]['R2'] = bcm[-1]['rad_2']
    output[0]['Teff1'] = bcm[-1]['teff_1']
    output[0]['Teff2'] = bcm[-1]['teff_2']
    output[0]['L1'] = bcm[-1]['lumin_1']
    output[0]['L2'] = bcm[-1]['lumin_2']
    output[0]['k1'] = bcm[-1]['kstar_1']
    output[0]['k2'] = bcm[-1]['kstar_2']


#
# def evolve(M1,
#            M2,
#            P_orb,
#            ecc,
#            v_kick_1,
#            theta_kick_1,
#            phi_kick_1,
#            v_kick_2,
#            theta_kick_2,
#            phi_kick_2,
#            time,
#            metallicity,
#            verbose_output,
#            idum = None,
#            neta = 0.5,
#            bwind = 0.0,
#            hewind = 0.5,
#            alpha1 = 1.0,
#            lambda_ce = 0.5,
#            ceflag = 0,
#            cekickflag = 2,
#            sigmadiv = 10,
#            tflag = 1,
#            ifflag = 0,
#            wdflag = 1,
#            GRflag = 0,
#            bhflag = 0,
#            nsflag = 3,
#            mxns = 2.5,
#            pts1 = 0.05,
#            pts2 = 0.01,
#            pts3 = 0.01,
#            sigma = 190.0,
#            beta = 0.125,
#            xi = 1.0,
#            acc2 = 1.5,
#            epsnov = 0.001,
#            eddfac = 1.0,
#            gamma = -1.0,
#            qcflag = 2):
#
#     """ A wrapper for BSE
#
#     GRflag : int
#         Flag to turn on gravitational wave radiation. Default = 0 (off)
#
#
#     """
#
#     if idum is None:
#         random_data = os.urandom(4)
#         idum = int.from_bytes(random_data, byteorder="big")
#
#
#     single_binary = InitialBinaryTable.InitialBinaries(m1=M1,
#                                                        m2=M2,
#                                                        porb=P_orb,
#                                                        ecc=ecc,
#                                                        tphysf=time,
#                                                        kstar1=1,
#                                                        kstar2=1,
#                                                        metallicity=metallicity)
#
#
#
#     # Note: dart_board sets the polar angle to between 0 and pi, whereas cosmic assumes it ranges between -pi/2 to pi/2.
#
#
#
#     BSEDict = {'xi': xi, 'bhflag': bhflag, 'neta': neta, 'windflag': 3, 'wdflag': wdflag, 'alpha1': alpha1,
#                'pts1': pts1, 'pts3': pts3, 'pts2': pts2, 'epsnov': epsnov, 'hewind': hewind,
#                'ck': -1000, 'bwind': bwind, 'lambdaf': lambda_ce, 'mxns': mxns, 'beta': beta, 'tflag': tflag,
#                'acc2': acc2, 'nsflag': nsflag, 'ceflag': ceflag, 'eddfac': eddfac, 'ifflag': ifflag, 'bconst': -3000,
#                'sigma': sigma, 'gamma': gamma, 'pisn': 45.0,
#                'natal_kick_array' : [v_kick_1,v_kick_2,np.pi/2-theta_kick_1,np.pi/2-theta_kick_2,phi_kick_1,phi_kick_2], 'bhsigmafrac' : 1.0,
#                'polar_kick_angle' : 90,
#                'qcflag' : qcflag, 'qcrit_array' : [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],
#                'cekickflag' : cekickflag, 'cehestarflag' : 0, 'cemergeflag' : 0, 'ecsn' : 2.5, 'ecsn_mlow' : 1.4,
#                'aic' : 1, 'ussn' : 0, 'sigmadiv' :10.0, 'eddlimflag' : 0,
#                'fprimc_array' : [2.0/21.0,2.0/21.0,2.0/21.0,2.0/21.0,2.0/21.0,2.0/21.0,2.0/21.0,2.0/21.0,2.0/21.0,2.0/21.0,2.0/21.0,2.0/21.0,2.0/21.0,2.0/21.0,2.0/21.0,2.0/21.0],
#                'bhspinflag' : 0, 'bhspinmag' : 0.0, 'rejuv_fac' : 1.0, 'rejuvflag' : 0, 'htpmb' : 1, 'ST_cr' : 1,
#                'ST_tide' : 0, 'bdecayfac' : 1}
#
#
#     bpp, bcm, initC = Evolve.evolve(initialbinarytable=single_binary, BSEDict=BSEDict)
#
#
#
#
#
#     dtype = [('M1', 'f8'), ('M2', 'f8'), ('a', 'f8'), ('ecc', 'f8'), ('v_sys', 'f8'),
#              ('mdot1', 'f8'), ('mdot2', 'f8'), ('t_SN1', 'f8'), ('t_SN2', 'f8'),
#              ('R1', 'f8'), ('R2', 'f8'), ('Teff1', 'f8'), ('Teff2', 'f8'),
#              ('L1', 'f8'), ('L2', 'f8'), ('k1','i8'), ('k2','i8')]
#
#     output = np.zeros(1, dtype=dtype)
#     output[0]['M1'] = bcm.iloc[-1]['mass_1']
#     output[0]['M2'] = bcm.iloc[-1]['mass_2']
#     output[0]['a'] = bcm.iloc[-1]['sep']
#     output[0]['ecc'] = bcm.iloc[-1]['ecc']
#     output[0]['v_sys'] = bcm.iloc[-1]['Vsys_final']
#     output[0]['mdot1'] = bcm.iloc[-1]['deltam_1']
#     output[0]['mdot2'] = bcm.iloc[-1]['deltam_2']
#     try:
#         output[0]['t_SN1'] = bpp.loc[bpp['evol_type'] == 15].iloc[0]['tphys']
#     except:
#         output[0]['t_SN1'] = 0.0
#     try:
#         output[0]['t_SN2'] = bpp.loc[bpp['evol_type'] == 15].iloc[0]['tphys']
#     except:
#         output[0]['t_SN2'] = 0.0
#     output[0]['R1'] = bcm.iloc[-1]['rad_1']
#     output[0]['R2'] = bcm.iloc[-1]['rad_2']
#     output[0]['Teff1'] = bcm.iloc[-1]['teff_1']
#     output[0]['Teff2'] = bcm.iloc[-1]['teff_2']
#     output[0]['L1'] = bcm.iloc[-1]['lumin_1']
#     output[0]['L2'] = bcm.iloc[-1]['lumin_2']
#     output[0]['k1'] = bcm.iloc[-1]['kstar_1']
#     output[0]['k2'] = bcm.iloc[-1]['kstar_2']


    return output[0]
