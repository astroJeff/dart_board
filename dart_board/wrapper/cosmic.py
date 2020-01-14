import numpy as np
from cosmic import _evolvebin


def evolve(mass1, mass2, orbital_period, ecc,
           v_kick1, theta_kick1, phi_kick1,
           v_kick2, theta_kick2, phi_kick2,
           tphysf, metallicity,
           verbose=False,
           xi=1.0, bhflag=1, neta=0.5, windflag=3, wdflag=1, alpha1=1.0,
           pts1=0.001, pts3=0.02, pts2=0.01, epsnov=0.001,
           hewind=0.5, ck=-1000, bwind=0.0, lambdaf=0.5, mxns=2.5,
           beta=0.125, tflag=1, acc2=1.5, nsflag=3, ceflag=0, eddfac=1.0,
           ifflag=0, bconst=-3000, sigma=265.0, gamma=-1.0, pisn=45.0,
           natal_kick_array=[-100.0,-100.0,-100.0,-100.0,-100.0,-100.0],
           bhsigmafrac=1.0, polar_kick_angle=90,
           qcrit_array=[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],
           cekickflag=2, cehestarflag=0, cemergeflag=0,
           ecsn=2.5, ecsn_mlow=1.4, aic=1, ussn=0,
           sigmadiv=-20.0, qcflag=2, eddlimflag=0,
           idum1=None,
           fprimc_array=[2.0/21.0,2.0/21.0,2.0/21.0,2.0/21.0,2.0/21.0,2.0/21.0,2.0/21.0,2.0/21.0,2.0/21.0,2.0/21.0,2.0/21.0,2.0/21.0,2.0/21.0,2.0/21.0,2.0/21.0,2.0/21.0],
           bhspinflag=0, bhspinmag=0.0, rejuv_fac=1.0, rejuvflag=0,
           htpmb=1, ST_cr=1, ST_tide=0, bdecayfac=1):

    natal_kick_array = [v_kick1, v_kick2, theta_kick1, theta_kick2, phi_kick1, phi_kick2]

    if idum1 is None:
        idum1 = -np.random.randint(np.iinfo(np.int32).min, np.iinfo(np.int32).max, size=1)


    # Set input variables
    _evolvebin.windvars.neta = neta
    _evolvebin.windvars.bwind = bwind
    _evolvebin.windvars.hewind = hewind
    _evolvebin.cevars.alpha1 = alpha1
    _evolvebin.cevars.lambdaf = lambdaf
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
    _evolvebin.rand1.idum1 = idum1
    _evolvebin.flags.bhspinflag = bhspinflag
    _evolvebin.snvars.bhspinmag = bhspinmag
    _evolvebin.mixvars.rejuv_fac = rejuv_fac
    _evolvebin.flags.rejuvflag = rejuvflag
    _evolvebin.flags.htpmb = htpmb
    _evolvebin.flags.st_cr = ST_cr
    _evolvebin.flags.st_tide = ST_tide
    _evolvebin.cmcpass.using_cmc = 0

# 0 k1,
# 1 k2,
# 2 m1,
# 3 m2,
# 4 porb,
# 5 ecc,
# 6 metallicity,
# 7 tphysf,
# 8 mass0_1,
# 9 mass0_2,
# 10 rad1,
# 11 rad2,
# 12 lumin1,
# 13 lumin2,
# 14 massc1,
# 15 massc2,
# 16 radc1,
# 17 radc2,
# 18 menv1,
# 19 menv2,
# 20 renv1,
# 21 renv2,
# 22 ospin1,
# 23 ospin2,
# 24 b_0_1,
# 25 b_0_2,
# 26 bacc1,
# 27 bacc2,
# 28 tacc1,
# 29 tacc2,
# 30 epoch1,
# 31 epoch2,
# 32 tms1,
# 33 tms2,
# 34 bhspin1,
# 35 bhspin2,
# 36 tphys,
# 37 binfrac

    print(_evolvebin.windvars.neta,
            _evolvebin.windvars.bwind,
            _evolvebin.windvars.hewind,
            _evolvebin.cevars.alpha1,
            _evolvebin.cevars.lambdaf,
            _evolvebin.ceflags.ceflag,
            _evolvebin.flags.tflag,
            _evolvebin.flags.ifflag,
            _evolvebin.flags.wdflag,
            _evolvebin.snvars.pisn,
            _evolvebin.flags.bhflag,
            _evolvebin.flags.nsflag,
            _evolvebin.ceflags.cekickflag,
            _evolvebin.ceflags.cemergeflag,
            _evolvebin.ceflags.cehestarflag,
            _evolvebin.windvars.mxns,
            _evolvebin.points.pts1,
            _evolvebin.points.pts2,
            _evolvebin.points.pts3,
            _evolvebin.snvars.ecsn,
            _evolvebin.snvars.ecsn_mlow,
            _evolvebin.flags.aic,
            _evolvebin.ceflags.ussn,
            _evolvebin.snvars.sigma,
            _evolvebin.snvars.sigmadiv,
            _evolvebin.snvars.bhsigmafrac,
            _evolvebin.snvars.polar_kick_angle,
            _evolvebin.snvars.natal_kick_array,
            _evolvebin.cevars.qcrit_array,
            _evolvebin.windvars.beta,
            _evolvebin.windvars.xi,
            _evolvebin.windvars.acc2,
            _evolvebin.windvars.epsnov,
            _evolvebin.windvars.eddfac,
            _evolvebin.windvars.gamma,
            _evolvebin.flags.bdecayfac,
            _evolvebin.magvars.bconst,
            _evolvebin.magvars.ck,
            _evolvebin.flags.windflag,
            _evolvebin.flags.qcflag,
            _evolvebin.windvars.eddlimflag,
            _evolvebin.tidalvars.fprimc_array,
            _evolvebin.rand1.idum1,
            _evolvebin.flags.bhspinflag,
            _evolvebin.snvars.bhspinmag,
            _evolvebin.mixvars.rejuv_fac,
            _evolvebin.flags.rejuvflag,
            _evolvebin.flags.htpmb,
            _evolvebin.flags.st_cr,
            _evolvebin.flags.st_tide,
            _evolvebin.cmcpass.using_cmc
            )




    [bpp, bcm] = _evolvebin.evolv2([1, 1],
                                   [mass1, mass2],
                                   orbital_period,
                                   ecc,
                                   metallicity,
                                   tphysf,
                                   tphysf,  # dtp,
                                   [mass1, mass2],
                                   [0, 0],
                                   [0, 0],
                                   [0, 0],
                                   [0, 0],
                                   [0, 0],
                                   [0, 0],
                                   [0, 0],
                                   [0, 0],
                                   [0, 0],
                                   [0, 0],
                                   [0, 0],
                                   [0, 0],
                                   [0, 0],
                                   # [mass0_1, mass0_2],
                                   # [rad1, rad2],
                                   # [lumin1, lumin2],
                                   # [massc1, massc2],
                                   # [radc1, radc2],
                                   # [menv1, menv2],
                                   # [renv1, renv2],
                                   # [ospin1, ospin2],
                                   # [b_0_1, b_0_2],
                                   # [bacc1, bacc2],
                                   # [tacc1, tacc2],
                                   # [epoch1, epoch2],
                                   # [tms1, tms2],
                                   # [bhspin1, bhspin2],
                                   0.0,
                                   np.zeros(20),
                                   np.zeros(20))


    print([1, 1],
           [mass1, mass2],
           orbital_period,
           ecc,
           metallicity,
           tphysf,
           tphysf,  # dtp,
           [mass1, mass2],
           [0, 0],
           [0, 0],
           [0, 0],
           [0, 0],
           [0, 0],
           [0, 0],
           [0, 0],
           [0, 0],
           [0, 0],
           [0, 0],
           [0, 0],
           [0, 0],
           [0, 0],
           # [mass0_1, mass0_2],
           # [rad1, rad2],
           # [lumin1, lumin2],
           # [massc1, massc2],
           # [radc1, radc2],
           # [menv1, menv2],
           # [renv1, renv2],
           # [ospin1, ospin2],
           # [b_0_1, b_0_2],
           # [bacc1, bacc2],
           # [tacc1, tacc2],
           # [epoch1, epoch2],
           # [tms1, tms2],
           # [bhspin1, bhspin2],
           0.0,
           np.zeros(20),
           np.zeros(20))


    return bcm, bpp
