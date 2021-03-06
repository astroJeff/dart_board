***
      SUBROUTINE evolv_wrapper(num_bin, m1, m2, tb, ecc,
     &                         v_kick1, theta_kick1, phi_kick1,
     &                         omega_kick1, v_kick2, theta_kick2,
     &                         phi_kick2, omega_kick2, tmax, z,
     &                         out_flag, idum_in,
     &                         neta_in, bwind_in, hewind_in,
     &                         alpha1_in, lambda_in, ceflag_in,
     &                         tflag_in, ifflag_in, wdflag_in,
     &                         GRflag_in,
     &                         bhflag_in, nsflag_in, mxns_in,
     &                         pts1_in, pts2_in, pts3_in, sigma_in,
     &                         beta_in, xi_in, acc2_in, epsnov_in,
     &                         eddfac_in, gamma_in,
     &                         m1_out, m2_out, a_out, ecc_out,
     &                         v_sys_out, mdot1_out, mdot2_out,
     &                         t_SN1, t_SN2, r1_out, r2_out,
     &                         teff1_out, teff2_out,
     &                         lum1_out, lum2_out,
     &                         k1_out, k2_out)
***
*
* Evolves a population of binaries using input parameters
*
***
      implicit none
*
      INCLUDE 'const_bse.h'
*
      integer i,j,k,jj,jp,nm1, last
      integer kw,kw2,kwx,kwx2,kstar(2)
      integer i1,i2,kdum
      integer, intent(in) :: num_bin
*
      real*8, intent(in) :: tb, ecc
      real*8, intent(in) :: v_kick1, theta_kick1, phi_kick1, omega_kick1
      real*8, intent(in) :: v_kick2, theta_kick2, phi_kick2, omega_kick2
      real*8, intent(in) :: m1, m2, tmax, z
      integer, intent(in) :: idum_in
      real*8, intent(in) :: neta_in, bwind_in, hewind_in
      real*8, intent(in) :: alpha1_in, lambda_in, mxns_in
      integer, intent(in) :: wdflag_in, nsflag_in, bhflag_in
      integer, intent(in) :: GRflag_in
      integer, intent(in) :: ceflag_in, tflag_in, ifflag_in
      real*8, intent(in) :: pts1_in, pts2_in, pts3_in, sigma_in
      real*8, intent(in) :: beta_in, xi_in, acc2_in, epsnov_in
      real*8, intent(in) :: eddfac_in, gamma_in
      real*8 mass0(2),mass(2),zpars(20)
      real*8 epoch(2),tms(2),tphys,tphysf,dtp
      real*8 rad(2),lum(2),ospin(2)
      real*8 massc(2),radc(2),menv(2),renv(2)
      real*8 sep0,tb0,ecc0,aursun,yeardy,yearsc,tol
      real*8 twopi,stef_boltz,Rsun
      PARAMETER(aursun=214.95d0,yeardy=365.25d0,yearsc=3.1557d+07)
      PARAMETER(tol=1.d-07)
      real*8 t1,t2,mx,mx2,tbx,eccx
      real*8 p_out
      real*8 mdot1, mdot2
      real*8, intent(out) :: m1_out, m2_out, a_out, ecc_out
      real*8, intent(out) :: v_sys_out, mdot1_out, mdot2_out
      real*8, intent(out) :: t_SN1, t_SN2
      real*8, intent(out) :: r1_out, r2_out, teff1_out, teff2_out
      real*8, intent(out) :: lum1_out, lum2_out
      real*8, intent(out) :: k1_out, k2_out
      logical out_flag
      CHARACTER*8 label(14)
*
************************************************************************
* BSE parameters:
*
* neta is the Reimers mass-loss coefficent (neta*4x10^-13: 0.5 normally).
* bwind is the binary enhanced mass loss parameter (inactive for single).
* hewind is a helium star mass loss factor (1.0 normally).
* alpha1 is the common-envelope efficiency parameter (1.0).
* lambda is the binding energy factor for common envelope evolution (0.5).
*
* ceflag > 0 activates spin-energy correction in common-envelope (0).
* tflag > 0 activates tidal circularisation (1).
* ifflag > 0 uses WD IFMR of HPE, 1995, MNRAS, 272, 800 (0).
* wdflag > 0 uses modified-Mestel cooling for WDs (0).
* bhflag > 0 allows velocity kick at BH formation (0).
* nsflag > 0 takes NS/BH mass from Belczynski et al. 2002, ApJ, 572, 407 (1).
* mxns is the maximum NS mass (1.8, nsflag=0; 3.0, nsflag=1).
* idum is the random number seed used by the kick routine.
*
* Next come the parameters that determine the timesteps chosen in each
* evolution phase:
*                 pts1 - MS                  (0.05)
*                 pts2 - GB, CHeB, AGB, HeGB (0.01)
*                 pts3 - HG, HeMS            (0.02)
* as decimal fractions of the time taken in that phase.
*
* sigma is the dispersion in the Maxwellian for the SN kick speed (190 km/s).
* beta is wind velocity factor: proportional to vwind**2 (1/8).
* xi is the wind accretion efficiency factor (1.0).
* acc2 is the Bondi-Hoyle wind accretion factor (3/2).
* epsnov is the fraction of accreted matter retained in nova eruption (0.001).
* eddfac is Eddington limit factor for mass transfer (1.0).
* gamma is the angular momentum factor for mass lost during Roche (-1.0).
*
      neta = neta_in
      bwind = bwind_in
      hewind = hewind_in
      alpha1 = alpha1_in
      lambda = lambda_in
      ceflag = ceflag_in
      tflag = tflag_in
      ifflag = ifflag_in
      wdflag = wdflag_in
      GRflag = GRflag_in
      bhflag = bhflag_in
      nsflag = nsflag_in
      mxns = mxns_in
      pts1 = pts1_in
      pts2 = pts2_in
      pts3 = pts3_in
      sigma = sigma_in
      beta = beta_in
      xi = xi_in
      acc2 = acc2_in
      epsnov = epsnov_in
      eddfac = eddfac_in
      gamma = gamma_in

*
* Set the seed for the random number generator.
*
      idum = idum_in
      if(idum.gt.0) idum = -idum
*
* Set the collision matrix.
*
      CALL instar
*
      label(1) = 'INITIAL '
      label(2) = 'KW CHNGE'
      label(3) = 'BEG RCHE'
      label(4) = 'END RCHE'
      label(5) = 'CONTACT '
      label(6) = 'COELESCE'
      label(7) = 'COMENV  '
      label(8) = 'GNTAGE  '
      label(9) = 'NO REMNT'
      label(10) = 'MAX TIME'
      label(11) = 'DISRUPT '
      label(12) = 'BEG SYMB'
      label(13) = 'END SYMB'
      label(14) = 'BEG BSS'
*
*
* Open the input file - list of binary initial parameters.
*
*      OPEN(10,file='binaries.in',status='unknown')
*      READ(10,*)nm1
*
* Open the output files.
*
*      OPEN(11,file='binaries.out',status='unknown')
*      OPEN(12,file='search.out',status='unknown')
*
* Some constants
*
          twopi = 2.d0*ACOS(-1.d0)
          stef_boltz = 5.6704d-5
          Rsun = 6.957d10
*
      do i = 1,num_bin
*
* Read in parameters and set coefficients which depend on metallicity.
*
*         tb = 0.0
*         READ(10,*)m1,m2,tb,ecc,z,tmax

         CALL zcnsts(z,zpars)
*
         ecc0 = ecc
         tb0 = tb/yeardy
         sep0 = aursun*(tb0*tb0*(mass(1) + mass(2)))**(1.d0/3.d0)
         tb0 = tb

*
* Initialize the binary.
*
         kstar(1) = 1
         mass0(1) = m1
         mass(1) = m1
         massc(1) = 0.0
         ospin(1) = 0.0
         epoch(1) = 0.0
*
         kstar(2) = 1
         mass0(2) = m2
         mass(2) = m2
         massc(2) = 0.0
         ospin(2) = 0.0
         epoch(2) = 0.0
*
         tphys = 0.0
         tphysf = tmax
         dtp = 0.0
*
* Evolve the binary.
*
         CALL evolv2(kstar,mass0,mass,rad,lum,massc,radc,
     &               menv,renv,ospin,epoch,tms,
     &               tphys,tphysf,dtp,z,zpars,tb,ecc,
     &               v_kick1,theta_kick1,phi_kick1,
     &               v_kick2,theta_kick2,phi_kick2)
*
* Search the BCM array for the formation of binaries of
* interest (data on unit 12 if detected) and also output
* the final state of the binary (unit 11).
*
* In this example we will search for CVs.
*

         jj = 0
         do while (bcm(jj,1).lt.tmax)
           m1_out = bcm(jj,4)
           m2_out = bcm(jj,18)
           k1_out = bcm(jj,2)
           k2_out = bcm(jj,16)
           ecc_out = bcm(jj,32)
           p_out = bcm(jj,30)*365.25
           a_out = bcm(jj,31)
           mdot1 = bcm(jj,14)
           mdot2 = bcm(jj,28)
           teff1_out = 10**bcm(jj,7)
           teff2_out = 10**bcm(jj,21)
           r1_out = 10**bcm(jj,6)
           r2_out = 10**bcm(jj,20)

           if(out_flag)then
             write(*,*) bcm(jj,1), m1_out, m2_out, k1_out, k2_out,
     &                   ecc_out, a_out, p_out, mdot1, mdot2,
     &                   teff1_out, teff2_out, r1_out, r2_out
           endif

           if(jj.gt.2.and.bcm(jj,1).eq.0.0)then
             jj = jj - 3
             EXIT
           endif

           jj = jj + 1

         enddo

         last = jj

         m1_out = bcm(last,4)
         m2_out = bcm(last,18)
         k1_out = bcm(last,2)
         k2_out = bcm(last,16)
         ecc_out = bcm(last,32)
         p_out = bcm(last,30)*365.25
         a_out = bcm(last,31)

         r1_out = 10**bcm(last,6)
         r2_out = 10**bcm(last,20)
         lum1_out = 10**bcm(last,5)
         lum2_out = 10**bcm(last,19)
         teff1_out = 10**bcm(last,7)
         teff2_out = 10**bcm(last,21)
         mdot1_out = bcm(last,14)
         mdot2_out = bcm(last,28)
         v_sys_out = bcm(last,33)

* To get t_SN1, we use the bpp array which stores values
* whenever the stellar k-type changes
         jp = 0
         do while (bpp(jp,1).lt.tmax)
           kstar(1) = INT(bpp(jp,4))
           kstar(2) = INT(bpp(jp,5))
* First time the loop encounters a NS or BH, set t_SN1 and exit loop
           if(kstar(1).gt.12.or.kstar(2).gt.12)then
             t_SN1 = bpp(jp,1)
             EXIT
           endif

           jp = jp + 1
         enddo

* To get t_SN2, we use the bpp array which stores values
* When both stellar k-types are 13 or 14, SN2 occurs.
         jp = 0
         do while (bpp(jp,1).lt.tmax)
           kstar(1) = INT(bpp(jp,4))
           kstar(2) = INT(bpp(jp,5))
* Second time the loop encounters a NS or BH, set t_SN1 and exit loop
           if(kstar(1).gt.12.and.kstar(2).gt.12)then
             t_SN2 = bpp(jp,1)
             EXIT
           endif

           jp = jp + 1
         enddo


*         if(out_flag)then
*            write(11,*) bcm(last,1), m1_out, m2_out, k1_out, k2_out,
*     &                 ecc_out, a_out, p_out, mdot1, mdot2
*         endif


         jj = 0
         t1 = -1.0
         t2 = -1.0
 30      jj = jj + 1
         if(bcm(jj,1).lt.0.0) goto 40
         kw = INT(bcm(jj,2))
         kw2 = INT(bcm(jj,16))
*
         i1 = 15
         i2 = 29
         if(kw.gt.kw2)then
            kdum = kw2
            kw2 = kw
            kw = kdum
            i2 = 15
            i1 = 29
         endif
*
         if(kw.le.1.and.bcm(jj,i1).ge.1.0)then
            if(kw2.ge.10.and.kw2.le.12)then
               if(t1.lt.0.0)then
                  t1 = bcm(jj,1)
                  kwx = kw
                  kwx2 = kw2
                  mx = bcm(jj,i1-11)
                  mx2 = bcm(jj,i2-11)
                  tbx = bcm(jj,30)
                  eccx = bcm(jj,32)
               endif
            endif
         endif
*
*         if(t1.gt.0.0.and.(bcm(jj,i1).lt.1.0.or.
*     &      kw.ne.kwx.or.kw2.ne.kwx2))then
*            if(t2.lt.0.0)then
*               t2 = bcm(jj,1)
*               if(t2.gt.(t1+tol))then
*                  WRITE(12,112)m1,m2,ecc0,tb0,t1,t2,kwx,kwx2,
*     &                         mx,mx2,eccx,tbx
*               endif
*               t1 = -1.0
*               t2 = -1.0
*            endif
*         endif
*
         goto 30
 40      continue
*
*         if(t1.gt.0.0)then
*            if(t2.lt.0.0) t2 = tmax
*            WRITE(12,112)m1,m2,ecc0,tb0,t1,t2,kwx,kwx2,mx,mx2,eccx,tbx
*         endif
*
*         jj = jj - 1
*         kw = INT(bcm(jj,2))
*         kw2 = INT(bcm(jj,16))
*         mx = bcm(jj,4)
*         mx2 = bcm(jj,18)
*         tbx = bcm(jj,30)*yeardy
*         eccx = bcm(jj,32)
*         WRITE(11,111)tmax,kw,kw2,mx,mx2,eccx,tbx
*
      enddo
*
 111  FORMAT(f10.1,2i3,3f8.3,1p,e14.6)
 112  FORMAT(3f8.3,1p,e14.6,0p,2f10.2,2i3,3f8.3,1p,e14.6)
*      CLOSE(10)
*      CLOSE(11)
*      CLOSE(12)


* The bpp array acts as a log, storing parameters at each change
* of evolution stage.
*
*50    j = 0
*      WRITE(*,*)'     TIME      M1       M2   K1 K2        SEP    ECC',
*     &          '  R1/ROL1 R2/ROL2  TYPE'
*52    j = j + 1
*      if(bpp(j,1).lt.0.0) goto 60
*      kstar(1) = INT(bpp(j,4))
*      kstar(2) = INT(bpp(j,5))
*      kw = INT(bpp(j,10))
*      WRITE(*,100)(bpp(j,k),k=1,3),kstar,(bpp(j,k),k=6,9),label(kw)
*      goto 52
*60    continue
*100   FORMAT(f11.4,2f9.3,2i3,f13.3,f6.2,2f8.3,2x,a8)
*
*
************************************************************************
*
      RETURN
      END
***
