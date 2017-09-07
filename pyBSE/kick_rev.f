***
      SUBROUTINE kick_rev(kw,m1,m1n,m2,ecc,sep,jorb,vs,vk,
     &                        theta_in,phi_in)
      implicit none
*
      integer kw,k
      INTEGER idum
      COMMON /VALUE3/ idum
      INTEGER idum2,iy,ir(32)
      COMMON /RAND3/ idum2,iy,ir
      integer bhflag
      real*8 m1,m2,m1n,ecc,sep,jorb,ecc2
      real*8 pi,twopi,gmrkm,yearsc,rsunkm
      parameter(yearsc=3.1557d+07,rsunkm=6.96d+05)
      real*8 mm,em,dif,der,del,r
      real*8 u1,u2,vk,v(4),s,theta,phi,theta_in,phi_in
      real*8 sphi,cphi,stheta,ctheta,salpha,calpha
      real*8 vr,vr2,vk2,vn2,hn2
      real*8 mu,cmu,vs(3),v1,v2,mx1,mx2
      real*8 sigma
      COMMON /VALUE4/ sigma,bhflag
      real ran3,xx
      external ran3
*
      do k = 1,3
         vs(k) = 0.d0
      enddo
*     if(kw.eq.14.and.bhflag.eq.0) goto 95
*
      pi = ACOS(-1.d0)
      twopi = 2.d0*pi
* Conversion factor to ensure velocities are in km/s using mass and
* radius in solar units.
      gmrkm = 1.906125d+05
* Assume a circular orbit
* Find the initial relative velocity vector.
      salpha = 1.0
      calpha = 0.0
      vr2 = gmrkm*(m1+m2)*(1.d0/sep)
      vr = SQRT(vr2)
*      vr = 0.d0
*      vr2 = 0.d0
*      salpha = 0.d0
*      calpha = 0.d0
*
* Use the input vick speed and angles
* BSE uses phi as the polar angle, and theta as the azimuthal
* How annoying.
*
      phi = theta_in
      theta = phi_in
*
      sphi = SIN(phi)
      cphi = COS(phi)
      stheta = SIN(theta)
      ctheta = COS(theta)
*     WRITE(66,*)' KICK VK PHI THETA ',vk,phi,theta
*      if(sep.le.0.d0.or.ecc.lt.0.d0) goto 90
      if(sep.le.0.d0.or.ecc.lt.0.d0) return
*
* Other inputs
      vk2 = vk*vk
      r = sep
* Determine the magnitude of the new relative velocity.
*      vn2 = vk2+vr2-2.d0*vk*vr*(ctheta*cphi*salpha-stheta*cphi*calpha)
      vn2 = vk2+vr2+2.d0*vk*vr*cphi
* Calculate the new semi-major axis.
      sep = 2.d0/r - vn2/(gmrkm*(m1n+m2))
      sep = 1.d0/sep
*     if(sep.le.0.d0)then
*        ecc = 1.1d0
*        goto 90
*     endif
* Determine the magnitude of the cross product of the separation vector
* and the new relative velocity.
      v1 = vk2 * sphi*sphi * stheta*stheta
      v2 = (vk*cphi + vr)**2
      hn2 = r*r*(v1 + v2)
* Calculate the new eccentricity.
      ecc2 = 1.d0 - hn2/(gmrkm*sep*(m1n+m2))
      ecc2 = MAX(ecc2,0.d0)
      ecc = SQRT(ecc2)

* Calculate the new orbital angular momentum taking care to convert
* hn to units of Rsun^2/yr.
      jorb = (m1n*m2/(m1n+m2))*SQRT(hn2)*(yearsc/rsunkm)
* Determine the angle between the new and old orbital angular
* momentum vectors.
      cmu = (vr*salpha-vk*ctheta*cphi)/SQRT(v1 + v2)
      mu = ACOS(cmu)
* Calculate the components of the velocity of the new centre-of-mass.
 90   continue
      if(ecc.le.1.0)then
* Calculate the components of the velocity of the new centre-of-mass.
         mx1 = vk*m1n/(m1n+m2)
         mx2 = vr*(m1-m1n)*m2/((m1n+m2)*(m1+m2))
         vs(1) = mx1*ctheta*cphi + mx2*salpha
         vs(2) = mx1*stheta*cphi + mx2*calpha
         vs(3) = mx1*sphi
      else
* Calculate the relative hyperbolic velocity at infinity (simple method).
         sep = r/(ecc-1.d0)
*        cmu = SQRT(ecc-1.d0)
*        mu = ATAN(cmu)
         mu = ACOS(1.d0/ecc)
         vr2 = gmrkm*(m1n+m2)/sep
         vr = SQRT(vr2)
         vs(1) = vr*SIN(mu)
         vs(2) = vr*COS(mu)
         vs(3) = 0.d0
         ecc = MIN(ecc,99.99d0)
      endif
*
 95   continue
*
      RETURN
      END
***
