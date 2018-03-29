

##################### CONSTANTS TO BE USED THROUGHOUT MODULE #####################
G = 6.674e-8 # Gravitational constant in cgs
GGG = 1.909e5 # Gravitational constant in Rsun * (km/s)^2 / Msun
c_light = 2.9979e10 # speed of light in cgs
km_to_cm = 1.0e5 # km to cm
Msun_to_g = 1.989e33 # Msun to g
Rsun_to_cm = 6.995e10 # Rsun to cm
AU_to_cm = 1.496e13 # AU to cm
pc_to_cm = 3.086e18 # parsec to cm
pc_to_km = 3.086e13 # parsec to km
yr_to_sec = 31557600.0 # Sec in yr
day_to_sec = 3600.0*24.0 # Sec in day
deg_to_rad = 0.0174532925199 # Degrees to radians
rad_to_deg = 57.2957795131 # Radians to degrees
asec_to_rad = 4.84814e-6 # Arcsec to radians


R_NS = 10.0  # NS radius in km
eta_bol = 0.15  # Calibrated to Chandra X-ray sensitivity
v_kick_sigma = 265.0  # Kick velocity Maxwellian dispersion - Fe-core collapse
v_kick_sigma_ECS = 50.0  # Kick velocity Maxwellian dispersion - ECS
alpha = -2.35  # IMF index



##################### PARAMETER RANGES #####################
min_mass_M1 = 8.0   # in Msun
max_mass_M1 = 150.0   # in Msun
min_mass_M2 = 2.0   # in Msun
max_mass_M2 = 150.0   # in Msun
min_a = 1.0e1   # In Rsun
max_a = 1.0e5   # In Rsun
min_t = 0.0
max_t = 1.0e4  # in Myr
min_z = 5.0e-5
max_z = 0.03

ra_max = None
ra_min = None
dec_max = None
dec_min = None



###################  DISTANCE TO OBJECT #################
distance = None


dist_LMC = 5.0e4 * pc_to_km # Distance to Large Magellanic Cloud (in km)
dist_SMC = 6.1e4 * pc_to_km # Distance to Small Magellanic Cloud (in km)
dist_NGC4244 = 4.3e6 * pc_to_km # Distance to NGC4244 (in km)
dist_NGC660 = 13.0e6 * pc_to_km # Distance to NGC 660 (13 Mpc in km)
