import numpy as np
import os

# Location of MIST data
data_dir = "./data/MIST_v1.2_vvcrit0.4_UBVRIplus/"

photo = None

for i, f in enumerate(os.listdir(data_dir)):

    # Only select data files
    if not f.endswith(".iso.cmd"): continue

    # Load up data
    if photo is None:
        photo = np.genfromtxt(data_dir+f, skip_header=12, names=True)
    else:
        photo = np.append(photo, np.genfromtxt(data_dir+f, skip_header=12, names=True))

    print(f)


np.save("./data/MIST_v1.2_vvcrit0.4_UBVRIplus.npy", photo)
