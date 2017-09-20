import numpy as np
import pickle
import sys



file_name = sys.argv[1]
in_file = "../data/" + file_name + "_chain.obj"


if len(sys.argv) == 2:
    delay = 200
else:
    delay = int(int(sys.argv[2]) / 100)


chains = pickle.load(open(in_file, "rb"))
if chains.ndim == 3:
    chains = chains[:,delay:,:]
elif chains.ndim == 4:
    chains = chains[0,:,delay:,:]
else:
    sys.exit()

n_chains, length, n_var = chains.shape
chains = chains.reshape((n_chains*length, n_var))
print(chains.shape)


chains[:,0] = np.exp(chains[:,0])
chains[:,1] = np.exp(chains[:,1])
chains[:,2] = np.exp(chains[:,2])
chains[:,-1] = np.exp(chains[:,-1])


def confidence_level(data, level):
    return np.sort(data)[level*len(data)]

print()
print(file_name)
print("1-sigma confidence levels.")

for i in range(nvar):

    # 1-sigma confidence levels
    lower = confidence_level(chains[:,i], 0.1587)
    upper = confidence_level(chains[:,i], 0.8413)
    print("Variable:", i, "mean:", np.mean(chains[:,i]), "mediam", np.median(chains[:,i]), lower, upper)

print()
