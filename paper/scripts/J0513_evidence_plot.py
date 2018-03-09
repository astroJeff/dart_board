import numpy as np
import matplotlib
matplotlib.rc('text', usetex=True)
matplotlib.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]

import matplotlib.pyplot as plt




# Load data
likelihood = np.exp(np.load("../data/J0513_evidence_lnlikelihood.npy"))

chains = np.load("../data/J0513_evidence_chain.npy")
# Create flatchains
nwalkers, nsteps, nvars = chains.shape
chains = chains.reshape((nwalkers*nsteps, nvars))



# Create figure of individual samples
plt.figure(figsize=(5,3))

idx = np.where(likelihood != 0.0)[0]

plt.scatter(np.log10(np.exp(chains[idx,2])), np.log10(likelihood[idx]), marker='.')

plt.ylim(-10,2)
plt.axvline(np.log10(500.0), color='k', linestyle='dashed')

plt.xlabel("Log Orbital Separation (R$_{\odot}$)", fontsize=12)
plt.ylabel("Log Likelihood", fontsize=12)

plt.tight_layout()
plt.savefig("../figures/J0513_evidence_orb_sep.pdf")
# plt.show()





# Calculate the ratio of evolutionary channels
# 500 Rsun is the separating orbital separation
def calc_P_ratio(sep, likelihood):
    idx_low = np.where(sep < 500.0)[0]
    N_low = len(idx_low)

    idx_high = np.where(sep > 500.0)[0]
    N_high = len(idx_high)

    P_low_over_P_high = np.sum(likelihood[idx_low]) / np.sum(likelihood[idx_high])

    return P_low_over_P_high



# Create evidence ratio figure
fig, ax = plt.subplots(1,1, figsize=(4,2.5))


idx = np.where(likelihood != 0.0)[0]

bins = np.linspace(-1.9, 3.9, 30)

# Run the bootstrapping algorithm
P_ratio = np.array([])
for j in range(1000):
    N_idx = len(idx)
    idx_small = np.random.choice(idx, size=int(0.95*N_idx))
    P_low_over_P_high = calc_P_ratio(np.exp(chains[idx_small,2]), likelihood[idx_small])
    P_ratio = np.append(P_ratio, P_low_over_P_high)


ax.hist(np.log10(P_ratio), bins=bins, normed=True, histtype='step', linewidth=2)

ax.set_yticklabels([])
ax.set_xlim(0.0,2.2)


ax.set_xlabel(r"Log Evidence Ratio $\frac{P(C_1 | \boldsymbol{D})}{P(C_2 | \boldsymbol{D})}$", fontsize=14)
ax.set_ylabel("Number", fontsize=14)

plt.tight_layout()
plt.savefig("../figures/J0513_evidence_ratio.pdf")
# plt.show()
