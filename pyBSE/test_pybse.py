import pybse

output = pybse.evolve(16.8, 6.18, 1018.7, 0.61, 50.3, 0.82, 0.37, \
                      50.3, 0.82, 0.37, 50.0, 0.008, False)

truths = 1.7052747,  6.20448112,  28.87253189,  0.6040554,  81.11018372,  \
            0.,  0.,  13.25808144,  0.,   1.40000047e-05,  4.57327557, \
            105125.3671875,  18684.19335938,   2.11840415e-05,  2255.65625, \
            13, 1

diff = []
for i in range(17):
    diff.append((truths[i] - output[i]) / max(truths[i], 1.0e-10))

    if diff[i] > 1.0e-5:
        print("Test failed")
        exit()

print("pyBSE has been successfully installed.")
