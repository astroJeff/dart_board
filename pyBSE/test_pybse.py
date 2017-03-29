import pybse

output = pybse.evolv_wrapper(1, 16.8, 6.18, 1018.7, 0.61, 197.3, 0.82, 0.37, \
                             197.3, 0.82, 0.37, 50.0, 0.008, True)

truths = 1.9201537370681763, 0.0, 0.0, -1.0, 13.0, 15.0, 0.0, 1.3650274865995016e-07

diff = []
for i in range(8):
    diff.append((truths[i] - output[i]) / max(truths[i], 1.0e-10))

    if diff[i] > 1.0e-5:
        print("Test failed")
        exit()

print("pyBSE has been successfully installed.")
