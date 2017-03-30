import pybse

output = pybse.evolv_wrapper(1, 16.8, 6.18, 1018.7, 0.61, 197.3, 0.82, 0.37, \
                             197.3, 0.82, 0.37, 50.0, 0.008, False)

truths = 2.0356035232543945, 0.9115774035453796, 2.88948130607605, 0.0, 0.0, 3.4887122080107247e-09, 0.0, 13.0, 1.0

diff = []
for i in range(9):
    diff.append((truths[i] - output[i]) / max(truths[i], 1.0e-10))

    if diff[i] > 1.0e-5:
        print("Test failed")
        exit()

print("pyBSE has been successfully installed.")
