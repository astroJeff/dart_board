import pybse

output = pybse.evolve(16.8, 6.18, 1018.7, 0.61, 50.3, 0.82, 0.37, \
                      50.3, 0.82, 0.37, 50.0, 0.008, False)

truths = 1.6422327756881714, 6.223701477050781, 27.427093505859375, 0.5741786956787109, 87.99896240234375, 0.0, 13.290934562683105, 13.0, 1.0

diff = []
for i in range(9):
    diff.append((truths[i] - output[i]) / max(truths[i], 1.0e-10))

    if diff[i] > 1.0e-5:
        print("Test failed")
        exit()

# print(output)
# print()

print("pyBSE has been successfully installed.")
