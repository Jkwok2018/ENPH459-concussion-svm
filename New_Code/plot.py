from statistics import fmean
import scipy.io
import matplotlib.pyplot as plt
from FeatureExtraction.extractFeatures import extractFeatures

fMatrix = extractFeatures('/Users/melodyzhao/Desktop/Python/ENPH459-concussion-svm/New_Code/Controls/NAW/NAW.mat')
print(fMatrix.shape)

fig, axs = plt.subplots(7)
for i in range(7):
    axs[i].plot(range(27), fMatrix[i])
    # plt.plot(range(27), fMatrix[i])
plt.show()