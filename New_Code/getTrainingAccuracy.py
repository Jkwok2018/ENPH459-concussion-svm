#directory = pwd;
#[fMatrix,labels] = featureExtraction(directory);
# "fitcsvm" or "fitclinear"
## This section loads preprocessed labels and feature matrix. This is because this takes the longest
import numpy as np
from numpy.lib.function_base import average
import scipy.io
from SVM.createSVMModel import createSVMModel
from Result.getAccuracy import getAccuracy
from Result.returnKfoldResults import returnKfoldResults


eeg_file = scipy.io.loadmat('FeatureMatrix.mat')
fMatrix = eeg_file["data"]
labels = eeg_file["labels"]

#Testing SVM
tries = 100
accuracy = 0
for i in range(tries):
    kfoldAccuracy = createSVMModel(fMatrix, labels, 'fitcsvm')
    # kfoldtestLabels = returnKfoldResults(kFoldsvmModel,fMatrix)
    # kfoldAccuracy = getAccuracy(kfoldtestLabels,labels)
    accuracy += kfoldAccuracy

average_accuracy = accuracy/tries
print(f"SVM accuracy: {average_accuracy}")

# accuracy = 0 # need to add accuracy
#label = predict(SVMModel,X)
# for i in np.arange(1,tries+1).reshape(-1):
#     kFoldsvmModel,holdoutsvmModel,leaveoutsvmModel = createSVMModel(fMatrix,labels,'fitcsvm')
#     kfoldtestLabels = returnKfoldResults(kFoldsvmModel,fMatrix)
#     kfoldAccuracy = getAccuracy(kfoldtestLabels,labels)
#     accuracy[i,1] = kfoldAccuracy
    # holdouttestLabels = predict(holdoutsvmModel,fMatrix)
    # holdoutAccuracy = getAccuracy(holdouttestLabels,labels)
    # accuracy[i,2] = holdoutAccuracy
    # leaveouttestLabels = returnLeaveOutResult(leaveoutsvmModel,fMatrix)
    # leaveoutAccuracy = getAccuracy(leaveouttestLabels,labels)
    # accuracy[i,3] = leaveoutAccuracy

# avgKfoldAccuracy = accuracy / float(tries)
# avgHoldOutAccuracy = sum(accuracy(:,2)) / tries
# avgLeaveOutAccuracy = sum(accuracy(:,3)) / tries
