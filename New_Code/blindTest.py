#directory = pwd;
#[fMatrix,labels] = featureExtraction(directory);
# "fitcsvm" or "fitclinear"

## This section loads preprocessed labels and feature matrix. This is because this takes the longest
import numpy as np
featureMatrix = scipy.io.loadmat('featureMatrix.mat')
field = fieldnames(featureMatrix)
fieldName = field[0]
featureMatrix = getattr(featureMatrix,(fieldName))
testMatrix = scipy.io.loadmat('blindTestFeatures.mat')
field = fieldnames(testMatrix)
fieldName = field[0]
testMatrix = getattr(testMatrix,(fieldName))
labels = scipy.io.loadmat('correctLabels.mat')
field = fieldnames(labels)
fieldName = field[0]
labels = getattr(labels,(fieldName))
## Testing of SVM Model below
tries = 1
accuracy = np.zeros((tries,3))
for i in np.arange(1,tries+1).reshape(-1):
    kFoldsvmModel,holdoutsvmModel,leaveoutsvmModel = createSVMModel(featureMatrix,labels,'fitcsvm')
    kfoldtestLabels = returnKfoldResults(kFoldsvmModel,testMatrix)
    holdouttestLabels = predict(holdoutsvmModel,testMatrix)
    leaveouttestLabels = returnLeaveOutResult(leaveoutsvmModel,testMatrix)
