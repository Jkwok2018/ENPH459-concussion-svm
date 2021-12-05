## This section loads preprocessed labels and feature matrix. This is because this takes the longest
import numpy as np
featureMatrixForTraining = scipy.io.loadmat('featureMatrixForTraining.mat')
field = fieldnames(featureMatrixForTraining)
fieldName = field[0]
featureMatrixForTraining = getattr(featureMatrixForTraining,(fieldName))
blindFeatureMatrix = scipy.io.loadmat('blindFeatureMatrix.mat')
field = fieldnames(blindFeatureMatrix)
fieldName = field[0]
blindFeatureMatrix = getattr(blindFeatureMatrix,(fieldName))
labelsForTraining = scipy.io.loadmat('labelsForTraining.mat')
field = fieldnames(labelsForTraining)
fieldName = field[0]
labelsForTraining = getattr(labelsForTraining,(fieldName))
## Testing of SVM Model below
tries = 1
for i in np.arange(1,tries+1).reshape(-1):
    kFoldsvmModel,holdoutsvmModel,leaveoutsvmModel = createSVMModel(featureMatrixForTraining,labelsForTraining,'fitcsvm')
    kfoldtestLabels = returnKfoldResults(kFoldsvmModel,blindFeatureMatrix)
    holdouttestLabels = predict(holdoutsvmModel,blindFeatureMatrix)
    leaveouttestLabels = returnLeaveOutResult(leaveoutsvmModel,blindFeatureMatrix)
