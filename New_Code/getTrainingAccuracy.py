#directory = pwd;
#[fMatrix,labels] = featureExtraction(directory);
# "fitcsvm" or "fitclinear"
## This section loads preprocessed labels and feature matrix. This is because this takes the longest
import numpy as np
fMatrix = scipy.io.loadmat('featureMatrix.mat')
field = fieldnames(fMatrix)
fieldName = field[0]
fMatrix = getattr(fMatrix,(fieldName))
labels = scipy.io.loadmat('correctLabels.mat')
field = fieldnames(labels)
fieldName = field[0]
labels = getattr(labels,(fieldName))
## Testing of SVM Model below
#[kFoldsvmModel, holdoutsvmModel, leaveoutsvmModel] = createSVMModel(fMatrix,labels,"fitcsvm");
tries = 1000
accuracy = np.zeros((tries,3))
#label = predict(SVMModel,X)
for i in np.arange(1,tries+1).reshape(-1):
    kFoldsvmModel,holdoutsvmModel,leaveoutsvmModel = createSVMModel(fMatrix,labels,'fitcsvm')
    kfoldtestLabels = returnKfoldResults(kFoldsvmModel,fMatrix)
    kfoldAccuracy = getAccuracy(kfoldtestLabels,labels)
    accuracy[i,1] = kfoldAccuracy
    holdouttestLabels = predict(holdoutsvmModel,fMatrix)
    holdoutAccuracy = getAccuracy(holdouttestLabels,labels)
    accuracy[i,2] = holdoutAccuracy
    leaveouttestLabels = returnLeaveOutResult(leaveoutsvmModel,fMatrix)
    leaveoutAccuracy = getAccuracy(leaveouttestLabels,labels)
    accuracy[i,3] = leaveoutAccuracy

avgKfoldAccuracy = sum(accuracy(:,1)) / tries
avgHoldOutAccuracy = sum(accuracy(:,2)) / tries
avgLeaveOutAccuracy = sum(accuracy(:,3)) / tries