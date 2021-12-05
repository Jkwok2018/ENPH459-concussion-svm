#{
import numpy as np
matrix = np.array([[1,1,1],[1,1,2],[2,1,3],[2,1,1],[3,2,2],[3,3,3]])
#[shannonmat, uniquesymbols, probmatrix] = getShannonEntropy(eegMatrix);
shannonmat = getShannonEntropy(matrix,1)
eegStruct = scipy.io.loadmat('BN.mat')
field = fieldnames(eegStruct)
fieldName = field[0]
eegMatrix = getattr(eegStruct,(fieldName))
entropy = getShannonEntropy(eegMatrix,2)
#}

## formatFeaturesForTraining Testing
#{
testMatrix = np.array([[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27],[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27],[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27],[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27]])
formattedMatrix = formatFeaturesForTraining(testMatrix)
#}
## getAccuracy Testing
# have 5 mistakes out of 25, so classification accuracy of 80# should be
# returned
#{
svmResults = np.array([[1],[1],[0],[1],[1],[1],[1],[0],[1],[1],[0],[1],[1],[1],[1],[0],[0],[0],[1],[1],[1],[1],[1],[0],[1]])
correctResults = np.array([[1],[1],[0],[1],[1],[0],[1],[1],[1],[1],[0],[1],[1],[1],[1],[0],[0],[1],[1],[1],[1],[0],[1],[1],[1]])
accuracy = getAccuracy(svmResults,correctResults)
#}
## Testing feature extraction label making
#{
concussedFilenames = np.array([[1],[1],[1],[1],[1],[1],[1],[1],[1],[1]])

controlFilenames = np.array([[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1]])

labels = np.ones((len(concussedFilenames) + len(controlFilenames),1))
labels[np.arange[len[concussedFilenames] + 1,len[labels]+1],1] = 0

#}
