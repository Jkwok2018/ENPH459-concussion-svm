import numpy as np
    
def getAccuracy(svmLabels = None,actualLabels = None): 
    #GETACCURACY Returns accuracy of SVM predictions
#   Inputs: svmLabels: predicted labels output by SVM
#           actualLabels: actual correct labels confirmed by doctors
#   Outputs: accuracyRate: the # of labels that were predicted correctly
    
    rowSize,colSize = svmLabels.shape
    missclass = 0
    for row in range(rowSize):
        if svmLabels[row,0] != actualLabels[row,0]:
            missclass += 1
    
    accuracyRate = (rowSize - missclass) / rowSize
    return accuracyRate
