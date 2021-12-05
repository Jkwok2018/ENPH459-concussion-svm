import numpy as np
    
def getAccuracy(svmLabels = None,actualLabels = None): 
    #GETACCURACY Returns accuracy of SVM predictions
#   Inputs: svmLabels: predicted labels output by SVM
#           actualLabels: actual correct labels confirmed by doctors
#   Outputs: accuracyRate: the # of labels that were predicted correctly
    
    rowSize,colSize = svmLabels.shape
    missclass = 0
    for row in np.arange(1,rowSize+1).reshape(-1):
        if svmLabels(row,1) != actualLabels(row,1):
            missclass = missclass + 1
    
    accuracyRate = (rowSize - missclass) / rowSize
    return accuracyRate
    
    return accuracyRate