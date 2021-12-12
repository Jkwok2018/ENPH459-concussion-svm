import numpy as np
import scipy.io
from .powerSpectral import powerSpectral
from .waveletDecompExtract import waveletDecompExtract
from .getShannonEntropy import getShannonEntropy

def extractFeatures(eegMat = None): 
    #EXTRACT_FEATURES Takes in the .mat file of EEG and returns a matrix
#containing 30 features (row) per channel (column)
    
    #   This function takes in the EEG data as a .mat file, converts it to a
#   matlab matrix object then calls on power spectral analysis, Wavelet
#   decomposition, and shannon entropy functions to get a matrix back from
#   each function. Then the matrices are combined into a feature matrix and
#   returned.
#   Input: eegMat - string - name of the .mat file containing the EEG data
#   Output: featureMatrix - 30x27 matrix containing features (row) of each
#   channel (column)
    
    print('file is', eegMat)
    eegStruct = scipy.io.loadmat(eegMat)
    field = eegStruct.keys()
    print(list(field))
    
    # data = eegStruct['data']
    data = eegStruct[list(field)[-1]]
    rows, cols = data.shape
    featureMatrix = np.zeros((28,cols))
    
    powerSpectralMatrix = powerSpectral(data)
    print('p', powerSpectralMatrix.shape) # &7, 27
    featureMatrix[0:7, 0:27] = powerSpectralMatrix #(0,0) to (7, 27)
    
    # # Wavelet Decomposition Analysis
    waveletDecompMatrix = waveletDecompExtract(data,'db8')
    print('w', waveletDecompMatrix.shape) # 20, 27
    featureMatrix[7:27, 0:27] = waveletDecompMatrix
    
    
    # Shannon Entropy Analysis
    shannonEntropyMatrix = getShannonEntropy(data,1)
    print('s', shannonEntropyMatrix.shape)
    featureMatrix[27, 0:27] = shannonEntropyMatrix
    # for j in np.arange(1,shannonEntropyMatrix.shape[1-1]+1).reshape(-1):
    #     k = k + 1
    #     featureMatrix[k,:] = shannonEntropyMatrix(j,:)
    
    return featureMatrix
    
    # return featureMatrix

# extractFeatures("/Users/melodyzhao/Desktop/Python/ENPH459-concussion-svm/New_Code/Concussed/AG/matlab.mat")

    # field = fieldnames(eegStruct)
    # fieldName = field[0]
    # eegMatrix = getattr(eegStruct,(fieldName))
    # rows,cols = eegMatrix.shape
    # #featureMatrix = zeros(30,cols);
    
    # # Power Spectral Analysis
    # powerSpectralMatrix = powerSpectral(eegMatrix)
    # for k in np.arange(1,powerSpectralMatrix.shape[1-1]+1).reshape(-1):
    #     featureMatrix[k,:] = powerSpectralMatrix(k,:)

