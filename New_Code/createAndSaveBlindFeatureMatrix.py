from scipy.io import savemat

def createAndSaveBlindFeatureMatrix():
    directory = '/Users/melodyzhao/Desktop/Python/ENPH459-concussion-svm/New_Code'
    blindFeatureMatrix = blindTestFeatureExtraction(directory)
    savemat('blindFeatureMatrix.mat',blindFeatureMatrix)