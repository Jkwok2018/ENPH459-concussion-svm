from scipy.io import savemat

# from FeatureExtraction.extractFeatures import extractFeatures
from FeatureExtraction.featureExtraction import featureExtraction

def createAndSaveFeatureMatrix():
    directory = '/Users/melodyzhao/Desktop/Python/ENPH459-concussion-svm/New_Code'
    featureMatrix, labels = featureExtraction(directory)
    blindFeatureMatrix = {"data":featureMatrix, "labels":labels}
    savemat('FeatureMatrix.mat', blindFeatureMatrix)

createAndSaveFeatureMatrix()