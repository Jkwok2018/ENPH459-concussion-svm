#Requires: Folders names "Concussed" and "Controls" in the given directory
#corresponding to EEG Data for conccussed patients and control patients
#respectively.
from .extractFeatures import extractFeatures
import numpy as np
import glob
    
def featureExtraction(directory = None): 
    #directory = 'C:\Users\Dylan\Desktop\New folder\Capstone Training Data';
    
    #Extract .mat files corresponding to concussed patients
    
    concussedDirectory = sorted(glob.glob(directory+ "/Concussed/*/*.mat"))
    controlDirectory = sorted(glob.glob(directory+ "/Controls/*/*.mat"))

    featureMatrix = np.zeros((len(concussedDirectory)+ len(controlDirectory), 756))
    
    for i, file_name in enumerate(concussedDirectory):
        features = extractFeatures(file_name)
        features = np.matrix(features).flatten('F')
        featureMatrix[i] = features
  
    #Extract .mat files corresponding to non-concussed patients
    for j, file_name in enumerate(controlDirectory):
        features = extractFeatures(file_name)
        features = np.matrix(features).flatten('F')
        featureMatrix[j+ len(concussedDirectory)] = features

  
    labels = np.append(np.ones((len(concussedDirectory))), np.zeros((len(concussedDirectory))))
    
    return featureMatrix,labels


    

    

    """
    concussedDirectory = dir(strcat(directory,'\Concussed\*\*.mat'))
    concussedFilenames = np.array([concussedDirectory.name])
    concussedFoldernames = np.array([concussedDirectory.folder])
    #Extract Folder names from Concussed folder
#dirFlags = [concussedDirectory.isdir];
#subfolders = concussedDirectory(dirFlags);
    
    #Append concussed features to feature matrix
    for k in np.arange(1,len(concussedFilenames)+1).reshape(-1):
        #eegMat = load(strcat(concussedFoldernames{k},'\',concussedFilenames{k}));
        features = extractFeatures(strcat(concussedFoldernames[k],'\',concussedFilenames[k]))
        featureMatrix[k,:] = np.transpose(features)"""

# featureExtraction("/Users/melodyzhao/Desktop/Python/ENPH459-concussion-svm/New_Code")
