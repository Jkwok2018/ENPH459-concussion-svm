#Requires: Folders names "Concussed" and "Controls" in the given directory
#corresponding to EEG Data for conccussed patients and control patients
#respectively.

import numpy as np
    
def blindTestFeatureExtraction(directory = None): 
    #Extract .mat files corresponding to concussed patients
    concussedDirectory = dir(strcat(directory,'\Blind\*.mat'))
    concussedFilenames = np.array([concussedDirectory.name])
    concussedFoldernames = np.array([concussedDirectory.folder])
    #Append concussed features to feature matrix
    for k in np.arange(1,len(concussedFilenames)+1).reshape(-1):
        #eegMat = load(strcat(concussedFoldernames{k},'\',concussedFilenames{k}));
        features = extractFeatures(strcat(concussedFoldernames[k],'\',concussedFilenames[k]))
        featureMatrix[k,:] = np.transpose(features)
    
    return featureMatrix
    
    return featureMatrix