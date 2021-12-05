#Requires: Folders names "Concussed" and "Controls" in the given directory
#corresponding to EEG Data for conccussed patients and control patients
#respectively.

import numpy as np
    
def featureExtraction(directory = None): 
    #directory = 'C:\Users\Dylan\Desktop\New folder\Capstone Training Data';
    
    #Extract .mat files corresponding to concussed patients
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
        featureMatrix[k,:] = np.transpose(features)
    
    #Extract .mat files corresponding to non-concussed patients
    controlDirectory = dir(strcat(directory,'\Controls\*\*.mat'))
    controlFilenames = np.array([controlDirectory.name])
    controlFoldernames = np.array([controlDirectory.folder])
    #Append control features to control matrix
    for j in np.arange(1,len(controlFilenames)+1).reshape(-1):
        k = k + 1
        features = extractFeatures(strcat(controlFoldernames[j],'\',controlFilenames[j]))
        featureMatrix[k,:] = np.transpose(features)
    
    labels = np.ones((len(concussedFilenames) + len(controlFilenames),1))
    labels[np.arange[len[concussedFilenames] + 1,len[labels]+1],1] = 0
    return featureMatrix,labels
    
    return featureMatrix,labels