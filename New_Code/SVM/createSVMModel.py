    
def createSVMModel(dataset = None,labels = None,svmMethod = None): 
    #CREATESVMMODEL_ Creates cross-validated SVM model based on loaded dataset, labels and
#method (either "fitcsvm" or "fitclinear")
#   Inputs: dataset - feature extraction matrix
#           labels - concussed/non-concussed label for each patient
#           method - which Matlab SVM model method to use. This is between
#           "fitcsvm" which is recommended for low-medium dimensional
#           parameters, and "fitclinear" which is recommended for high
#           dimensional parameters
    
    if svmMethod == 'fitcsvm':
        kFoldSVMModel = fitcsvm(dataset,labels,'Crossval','on','Standardize',True,'KernelFunction','RBF','KernelScale','auto')
        holdOutSVMModel = fitcsvm(dataset,labels,'Crossval','on','Holdout',0.1,'Standardize',True,'KernelFunction','RBF','KernelScale','auto')
        leaveOutSVMModel = fitcsvm(dataset,labels,'Crossval','on','Leaveout','on','Standardize',True,'KernelFunction','RBF','KernelScale','auto')
    else:
        if svmMethod == 'fitclinear':
            kFoldSVMModel = fitclinear(dataset,labels)
    
    holdOutSVMModel = holdOutSVMModel.Trained[0]
    return kFoldSVMModel,holdOutSVMModel,leaveOutSVMModel
    
    return kFoldSVMModel,holdOutSVMModel,leaveOutSVMModel