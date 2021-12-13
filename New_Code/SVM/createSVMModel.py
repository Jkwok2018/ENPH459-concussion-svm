    
from sklearn import svm
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def createSVMModel(dataset = None,labels = None,svmMethod = None): 
    #CREATESVMMODEL_ Creates cross-validated SVM model based on loaded dataset, labels and
#method (either "fitcsvm" or "fitclinear")
#   Inputs: dataset - feature extraction matrix
#           labels - concussed/non-concussed label for each patient
#           method - which Matlab SVM model method to use. This is between
#           "fitcsvm" which is recommended for low-medium dimensional
#           parameters, and "fitclinear" which is recommended for high
#           dimensional parameters
    dataset = np.asarray(dataset)
    labels = np.asarray(labels[0])

    dataset, labels = shuffle(dataset, labels)
    print(labels)
    kf = KFold(n_splits= 5, shuffle=True)
    kf.get_n_splits(dataset)
    print(kf)
    accuracy = []
    for train_index, test_index in kf.split(dataset):
        print(train_index)
        X_train, X_test = dataset[train_index], dataset[test_index]
        y_train, y_test = labels[train_index], labels[test_index]
        if svmMethod == 'fitcsvm':
            # kFoldSVMModel = svm.SVC(kernel = 'rbf', gamma='auto').fit(X_train, y_train)
            kFoldSVMModel = make_pipeline(StandardScaler(), svm.SVC(kernel = 'rbf', gamma='auto')).fit(X_train, y_train)
        else:
            kFoldSVMModel = make_pipeline(StandardScaler(), svm.LinearSVC()).fit(X_train, y_train)

        predictions = kFoldSVMModel.predict(X_test)
        print("predictions", predictions)
        accuracy.append(accuracy_score(predictions, y_test))
    
    accuracy_average = sum(accuracy)/len(accuracy)
    
    print("accuracy for each model: ", accuracy)
    print(f"average accuracy: {accuracy_average}")
    return accuracy_average

    # if svmMethod == 'fitcsvm':
    #     # kFoldSVMModel = fitcsvm(dataset,labels,'Crossval'='on','Standardize' = True,'KernelFunction'='RBF','KernelScale' = 'auto')
    #     # holdOutSVMModel = fitcsvm(dataset,labels,'Crossval','on','Holdout',0.1,'Standardize',True,'KernelFunction','RBF','KernelScale','auto')
    #     # leaveOutSVMModel = fitcsvm(dataset,labels,'Crossval','on','Leaveout','on','Standardize',True,'KernelFunction','RBF','KernelScale','auto')
    #     # clf = svm.SVC(kernel='rbf')
    #     # clf.fit(dataset, labels)
        

    #     kFoldSVMModel = svm.SVC(kernel = 'rbf')
    # else:
    #     if svmMethod == 'fitclinear':
    #         kFoldSVMModel = svm.LinearSVC()
    # kFoldSVMModel.fit(dataset, labels)
    
    # holdOutSVMModel = holdOutSVMModel.Trained[0]
    # return kFoldSVMModel,holdOutSVMModel,leaveOutSVMModel
    # return kFoldSVMModel
    