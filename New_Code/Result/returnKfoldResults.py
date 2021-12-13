import numpy as np
    
# TODO: how to get trained from kFoldObject
def returnKfoldResults(kFoldObject = None,predictionData = None): 
    #RETURNKFOLDRESULTS Returns averaged predictions of kfold cross-validated
#SVM models
#   Inputs: kFoldObject: MATLAB SVM object containing all the trained SVM
#           models
#           predictionData: data used for predictions
    
    #   Outputs: prediction: result taken as the majority prediction of all the
#            SVM models
    
    # rowSize,colSize = kFoldObject.Trained.shape
    # dataRow,dataCol = predictionData.shape
    # predictions = np.zeros((dataRow,rowSize))
    # prediction = np.zeros((dataRow,1))
    # for i in np.arange(1,rowSize+1).reshape(-1):
    #     predictions[:,i] = predict(kFoldObject.Trained[i],predictionData)
    
    # for i in np.arange(1,dataRow+1).reshape(-1):
    #     prediction[i,1] = mode(predictions(i,:))

    prediction = kFoldObject.fit(predictionData['data'], predictionData['labels'])
    
    return prediction
    
