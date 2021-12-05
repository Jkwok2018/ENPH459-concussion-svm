import numpy as np
    
def returnLeaveOutResult(leaveOutObject = None,predictionData = None): 
    #RETURNLEAVEOUTRESULTS Returns averaged predictions of kfold cross-validated
#SVM models
#   Inputs: leaveOutObject: MATLAB SVM object containing all the trained SVM
#           models
#           predictionData: data used for predictions
    
    #   Outputs: prediction: result taken as the majority prediction of all the
#            SVM models
    
    rowSize,colSize = leaveOutObject.Trained.shape
    dataRow,dataCol = predictionData.shape
    predictions = np.zeros((dataRow,rowSize))
    prediction = np.zeros((dataRow,1))
    for i in np.arange(1,rowSize+1).reshape(-1):
        predictions[:,i] = predict(leaveOutObject.Trained[i],predictionData)
    
    for i in np.arange(1,dataRow+1).reshape(-1):
        prediction[i,1] = mode(predictions(i,:))
    
    return prediction
    
    return prediction