from sklearn.model_selection import cross_validate
from sklearn import metrics
import math
import numpy as np

def cv_score(model, X, y, cv):
    cv_results = cross_validate(
        model,
        X,
        y,
        cv=cv,
        scoring=["neg_mean_absolute_error", "neg_root_mean_squared_error", "r2"],
        return_train_score=False,
    )
    mae = -cv_results["test_neg_mean_absolute_error"]
    rmse = -cv_results["test_neg_root_mean_squared_error"]
    r2 = cv_results["test_r2"]
    
    return (
        f"Mean Absolute Error:     {mae.mean():.3f} +/- {mae.std():.3f}\n"
        f"R2:     {r2.mean():.3f} +/- {r2.std():.3f}\n"
        f"Root Mean Squared Error: {rmse.mean():.3f} +/- {rmse.std():.3f}"
    )




def score_simple(y_true, y_pred):
    '''Compute scores without using cross_validate()
    
    Usefull to evaluate recursive prediction, or any other prediction made without cross_validate 
    '''
    
    mae = metrics.mean_absolute_error(y_true, y_pred)    
    r2 = metrics.r2_score(y_true, y_pred)
    rmse = math.sqrt(metrics.mean_squared_error(y_true, y_pred))
    
    
    return (
        f"Mean Absolute Error:     {mae:.3f}\n"
        f"R2:     {r2:.3f}\n"
        f"Root Mean Squared Error: {rmse:.3f}"
    )
    
#     return mae, r2, rmse