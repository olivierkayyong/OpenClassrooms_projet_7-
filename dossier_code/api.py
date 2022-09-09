# Import library
import uvicorn
import pandas as pd
from fastapi import FastAPI
from api_processing import *


# Create the app object
app = FastAPI()

# Index route
@app.get('/')
def welcome():
    return {'message':"Hello welcome to the customer prediction"}

# Customer prediction
@app.get('/predict')
def customer_predict(id_client:int):
    """ Customer creditworthiness prediction. 
    Args:
        id_client: customer identity   
    Returns:
        Prediction: the customer prediction
        Customer score: predict_proba value
    """
    prediction = predict(id_client)      
    return {'Prediction': prediction}

# Customer predict_proba_score
@app.get('/get-Score')
def score(id_client:int):
    score = predict_proba(id_client) 
    return {'Predict_proba score': score.tolist() }

# Dataframe indexes
@app.get('/get-Indexes')
def indexes():
    """ Dataframe index"""
    return data.index.tolist() 

# Dataframe columns
@app.get('/get-Columns')
def columns():
    """ Dataframe columns"""
    return data.columns.tolist() 
# Dataframe columns
@app.get('/get-Columns2')
def columns():
    """ Dataframe columns"""
    return df_2.columns.tolist() 

# Interpretability of the model:Local feature importance
@app.get('/get-LocalFeatureImportance')
def local_feature_importance(id_client:int):
    """ Determination of local feature importance. 
    Args:
        id_client: customer identity   
    Returns:
        indices: local feature importance index
        values: lime explainer values
    """
    res = feature_importance_loc(id_client)
    return {'indices': res[0],
            "values": res[1]
            }

# Similary customers
@app.get('/get-CustomersNeighbors')
def similary_customers(id_client:int):
    """ Determination of the similary customers with kneighbors. 
    Args:
        id_client: customer identity   
    Returns:
        indices: similary customers index
    """
    res = nearest_neighbors(id_client)
    return res

# Global feature importance
@app.get('/get-GlobalFeatureImportance')
def features_globale():
    """ Determination of global feature importance.   
    Returns:
        indices: index of global feature importance
        features: list of  global feature importance
    """
    res = features_imp_globale()
    return {'indices': res[0].tolist(),
            "features": res[1]
            }

# Confusion matrix
@app.get('/get-ConfusionMatrix')
def confusion_mat():
    """ Calculation of the confusion matrix."""
    cf_matrix = conf_mat()
    return cf_matrix.tolist()

# Test data
@app.get('/get-TestData')
def data_test():
    """Loading test data"""
    return test_data()

# Test data_2
@app.get('/get-TestData2')
def data_test_2():
    """Loading test data"""
    return test_data_2()
