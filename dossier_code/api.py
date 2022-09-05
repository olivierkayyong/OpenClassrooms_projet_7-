# Import library
import uvicorn
from fastapi import FastAPI
import pandas as pd
import numpy as np
import pickle
from lime.lime_tabular import LimeTabularExplainer
from operator import itemgetter
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import confusion_matrix


# Loading data & Model
data = pd.read_csv("data_test.csv")
y_test = pd.read_csv("y_test.csv")
pickle_in = open("lgbm.pkl", "rb")
classifier = pickle.load(pickle_in)

# Scaled data
data_scaled = classifier[1].transform(data)
df_scaled = pd.DataFrame(data_scaled ,columns=data.columns, index=data.index)


# Create the app object
app = FastAPI()

# Index route
@app.get('/')
def index():
    return {'message':"Hello welcome to the customer prediction"}

# Customer prediction
@app.get('/{predict}')
def predict_creditworthiness(id_client:int):
    """ Calculation of the customer creditworthiness . 
    Args:
        id_client: customer identity   
    Returns:
        Prediction: the customer prediction
        Customer score: predict_proba value
    """
    score = classifier[-1].predict_proba(np.array(df_scaled.loc[id_client]).reshape(1, -1)).flatten()
    prediction = score[0]
    if (prediction>0.5):
       prediction = "The customer is solvent"
    else:
        prediction = "The customer is insolvent"
        
    return {'Prediction': prediction,
            "Customer score":score.tolist()
            }
             
# Dataframe indexes
@app.get('/{predict}/{index}')
def indexes():
    """ Dataframe index"""
    return data.index.tolist() 

# Dataframe columns
@app.get('/{predict}/{index}/{columns}')
def columns():
    """ Dataframe columns"""
    return data.columns.tolist() 

# Interpretability of the model:Local feature importance
@app.get('/{predict}/{index}/{columns}/{local feature importance}')
def local_feature_importance(id_client:int):
    """ Determination of local feature importance. 
    Args:
        id_client: customer identity   
    Returns:
        indices: local feature importance index
        values: lime explainer values
    """
    lime1 = LimeTabularExplainer(data_scaled,
                             feature_names=data.columns,
                             class_names=["Solvable", "Non Solvable"],
                             discretize_continuous=False)
    exp = lime1.explain_instance(df_scaled.loc[id_client],
                                 classifier[-1].predict_proba,
                                 num_samples=100)
    indices, values = [], []
    for ind, val in sorted(exp.as_list(), key=itemgetter(1)):
        indices.append(ind)
        values.append(val)
    return {'indices': indices,
            "values": values
            }

# Similary customers
nbrs = NearestNeighbors(n_neighbors=20, algorithm='ball_tree').fit(df_scaled)
@app.get('/{predict}/{index}/{columns}/{local feature importance}/{customers neighbors}')
def similary_customers(id_client:int):
    """ Determination of the similary customers with kneighbors. 
    Args:
        id_client: customer identity   
    Returns:
        indices: similary customers index
    """
    indices = nbrs.kneighbors(np.array(df_scaled.loc[id_client]).reshape(1, -1))[1].flatten()
    return indices.tolist()

# Global feature importance
@app.get('/{predict}/{index}/{columns}/{local feature importance}/{customers neighbors}/{global feature importance}')
def features_globale():
    """ Determination of global feature importance.   
    Returns:
        indices: index of global feature importance
        features: list of  global feature importance
    """
    indices = np.argsort(classifier[-1].feature_importances_)[::-1]
    features = []
    for i in range(20):
        features.append(data.columns[indices[i]])
    indexes = classifier[-1].feature_importances_[indices[range(20)]]
    return {'indices': indexes.tolist(),
            "features":features
            }

# Confusion matrix
@app.get('/{predict}/{index}/{columns}/{local feature importance}/{customers neighbors}/{global feature importance}/{confusion matrix}')
def confusion_mat():
    """ Calculation of the confusion matrix."""
    cf_matrix = confusion_matrix(y_test, classifier.predict(data))
    return cf_matrix.tolist()


# uvicorn api:app --reload