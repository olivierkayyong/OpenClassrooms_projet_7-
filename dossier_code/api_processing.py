import pandas as pd
import numpy as np
import pickle
from lime.lime_tabular import LimeTabularExplainer
from operator import itemgetter
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import confusion_matrix

# Loading data & Model
data = pd.read_csv("x_data.csv")
df_2 = pd.read_csv("x_data_2.csv")
y_test = pd.read_csv("y_data.csv")
pickle_in = open("lgbm.pkl", "rb")
classifier = pickle.load(pickle_in)

# Scaled data
data_scaled = classifier[1].transform(data)
df_scaled = pd.DataFrame(data_scaled ,columns=data.columns, index=data.index)

# Predict proba score
def predict_proba(id_client):
    """ Calculation of predict_proba score . 
    Args:
        id_client: customer identity   
    Returns:
        score: predict_proba value
    """
    score = classifier[-1].predict_proba(np.array(df_scaled.loc[id_client]).reshape(1, -1)).flatten()
    return score

# Customer's prediction
def predict(id_client):
    score = predict_proba(id_client)
    prediction = score[0]
    if (prediction>0.5):
       prediction = "The customer is solvent"
    else:
        prediction = "The customer is insolvent"      
    return prediction

# Local feature importance
def feature_importance_loc(id_client):
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
    return indices,values

# Similary customers
nbrs = NearestNeighbors(n_neighbors=20, algorithm='ball_tree').fit(df_scaled)
def nearest_neighbors(id_client):
    """ Determination of the similary customers with kneighbors. 
    Args:
        id_client: customer identity   
    Returns:
        indices: similary customers index
    """
    indices_similary_clients = nbrs.kneighbors(np.array(df_scaled.loc[id_client]).reshape(1, -1))[1].flatten()
    dff = data.iloc[indices_similary_clients]
    return dff.to_dict('records')

# Global feature importance
def features_imp_globale():
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
    return indexes, features

# Confusion matrix
def conf_mat():
    """ Calculation of the confusion matrix."""
    cf_matrix = confusion_matrix(y_test, classifier.predict(data))
    return cf_matrix

# Test data
def test_data():
    """Loading test data"""
    x = data.to_string(header=False,
                  index=False,
                  index_names=False).split('\n')
    vals = [','.join(ele.split()) for ele in x]
    return vals

# Test data_2
def test_data_2():
    """Loading test data"""
    x = df_2.to_string(header=False,
                  index=False,
                  index_names=False).split('\n')
    vals = [','.join(ele.split()) for ele in x]
    return vals