# Importing library
from dash import Dash, dcc, html
import pandas as pd
import numpy as np
import dash_table
from dash.dependencies import Input, Output
import requests
import plotly.graph_objs as go
from csv import reader


# Create the app object
app = Dash(__name__)
server=app.server

# Api url
API_URL = 'https://kay10api.herokuapp.com/'

# Dataframe index
indexes = requests.get(url=API_URL+ "get-Indexes").json()

# Dataframe columns
columns = requests.get(url=API_URL+ "get-Columns").json()
columns_2 = requests.get(url=API_URL+ "get-Columns2").json()

# Loading test data
vals = requests.get(url = API_URL + "get-TestData").json()
df = pd.DataFrame( list(reader(vals)),columns=columns)
vals_2 = requests.get(url = API_URL + "get-TestData2").json()
df_2 = pd.DataFrame( list(reader(vals_2)),columns=columns_2)

# Features importances globale
def plot_feature_importances():
    """  Plot importances returned by a model."""
    r = requests.get(url=API_URL+ "get-GlobalFeatureImportance").json()
    indices = r['indices']
    values = r['features']
    data = pd.DataFrame(values, columns=["values"], index=indices)
    return {
        'data': [go.Bar(
                    x=data.index,
                    y=data["values"],
                    orientation='h',
        )],
        
        'layout': go.Layout(
                            margin={'l': 300, 'b': 50, 't': 30, 'r': 30},
                            height=700,
                            width=1200,
                           )
    }

# Confusion matrix
def plot_confusion_matrix():
    """  Plot of confusion matrix."""
    r = requests.get(url=API_URL+ "get-ConfusionMatrix").json()
    conf_mx = np.array(r)
    labels = ["Solvent", "Insolvent"]
    
    annotations = go.Annotations()
    for n in range(conf_mx.shape[0]):
        for m in range(conf_mx.shape[1]):
            annotations.append(go.Annotation(text=str(conf_mx[n][m]), x=labels[m], y=labels[n],
                                             showarrow=False))

    trace = go.Heatmap(x=labels,
                       y=labels,
                       z=conf_mx,
                       colorscale='twilight',
                       showscale=False)

    fig = go.Figure(data=go.Data([trace]))
    fig['layout'].update(
        annotations=annotations,
        xaxis= dict(title='Predicted classes'), 
        yaxis=dict(title='Actual classes', dtick=1),
        margin={'b': 30, 'r': 20, 't': 10},
        width=700,
        height=500,
        autosize=False
    )
    
    return fig 


app.layout = html.Div([
    
    dcc.Tabs([
        # First tab: Solvency Client
        dcc.Tab(label='Customer Creditworthiness', children=[
            # Selecting the customer identity from a drop-down list
            html.Div([
                html.H3("Id Client"),
                dcc.Dropdown(
                id='id-client',
                options=[{'label': i, 'value': i} for i in indexes],
                value=indexes[0]
                ),
            ]),
            html.Div([
                # Probability of a customer's solvency as a pie plot
                html.Div([
                    html.H3("Customer Solvency Probability"),
                    dcc.Graph(id='proba',
                              figure={},
                              style={"height": 500,
                                     "width": 500}
                             ),
                ], className='six columns'),
                # Interpretability of the model: local feature importance
                html.Div([
                    html.H3("Local Feature Importance"), 
                    dcc.Graph(id='graph',
                              figure={},
                              style={"height":500,
                                     "width":800}
                             ),       
                ], className='six columns'),        
            ], className="row"),
            # Similar customers
            html.Div([
                html.H3("Similar customers"),
                dash_table.DataTable(
                    id='table',
                    data = df.to_dict('records'),
                    columns=[
                      {"name": i, "id": i} for i in df.columns[:10]
                   ],
                 
               ), 
           ]),
               
       ]),
       # Second tab : Model Performance
       dcc.Tab(label="Model Performance", children=[
           html.Div([
               # Plot of confusion matrix
               html.Div([
                   html.H3("Confusion Matrix"),
                   dcc.Graph(id='cf_mat',
                             figure= plot_confusion_matrix(),
                            ),
               ], className='six columns'),
               # Plot of global feature importance
               html.Div([
                   html.H3("Global Feature Importance"), 
                   dcc.Graph(id='graph_feature',
                             figure=plot_feature_importances()),   
                ], className="six columns"),
            ]),
        ]),
    
        # Third tab: Data exploration
        dcc.Tab(label='Data Exploration', children=[
           # Univariate analysis
           html.Div([html.H3("Univariate analysis"),
                html.Div([
                    dcc.Dropdown(
                        id='xis-column',
                        options=[{'label': i, 'value': i} for i in columns],
                        value='CODE_GENDER'
                    ),
                    dcc.RadioItems(
                        id='xis-type',
                        options=[{'label': i, 'value': i} for i in ['Linear', 'Log']],
                        value='Linear',
                        labelStyle={'display': 'inline-block'}
                    )
                ],
                style={'width': '48%', 'display': 'inline-block'}),

                
            ]),

            dcc.Graph(id='indicator'),     
          # Bivariate analysis
          html.Div([html.H3("Bivariate analysis"),
                html.Div([
                    dcc.Dropdown(
                        id='xaxis-column',
                        options=[{'label': i, 'value': i} for i in columns],
                        value='AMT_CREDIT'
                    ),
                    dcc.RadioItems(
                        id='xaxis-type',
                        options=[{'label': i, 'value': i} for i in ['Linear', 'Log']],
                        value='Linear',
                        labelStyle={'display': 'inline-block'}
                    )
                ],
                style={'width': '48%', 'display': 'inline-block'}),

                html.Div([
                    dcc.Dropdown(
                        id='yaxis-column',
                        options=[{'label': i, 'value': i} for i in columns],
                        value='AMT_ANNUITY'
                    ),
                    dcc.RadioItems(
                        id='yaxis-type',
                        options=[{'label': i, 'value': i} for i in ['Linear', 'Log']],
                        value='Linear',
                        labelStyle={'display': 'inline-block'}
                    )
                ],style={'width': '48%', 'float': 'right', 'display': 'inline-block'})
            ]),

            dcc.Graph(id='indicator-graphic'),
            
]),

    ]),
])

    
# Updating the solvency pieplot of the customer
@app.callback(
    Output('proba', 'figure'),
    [Input('id-client', 'value')])
def customer_solvency(id_client):
    """Plot probability of a customer's solvency as a pie plot"""
    
    r = requests.get(url=API_URL+ "get-Score", params={"id_client": id_client})
    json_data = r.json()['Predict_proba score']
    values = np.array(json_data)     
    return {
        'data': [go.Pie(labels=["Solvent", "Insolvent"],
                        values=values,
                        marker_colors=["#2ecc71", "#e74c3c"],
                        hole=.5
                       )],
        'layout': go.Layout(margin=dict(b=100)
                           )
    }
    del values

# Updating customer's local importance features 
@app.callback(
    Output('graph', 'figure'),
    [Input('id-client', 'value'),
    ])
def local_feature_importance(id_client) :
    """ Plot customer's local importance features"""
    r = requests.get(url=API_URL+ "get-LocalFeatureImportance", params={"id_client": id_client}).json()
    values = r['values']
    indices = r['indices']
    data = pd.DataFrame(values, columns=["values"], index=indices)
    data["positive"] = data["values"]>0
    del indices, values
    return {
        
        'data': [go.Bar(
                    x=data["values"],
                    y=data.index,
                    orientation='h',
                    marker_color=list(data.positive.map({True: '#e74c3c', False: '#2ecc71'}).values)
        )],
        
        'layout': go.Layout(
                            margin=dict(l=300, r=0, t=30, b=100)
                           )
    }

# Updating customer's similary neighbors
@app.callback(
        Output('table','data'),
        [Input('id-client', 'value')
         ])
def similary_customers(id_client):
    """ Plot customer's similary neighbors """
    r = requests.get(url=API_URL+ "get-CustomersNeighbors", params={"id_client": id_client}).json()
    return r

# Updating bivariate analysis
@app.callback(
    Output('indicator-graphic', 'figure'),
    [Input('xaxis-column', 'value'),
     Input('yaxis-column', 'value'),
     Input('xaxis-type', 'value'),
     Input('yaxis-type', 'value')])
def bivariate_analysis(xaxis_column_name, yaxis_column_name,
                 xaxis_type, yaxis_type):
    """ plot bivariate analysis"""
     
    traces = []
    solvable_labels = ["Solvent", "Insolvent"]
    for i, target in enumerate(df_2.TARGET.unique()):
        filtered_df = df_2[df_2['TARGET'] == target].reset_index()
        traces.append(dict(
            x=filtered_df[xaxis_column_name],
            y=filtered_df[yaxis_column_name],
            mode='markers',
            opacity=0.7,
            marker={
                'size': 5,
                'line': {'width': 0.15, 'color': 'white'}
            },
            name=solvable_labels[i]
        ))   
        
    return {
        'data': traces,
        'layout': dict(
            xaxis={
                'title': xaxis_column_name,
                'type': 'linear' if xaxis_type == 'Linear' else 'log'
            },
            yaxis={
                'title': yaxis_column_name,
                'type': 'linear' if yaxis_type == 'Linear' else 'log'
            },
            margin={'l': 60, 'b': 40, 't': 10, 'r': 0},
            hovermode='closest'
        )
    }

# # Updating univariate analysis 
@app.callback(
    Output('indicator', 'figure'),
    [Input('xis-column', 'value'),
     Input('xis-type', 'value')
   ])
def univariate_analysis(xis_column_name, 
                 xis_type):
    """ plot univariate analysis"""
       
    traces = []
    solvable_labels = ["Solvent", "Insolvent"]
    for i, target in enumerate(df_2.TARGET.unique()):
        filtered_df = df_2[df_2['TARGET'] == target].reset_index()
        traces.append(dict(
            x=filtered_df[xis_column_name],
            mode='markers',
            opacity=0.7,
            marker={
                'size': 5,
                'line': {'width': 0.15, 'color': 'white'}
            },
            name=solvable_labels[i]
        ))   
        
    return {
        'data': traces,
        'layout': dict(
            xis={
                'title': xis_column_name,
                'type': 'linear' if xis_type == 'Linear' else 'log'
            },
            
            margin={'l': 40, 'b': 40, 't': 10, 'r': 0},
            hovermode='closest'
        )
    }


if __name__ == '__main__':
    app.run_server(debug=True)
