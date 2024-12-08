import dash
from dash import dcc, html
import plotly.express as px
import pandas as pd

# Sample data for visualization
ratings = pd.read_csv('C:/Users/rohit/Downloads/ml-latest-small/ml-latest-small/ratings.csv')
sample_ratings = ratings.sample(n=1000)

# Plotly express figure
fig = px.histogram(sample_ratings, x='rating', nbins=10, title='Distribution of Ratings')

# Create Dash app
app = dash.Dash(__name__)

# Layout of the app
app.layout = html.Div(children=[
    html.H1(children='Interactive Visualization Dashboard'),
    
    dcc.Graph(
        id='example-graph',
        figure=fig
    ),
])

if __name__ == '__main__':
    app.run_server(debug=True)
