import dash
import pandas as pd
from dash import html, dcc
import plotly.express as px
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate

df = pd.read_csv('output/visualization_dash.csv')
app = dash.Dash(__name__, external_stylesheets=['https://codepen.io/chriddyp/pen/bWLwgP.css'])

def serve_layout():
    return html.Div([
        html.H3("Play Store Insights", style={'textAlign': 'center'}),
        dcc.Tabs(id="tabs", value='tab-1', children=[
            dcc.Tab(label='Cube', value='tab-1', children=[
                html.Div(id='output-div'),
                dcc.Graph(id='main-graph'),
                html.H3("Select Category:", style={'marginTop': '20px'}),
                dcc.Dropdown(
                    id='category-dropdown',
                    options=[{'label': cat, 'value': cat} for cat in df['Category'].unique()],
                    value=df['Category'].unique(),
                    multi=True
                ),
                html.Br(),
                html.H3("Choose Content Ratings:"),
                dcc.Checklist(
                    id='content-rating-checklist',
                    options=[{'label': rating, 'value': rating} for rating in df['Content Rating'].unique()],
                    value=[df['Content Rating'].unique()[0]],
                    inline=True
                ),
                html.Br(),
                html.H3("App Type (Free/Paid):"),
                dcc.RadioItems(
                    id='free-paid-radio',
                    options=[
                        {'label': 'Free', 'value': True},
                        {'label': 'Paid', 'value': False}
                    ],
                    value=True,
                    inline=True
                ),
                html.Br(),
                html.H3("Filter by App Age:"),
                dcc.RangeSlider(
                    id='app-age-slider',
                    min=df['App Age'].min(),
                    max=df['App Age'].max(),
                    value=[df['App Age'].min(), df['App Age'].max()],
                    marks={i: str(i) for i in range(int(df['App Age'].min()), int(df['App Age'].max())+1, 100)}
                ),
                html.Br(),
                html.H3("Select App Size in MB:"),
                dcc.Slider(
                    id='size-slider',
                    min=df['Size in MB'].min(),
                    max=df['Size in MB'].max(),
                    value=df['Size in MB'].max(),
                    marks={i: str(i) for i in range(int(df['Size in MB'].min()), int(df['Size in MB'].max())+1, 10)},
                    step=0.1
                ),
                html.Br(),
            ]),
            dcc.Tab(label='Resources', value='tab-2', children=[
                html.Button("Download Dataset", id="download-button"),
                dcc.Download(id="download-dataset")
            ]),
        ]),
    ], style={'padding': '20px'})

app.layout = serve_layout


@app.callback(
    Output('main-graph', 'figure'),
    Output('output-div', 'children'),
    Input('category-dropdown', 'value'),
    Input('content-rating-checklist', 'value'),
    Input('free-paid-radio', 'value'),
    Input('app-age-slider', 'value'),
    Input('size-slider', 'value')
)
def update_graph(selected_categories, selected_content_rating, free_paid, app_age_range, size):
    filtered_df = df[df['Category'].isin(selected_categories) &
                     (df['Content Rating'].isin(selected_content_rating)) &
                     (df['Free'] == free_paid) &
                     (df['App Age'] >= app_age_range[0]) &
                     (df['App Age'] <= app_age_range[1]) &
                     (df['Size in MB'] <= size)]

    # Generating the graph
    fig = px.scatter_3d(filtered_df, x='Rating', y='Rating Count', z='Installs', color='Category', size = 'Installs')
    fig.update_layout(height=1000)
    output_message = f"Displaying data for {len(filtered_df)} apps"
    return fig, output_message


@app.callback(
    Output("download-dataset", "data"),
    Input("download-button", "n_clicks"),
    prevent_initial_call=True
)
def download_data(n_clicks):
    if n_clicks is None:
        raise PreventUpdate
    return dcc.send_data_frame(df.to_csv, "visualization_dash.csv")


if __name__ == '__main__':
    app.run_server(debug=True)

