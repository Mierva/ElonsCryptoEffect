from dash import Dash, dcc, html
from dash.dependencies import Output, Input
import plotly.express as px
import pandas as pd


def load_data():
    A_group = pd.read_csv('Models/Tuned_in_csv/with_tweets_models (1).csv', index_col=0)
    B_group = pd.read_csv('Models/Tuned_in_csv/without_tweets_models (1).csv', index_col=0)
    df = pd.DataFrame(A_group['mean_test_score']).join(B_group['mean_test_score'], lsuffix='_A')
    df = df.rename(columns={'mean_test_score_A':'A_scores',
                            'mean_test_score':'B_scores'})
    
    return df

def run_app(debug=True):
    MIN,MAX = 5,50
    df = load_data()
    
    app = Dash(__name__)
    app.layout = html.Div([dcc.Graph(id='histogram'),
                        dcc.Slider(id='bins-slider',min=MIN, max=MAX, step=1, value=10,
                                    tooltip={"placement": "bottom", "always_visible": True})                                 
                        ])

    @app.callback(Output('histogram', 'figure'),
                [Input('bins-slider', 'value')])
    def update_histogram(bins):
        fig = px.histogram(df, 
                        nbins=int(bins), 
                        barmode="stack",
                        labels={'value':'mean test score'},
                        title=f'bins = {bins}',
                        width=950,
                        height=500)
        
        fig.update_traces(marker_line_color='black',
                        marker_line_width=0.5)
        return fig

    app.run(debug=debug)
    
        
if __name__ =='__main__':
    app = run_app()    
    