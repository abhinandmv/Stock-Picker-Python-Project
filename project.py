import datetime
import yfinance as yf
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objs as go
import dash_bootstrap_components as dbc
import numpy as np
from threading import Timer
import webbrowser

# Function to predict future prices (dummy prediction logic)
def predict_future_prices(stock_data, days=5):
    # Simple prediction: assume the last closing price continues for the next 'days' days
    last_close = stock_data['Close'].iloc[-1]
    return [last_close * (1 + np.random.uniform(-0.05, 0.05)) for _ in range(days)]

# Initializing the Dash app with Bootstrap for a more professional look
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "Indian Stock Market Analysis"

# Layout of the app with improved UI elements and a dark theme
app.layout = dbc.Container([
    # Header row
    dbc.Row([
        dbc.Col(html.H1("Indian Stock Market Analysis", className='text-center text-light mb-4'), width=12),
    ]),
    
    # Input row for stock ticker
    dbc.Row([
        dbc.Col(html.H4("Enter an NSE stock ticker symbol (e.g., RELIANCE.NS):", className='text-light'), width={"size": 6, "offset": 3}),
        dbc.Col(dcc.Input(id='input-stock', value='RELIANCE.NS', type='text', className='form-control mb-3'), width={"size": 6, "offset": 3}),
    ]),

    # Row for displaying stock details
    dbc.Row([
        dbc.Col(html.Div(id='stock-details', className='text-center text-light'), width=12),
    ]),

    # Row for daily data
    dbc.Row([
        dbc.Col(html.Div(id='daily-data-box', className='my-4 text-light'), width=12),
    ]),

    # Row for output graphs
    dbc.Row([
        dbc.Col(html.Div(id='output-graphs'), width=12),
    ]),

    # Row for displaying financial data
    dbc.Row([
        dbc.Col(html.Div(id='financials-data', className='my-4 text-light'), width=12),
    ]),

    # Interval component for refreshing data
    dcc.Interval(
        id='interval-component',
        interval=60*1000,  # 1 minute interval
        n_intervals=0
    )
], fluid=True, style={'backgroundColor': '#121212'})  # Dark background color

# Callback for stock details, graphs, and financial data
@app.callback(
    [Output(component_id='stock-details', component_property='children'),
     Output(component_id='daily-data-box', component_property='children'),
     Output(component_id='output-graphs', component_property='children'),
     Output(component_id='financials-data', component_property='children')],
    [Input(component_id='input-stock', component_property='value'),
     Input(component_id='interval-component', component_property='n_intervals')]
)
def Stock_info(stock_ticker, n_intervals):

    start_date = datetime.datetime(2010, 1, 1)
    end_date = datetime.datetime.now()

    try:
        # Validate the stock ticker symbol
        if not stock_ticker.endswith('.NS'):
            raise ValueError("Please enter a valid NSE stock ticker symbol ending with '.NS'.")

        # Download stock data
        stock_data = yf.download(stock_ticker, start=start_date, end=end_date)

        if stock_data.empty:
            raise ValueError(f"No data found for the ticker symbol '{stock_ticker}'.")

        # Extracting the most recent stock data
        latest_data = stock_data.iloc[-1]
        open_price = latest_data['Open']
        close_price = latest_data['Close']
        volume = latest_data['Volume']
        high_price = latest_data['High']
        low_price = latest_data['Low']

        percent_change = ((close_price - open_price) / open_price) * 100

        # Display whether the price is up or down
        if percent_change > 0:
            price_direction = "up"
            color = "lightgreen"  # Using a lighter green for better visibility
        else:
            price_direction = "down"
            color = "salmon"  # Using a lighter red for better visibility

        # Fetching intraday data for the current day
        intraday_data = yf.download(stock_ticker, period="1d", interval="5m")

        if intraday_data.empty:
            raise ValueError("No intraday data available. The market may be closed.")

        # Displaying the stock details
        stock_details = html.Div([
            html.H4(f"Details for {stock_ticker.upper()}", className='text-primary'),
            html.P(f"Open Price: ₹{open_price:.2f}", className='mb-2'),
            html.P(f"Close Price: ₹{close_price:.2f}", className='mb-2'),
            html.P(f"Volume: {volume}", className='mb-2'),
            html.P(f"High Price: ₹{high_price:.2f}", className='mb-2'),
            html.P(f"Low Price: ₹{low_price:.2f}", className='mb-2'),
            html.P(f"Price is {price_direction} by {abs(percent_change):.2f}%", style={'color': color}, className='mb-2')
        ], className='border p-3 bg-dark rounded text-light')

        # Creating the intraday graph for the present day
        intraday_graph = dcc.Graph(
            id="intraday-graph",
            figure={
                'data': [
                    go.Scatter(x=intraday_data.index, y=intraday_data['Close'], mode='lines', name='Intraday Price', line=dict(color='cyan'))
                ],
                'layout': go.Layout(
                    title=f"Intraday Price of {stock_ticker.upper()} (Today)",
                    xaxis={'title': 'Time', 'gridcolor': '#E2E2E2'},
                    yaxis={'title': 'Price (₹)', 'gridcolor': '#E2E2E2'},
                    height=300,
                    plot_bgcolor='#1E1E1E',  # Darker background for the plot area
                    paper_bgcolor='#121212'  # Dark background for the overall layout
                )
            }
        )

        daily_data_box = html.Div([
            html.H4("Daily Data", className='text-primary'),
            html.Table([
                html.Tr([html.Th("Date"), html.Td(latest_data.name.strftime('%Y-%m-%d'))]),
                html.Tr([html.Th("Open"), html.Td(f"₹{open_price:.2f}")]),
                html.Tr([html.Th("High"), html.Td(f"₹{high_price:.2f}")]),
                html.Tr([html.Th("Low"), html.Td(f"₹{low_price:.2f}")]),
                html.Tr([html.Th("Close"), html.Td(f"₹{close_price:.2f}")]),
                html.Tr([html.Th("Volume"), html.Td(volume)]),
                html.Tr([html.Th("Change"), html.Td(f"{price_direction} by {abs(percent_change):.2f}%", style={'color': color})])
            ], className='table table-striped table-hover text-light'),
            intraday_graph
        ], className='border p-4 bg-dark rounded text-light')

        # Fetching financial data
        stock_info = yf.Ticker(stock_ticker)
        financials = stock_info.financials
        if financials.empty:
            raise ValueError("No financial data available.")

        # Dummy predictions for future prices
        future_predictions = predict_future_prices(stock_data)

        # Displaying financial data
        financials_data = html.Div([
            html.H4("Financial Data", className='text-primary'),
            html.Table([
                html.Tr([html.Th("Metric"), html.Th("Value")]),
                *[html.Tr([html.Td(metric), html.Td(f"₹{value:.2f}")]) for metric, value in zip(financials.index, financials.iloc[:, 0])]
            ], className='table table-striped table-hover text-light'),
            html.H4("Future Price Predictions (Next 5 Days)", className='text-primary'),
            html.Table([
                html.Tr([html.Th("Day"), html.Th("Predicted Price")]),
                *[html.Tr([html.Td(f"Day {i + 1}"), html.Td(f"₹{price:.2f}")]) for i, price in enumerate(future_predictions)]
            ], className='table table-striped table-hover text-light')
        ], className='border p-4 bg-dark rounded text-light')

        # Creating the closing price graph
        closing_price_graph = dcc.Graph(
            id="closing-price-graph",
            figure={
                'data': [
                    go.Scatter(x=stock_data.index, y=stock_data['Close'], mode='lines', name='Close Price', line=dict(color='orange'))
                ],
                'layout': go.Layout(
                    title=f"Closing Prices of {stock_ticker.upper()}",
                    xaxis={'title': 'Date', 'gridcolor': '#E2E2E2'},
                    yaxis={'title': 'Price (₹)', 'gridcolor': '#E2E2E2'},
                    height=400,
                    plot_bgcolor='#1E1E1E',
                    paper_bgcolor='#121212'
                )
            }
        )

        # Creating the opening price graph
        opening_price_graph = dcc.Graph(
            id="opening-price-graph",
            figure={
                'data': [
                    go.Scatter(x=stock_data.index, y=stock_data['Open'], mode='lines', name='Open Price', line=dict(color='magenta')) 
                ],
                'layout': go.Layout(
                    title=f"Opening Prices of {stock_ticker.upper()}",
                    xaxis={'title': 'Date', 'gridcolor': '#E2E2E2'},
                    yaxis={'title': 'Price (₹)', 'gridcolor': '#E2E2E2'},
                    height=400,
                    plot_bgcolor='#1E1E1E',
                    paper_bgcolor='#121212'
                )
            }
        )

        # Combine all graphs into a single output component
        output_graphs = html.Div([
            closing_price_graph,
            opening_price_graph,
            intraday_graph
        ])

        return stock_details, daily_data_box, output_graphs, financials_data

    except Exception as e:
        return str(e), "", "", ""

if __name__ == '__main__':
    # Open the app in a new window
    Timer(1, lambda: webbrowser.open_new('http://127.0.0.1:8050/')).start()
    app.run_server(debug=True, use_reloader=False)
