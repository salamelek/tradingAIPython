import plotly.graph_objects as go
import pandas as pd
from dataGetter import *

# Load the data
df = getCandles("./marketData/XRPUSDT-5m-2024")
df = df.reset_index()  # Ensure 'open_time' is a column
df = df[:100]

# Example positions (start_time, end_time)
positions = [
    {'start_time': '2024-01-01 12:00', 'end_time': '2024-01-01 14:00', 'color': 'rgba(0, 255, 0, 0.5)'},  # Green rectangle
    {'start_time': '2024-01-02 08:00', 'end_time': '2024-01-02 10:00', 'color': 'rgba(255, 0, 0, 0.5)'},  # Red rectangle
]

# Create the candlestick chart
fig = go.Figure(data=[go.Candlestick(
    x=df['open_time'],
    open=df['Open'],
    high=df['High'],
    low=df['Low'],
    close=df['Close']
)])

# Add rectangles for each position
for pos in positions:
    fig.add_shape(
        type="rect",
        x0=pos['start_time'],
        x1=pos['end_time'],
        y0=df['Low'].min(),  # Bottom of the rectangle (minimum price)
        y1=df['High'].max(),  # Top of the rectangle (maximum price)
        line=dict(color=pos['color'], width=0),  # No border
        fillcolor=pos['color'],  # Fill color with transparency
        layer="below",  # Place the rectangle below the candlesticks
        opacity=0.2  # Transparency of the rectangle
    )

# Update layout
fig.update_layout(
    xaxis=dict(rangeslider=dict(visible=True)),
    title='Candlestick Chart with Positions'
)

# Show the chart
fig.show()