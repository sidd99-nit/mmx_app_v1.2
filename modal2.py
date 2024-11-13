import plotly.graph_objects as go

def create_plot(df, selected_column, target_column):
    fig = go.Figure()

    # Add line trace for selected_column with smooth lines and customized appearance
    fig.add_trace(go.Scatter(
        x=df['TIME'],
        y=df[selected_column],
        mode='lines+markers',
        line=dict(shape='linear', color='#1f77b4', width=3),  # A richer blue color
        marker=dict(color='#ff7f0e', size=6, line=dict(color='#FFFFFF', width=1.5)),  # Orange markers with white borders
        name=selected_column
    ))

    # Add line trace for target_column with distinct color
    fig.add_trace(go.Scatter(
        x=df['TIME'],
        y=df[target_column],
        mode='lines+markers',
        line=dict(shape='linear', color='#d62728', width=3),  # A different color (red) for target column
        marker=dict(color='#9467bd', size=6, line=dict(color='#FFFFFF', width=1.5)),  # Purple markers with white borders
        name=target_column
    ))

    # Customize the layout for aesthetics
    fig.update_layout(
        width=1000,
        title=f'{selected_column} and {target_column} Over Time',
        title_font=dict(size=26, family='Helvetica, Arial', color='#333333'),  # Stylish font and darker title color
        xaxis_title='Date',
        yaxis_title='Value',
        xaxis=dict(showgrid=True, gridcolor='rgba(200, 200, 200, 0.4)', zeroline=False, tickangle=-45),  # Light grid and angled ticks
        yaxis=dict(showgrid=True, gridcolor='rgba(200, 200, 200, 0.4)', zeroline=False),
        plot_bgcolor='#f4f4f4',  # Slightly darker background for contrast
        hovermode='x unified',  # Unified hover mode
        legend=dict(
            orientation='h', 
            yanchor='bottom', 
            y=1.02, 
            xanchor='right', 
            x=1, 
            font=dict(size=12, color='#333333')  # Stylish legend font
        ),
        margin=dict(l=60, r=60, t=90, b=60),  # Adjusted margins for better spacing
        paper_bgcolor='#ffffff',  # White paper background
        font=dict(family='Helvetica, Arial', size=12, color='#333333')  # General font settings
    )

    # Add subtle shadow effect to the plot area
    fig.update_layout(
        shapes=[
            dict(
                type="rect",
                xref="paper", yref="paper",
                x0=0, y0=0, x1=1, y1=1,
                line=dict(width=0),
                fillcolor="rgba(0, 0, 0, 0.1)",
                layer="below"
            )
        ]
    )

    # Add interactive elements like a range slider and buttons
    fig.update_xaxes(
        rangeselector=dict(
            buttons=list([
                dict(count=1, label="1M", step="month", stepmode="backward"),
                dict(count=3, label="3M", step="month", stepmode="backward"),
                dict(count=6, label="6M", step="month", stepmode="backward"),
                dict(step="all", label="All")
            ]),
            bgcolor='rgba(255, 255, 255, 0.8)',  # Background color for the range selector
            activecolor='#1f77b4',  # Active button color
            font=dict(size=11, color='#333333')
        ),
        rangeslider=dict(visible=True, bgcolor='rgba(200, 200, 200, 0.3)'),  # Transparent background for the range slider
        type="date"
    )

    return fig
