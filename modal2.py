import plotly.graph_objects as go

def create_plot(df, selected_column, target_column):
    fig = go.Figure()

    # Add line trace for selected_column on primary y-axis
    fig.add_trace(go.Scatter(
        x=df['TIME'],
        y=df[selected_column],
        mode='lines+markers',
        line=dict(shape='linear', color='#1f77b4', width=3),  # Blue color for selected_column
        marker=dict(color='#ff7f0e', size=6, line=dict(color='#FFFFFF', width=1.5)),  # Orange markers
        name=selected_column,
        yaxis='y1'  # Use primary y-axis
    ))

    # Add line trace for target_column on secondary y-axis
    fig.add_trace(go.Scatter(
        x=df['TIME'],
        y=df[target_column],
        mode='lines+markers',
        line=dict(shape='linear', color='#d62728', width=3),  # Red color for target_column
        marker=dict(color='#9467bd', size=6, line=dict(color='#FFFFFF', width=1.5)),  # Purple markers
        name=target_column,
        yaxis='y2'  # Use secondary y-axis
    ))

    # Customize layout for aesthetics
    fig.update_layout(
        width=1000,
        title=f'{selected_column} and {target_column} Over Time',
        title_font=dict(size=26, family='Helvetica, Arial', color='#333333'),
        xaxis_title='Date',
        yaxis=dict(
            title=selected_column,
            showgrid=True,
            gridcolor='rgba(200, 200, 200, 0.4)',
            zeroline=False,
            titlefont=dict(color='#1f77b4'),
            tickfont=dict(color='#1f77b4')
        ),
        yaxis2=dict(
            title=target_column,
            overlaying='y',
            side='right',
            showgrid=False,
            titlefont=dict(color='#d62728'),
            tickfont=dict(color='#d62728')
        ),
        plot_bgcolor='#f4f4f4',
        hovermode='x unified',
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='right',
            x=1,
            font=dict(size=12, color='#333333')
        ),
        margin=dict(l=60, r=60, t=90, b=60),
        paper_bgcolor='#ffffff',
        font=dict(family='Helvetica, Arial', size=12, color='#333333')
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
            bgcolor='rgba(255, 255, 255, 0.8)',
            activecolor='#1f77b4',
            font=dict(size=11, color='#333333')
        ),
        rangeslider=dict(visible=True, bgcolor='rgba(200, 200, 200, 0.3)'),
        type="date"
    )

    return fig
