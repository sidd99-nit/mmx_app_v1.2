import plotly.graph_objects as go
import numpy as np
import pandas as pd

def plot_media_baseline_contribution_area_plotly(
    media_mix_model,  # Placeholder for the actual LightweightMMM model
    target_scaler=None,
    channel_names=None,
    fig_size=(900, 500),
    legend_outside=False
):
    """
    Plots an interactive stacked area chart using Plotly to visualize weekly
    media & baseline contribution.
    """
    # Assume create_media_baseline_contribution_df() returns a DataFrame
    # Here, we're using a placeholder for demonstration purposes
    # Replace this with the actual data extraction from media_mix_model
    contribution_df = create_media_baseline_contribution_df(
        media_mix_model=media_mix_model,
        target_scaler=target_scaler,
        channel_names=channel_names
    )

    # Filter columns that contain "contribution" and reverse for stacking order
    contribution_columns = [col for col in contribution_df.columns if "contribution" in col]
    contribution_df_for_plot = contribution_df[contribution_columns]
    contribution_df_for_plot = contribution_df_for_plot[contribution_df_for_plot.columns[::-1]]
    period = np.arange(1, contribution_df_for_plot.shape[0] + 1)
    contribution_df_for_plot["Period"] = period

    # Create a color palette for the channels
    colors = plotly.colors.sequential.Viridis

    # Initialize the Plotly figure
    fig = go.Figure()

    # Add traces for each contribution column as a stacked area chart
    for i, col in enumerate(contribution_columns):
        fig.add_trace(go.Scatter(
            x=contribution_df_for_plot["Period"],
            y=contribution_df_for_plot[col],
            mode='none',  # No lines, only fill areas
            name=col,
            stackgroup='one',  # Enables stacking
            fill='tonexty',  # Fills to the previous trace for stacking
            line=dict(width=0.5, color=colors[i % len(colors)]),
            hoverinfo='x+y',  # Show x and y values in hover
            fillcolor=colors[i % len(colors)]
        ))

    # Update layout for aesthetics
    fig.update_layout(
        title="Weekly Media & Baseline Contribution Over Time",
        title_x=0.5,
        title_font=dict(size=24, color="darkblue"),
        xaxis=dict(
            title="Period",
            tickvals=contribution_df_for_plot["Period"],
            ticktext=contribution_df_for_plot["Period"],
            title_font=dict(size=18, color="grey"),
            tickangle=-45,
            showgrid=False
        ),
        yaxis=dict(
            title="Contribution",
            title_font=dict(size=18, color="grey"),
            showgrid=True,
            gridcolor='lightgrey'
        ),
        legend=dict(
            title="Channels",
            font=dict(size=14),
            orientation="h",
            yanchor="bottom",
            y=-0.3 if legend_outside else 1.0,
            x=0.5,
            xanchor="center"
        ),
        margin=dict(t=100, b=100, l=50, r=50),
        hovermode="x unified",  # Unify hover across all traces on the same x-value
        plot_bgcolor='white',  # White background for a clean look
        height=fig_size[1],
        width=fig_size[0]
    )

    # Add hover template for a better user experience
    for trace in fig.data:
        trace.hovertemplate = '<b>Period %{x}</b><br>Contribution: %{y}<extra></extra>'

    fig.show()

# Replace this with actual data fetching
# Example call (assuming necessary data and functions are available)
# plot_media_baseline_contribution_area_plotly(media_mix_model)
