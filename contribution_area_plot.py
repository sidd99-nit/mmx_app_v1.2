# Create contribution dataframe for the plot.
contribution_columns = [
    col for col in contribution_df.columns if "contribution" in col
]
contribution_df_for_plot = contribution_df.loc[:, contribution_columns]
contribution_df_for_plot = contribution_df_for_plot[
    contribution_df_for_plot.columns[::-1]]
period = np.arange(1, contribution_df_for_plot.shape[0] + 1)
contribution_df_for_plot.loc[:, "period"] = period

# Plot the stacked area chart.
fig, ax = plt.subplots(figsize=fig_size)

# Use a vibrant color palette for the area chart.
colors = plt.cm.viridis(np.linspace(0, 1, len(contribution_columns)))
contribution_df_for_plot.plot.area(
    x="period", stacked=True, ax=ax, alpha=0.85, color=colors)

# Enhance the title.
ax.set_title(
    "Attribution Over Time", fontsize=18, fontweight="bold", color="navy", pad=20
)

# Improve axis label styling.
ax.set_ylabel("Baseline & Media Channels Attribution", fontsize=14, fontweight="medium")
ax.set_xlabel("Period", fontsize=14, fontweight="medium")
ax.set_xlim(1, contribution_df_for_plot["period"].max())
ax.set_xticks(contribution_df_for_plot["period"])
ax.set_xticklabels(
    contribution_df_for_plot["period"], fontsize=10, rotation=45, ha="right"
)
ax.tick_params(axis="y", labelsize=10)

# Add gridlines for better readability.
ax.grid(axis="y", linestyle="--", alpha=0.7)

# Get handles and labels for sorting.
handles, labels = ax.get_legend_handles_labels()

# Customize the legend.
if legend_outside:
    ax.legend(
        handles[::-1],
        labels[::-1],
        loc="center left",
        bbox_to_anchor=(1, 0.5),
        title="Channels",
        title_fontsize=12,
        fontsize=10,
        frameon=True,
        edgecolor="black",
    )
else:
    ax.legend(
        handles[::-1],
        labels[::-1],
        title="Channels",
        title_fontsize=12,
        fontsize=10,
        frameon=True,
        edgecolor="black",
    )

# Add a subtle background color for the plot area.
ax.set_facecolor("#f9f9f9")

# Final adjustments and return.
plt.tight_layout()
plt.close()
return fig

///////

import plotly.graph_objects as go
import numpy as np
import plotly.express as px
import plotly.io as pio

# Ensure baseline contribution is at the bottom.
baseline_column = "baseline_contribution"  # Adjust this to match the column name.
contribution_columns = [
    col for col in contribution_df.columns if "contribution" in col and col != baseline_column
]
contribution_columns = [baseline_column] + contribution_columns  # Move baseline to the front.

# Prepare the data for the plot.
contribution_df_for_plot = contribution_df[contribution_columns]
period = np.arange(1, contribution_df_for_plot.shape[0] + 1)
contribution_df_for_plot.loc[:, "period"] = period

# Create a stacked area chart using Plotly.
fig = go.Figure()

# Define an aesthetic color palette for better visibility.
colors = px.colors.qualitative.Pastel + px.colors.qualitative.Bold

# Add traces for each contribution column.
for idx, column in enumerate(contribution_df_for_plot.columns[:-1]):  # Exclude "period".
    fig.add_trace(
        go.Scatter(
            x=contribution_df_for_plot["period"],
            y=contribution_df_for_plot[column],
            mode="lines",
            fill="tonexty",
            name=column,
            line=dict(width=0.5, color=colors[idx % len(colors)]),
            hoverinfo="x+y+name",
        )
    )

# Update layout for aesthetics.
fig.update_layout(
    title=dict(
        text="Attribution Over Time",
        font=dict(size=24, color="navy", family="Arial"),
        x=0.5,
        xanchor="center",
    ),
    xaxis=dict(
        title="Period",
        titlefont=dict(size=16, family="Arial"),
        tickmode="array",
        tickvals=contribution_df_for_plot["period"],
        ticktext=contribution_df_for_plot["period"],
    ),
    yaxis=dict(
        title="Baseline & Media Channels Attribution",
        titlefont=dict(size=16, family="Arial"),
        gridcolor="lightgrey",
    ),
    legend=dict(
        title=dict(text="Channels", font=dict(size=14)),
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="center",
        x=0.5,
    ),
    plot_bgcolor="#f9f9f9",  # Subtle background color.
    margin=dict(l=50, r=50, t=50, b=50),  # Adjust margins.
)

# Add gridlines for better readability.
fig.update_xaxes(showgrid=True, gridwidth=0.5, gridcolor="lightgrey")
fig.update_yaxes(showgrid=True, gridwidth=0.5, gridcolor="lightgrey")

# Save the figure as an image.
image_path = "attribution_over_time.png"  # Define the output path.
pio.write_image(fig, file=image_path, format="png", width=1200, height=800)

# Return the figure for display if needed.
return fig

