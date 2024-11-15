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
        loc="upper right",       # Place the legend inside the plot (e.g., 'upper right')
        fontsize=8,              # Reduce font size
        markerscale=0.7,         # Shrink marker size
        labelspacing=0.4,        # Reduce vertical spacing between labels
        borderpad=0.3,           # Reduce padding around the legend
        handlelength=1,          # Adjust marker line length
        handleheight=0.5,        # Adjust marker height
        frameon=True,            # Enable frame (optional: set to False for no frame)
        framealpha=0.7,          # Make frame slightly transparent
        edgecolor="grey"         # Frame border color
    )

# Add a subtle background color for the plot area.
ax.set_facecolor("#f9f9f9")

# Final adjustments and return.
plt.tight_layout()
plt.close()
return fig

///////


import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set Seaborn aesthetic style.
sns.set_theme(style="whitegrid", palette="muted")

# Create contribution dataframe for the plot.
contribution_columns = [
    col for col in contribution_df.columns if "contribution" in col
]
contribution_df_for_plot = contribution_df.loc[:, contribution_columns]
contribution_df_for_plot = contribution_df_for_plot[
    contribution_df_for_plot.columns[::-1]
]
period = np.arange(1, contribution_df_for_plot.shape[0] + 1)
contribution_df_for_plot.loc[:, "period"] = period

# Plot the stacked area chart with enhanced aesthetics.
fig, ax = plt.subplots(figsize=(12, 8))

# Apply Seaborn color palette.
colors = sns.color_palette("coolwarm", len(contribution_columns))

# Stack the area plot.
contribution_df_for_plot.plot.area(
    x="period", 
    stacked=True, 
    ax=ax, 
    color=colors, 
    alpha=0.8
)

# Enhance the appearance.
ax.set_title("Attribution Over Time", fontsize=18, fontweight="bold", color="navy")
ax.set_ylabel("Baseline & Media Channels Attribution", fontsize=14, color="darkblue")
ax.set_xlabel("Period", fontsize=14, color="darkblue")
ax.tick_params(axis="x", rotation=45, labelsize=10)
ax.tick_params(axis="y", labelsize=10)
ax.set_xlim(1, contribution_df_for_plot["period"].max())
ax.set_xticks(contribution_df_for_plot["period"])
ax.set_xticklabels(contribution_df_for_plot["period"])

# Get handles and labels for sorting.
handles, labels = ax.get_legend_handles_labels()

# Customize the legend.
if legend_outside:
    ax.legend(
        handles[::-1], 
        labels[::-1], 
        loc="center left", 
        bbox_to_anchor=(1, 0.5), 
        fontsize=12, 
        title="Channels", 
        title_fontsize=14
    )
else:
    ax.legend(
        handles[::-1], 
        labels[::-1], 
        fontsize=12, 
        title="Channels", 
        title_fontsize=14
    )

# Add gridlines and background color for better readability.
ax.grid(color="gray", linestyle="--", linewidth=0.5, alpha=0.7)
fig.patch.set_facecolor("white")

# Save the figure as PNG.
image_path = "attribution_over_time.png"
fig.savefig(image_path, dpi=300, bbox_inches="tight", transparent=False)

plt.close(fig)

return fig

