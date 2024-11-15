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
