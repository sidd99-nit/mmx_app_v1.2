import seaborn as sns
import matplotlib.pyplot as plt
import jax.numpy as jnp
from sklearn.metrics import mean_absolute_percentage_error
import arviz

# Function to plot shaded line plot
def _create_shaded_line_plot(predictions: jnp.ndarray,
                             target: jnp.ndarray,
                             axis: plt.Axes,
                             title_prefix: str = "",
                             interval_mid_range: float = 0.9,
                             digits: int = 3) -> None:
    """
    Creates a plot of ground truth, predicted values, and credibility intervals.

    Args:
        predictions: 2D array of predicted values (samples x timesteps).
        target: 1D array of true values. Must match predictions in length.
        axis: Matplotlib axis to plot the data.
        title_prefix: Prefix for the plot title.
        interval_mid_range: Mid-range interval for plotting (e.g., 0.9 for 90% CI).
        digits: Number of decimals to display in metrics.
    """
    if predictions.shape[1] != len(target):
        raise ValueError("Predictions and target lengths do not match.")
    
    # Calculate credibility intervals
    upper_quantile = 1 - (1 - interval_mid_range) / 2
    lower_quantile = (1 - interval_mid_range) / 2
    upper_bound = jnp.quantile(predictions, q=upper_quantile, axis=0)
    lower_bound = jnp.quantile(predictions, q=lower_quantile, axis=0)

    # Compute metrics
    r2 = arviz.r2_score(y_true=target, y_pred=predictions.mean(axis=0))[0]
    mape = 100 * mean_absolute_percentage_error(target, predictions.mean(axis=0))

    # Plot true and predicted values
    time_steps = jnp.arange(len(target))
    axis.plot(time_steps, target, label="True KPI", color="blue", linewidth=2)
    axis.plot(time_steps, predictions.mean(axis=0), label="Predicted KPI", color="green", linewidth=2)
    axis.fill_between(time_steps, lower_bound, upper_bound, color="green", alpha=0.2, label="Credibility Interval")

    # Add legend, title, and grid
    axis.legend(loc="upper left", fontsize=8)
    axis.grid(color="gray", linestyle="--", linewidth=0.5, alpha=0.7)
    axis.set_title(f"{title_prefix} R2: {r2:.{digits}f}, MAPE: {mape:.{digits}f}%", fontsize=12)
    axis.set_xlabel("Time")
    axis.set_ylabel("KPI Value")
    axis.tick_params(axis="both", labelsize=10)

# Function to handle multiple plots
def _call_fit_plotter(predictions: jnp.ndarray,
                      target: jnp.ndarray,
                      interval_mid_range: float = 0.9,
                      digits: int = 3) -> plt.Figure:
    """
    Generates shaded line plots for single or multiple models.

    Args:
        predictions: 2D or 3D array of predicted values (samples x timesteps x geos).
        target: 1D or 2D array of true values (timesteps x geos).
        interval_mid_range: Mid-range interval for plotting (e.g., 0.9 for 90% CI).
        digits: Number of decimals to display in metrics.

    Returns:
        A Matplotlib figure containing the plots.
    """
    sns.set_theme(style="whitegrid")  # Apply Seaborn theme
    is_geo_model = predictions.ndim == 3

    # Determine figure layout
    if is_geo_model:
        num_geos = predictions.shape[-1]
        fig, axes = plt.subplots(num_geos, figsize=(8, 3 * num_geos), constrained_layout=True)
        axes = axes if num_geos > 1 else [axes]  # Ensure axes is iterable
        for i, ax in enumerate(axes):
            _create_shaded_line_plot(predictions=predictions[..., i],
                                     target=target[..., i],
                                     axis=ax,
                                     title_prefix=f"Geo {i}:",
                                     interval_mid_range=interval_mid_range,
                                     digits=digits)
    else:
        fig, ax = plt.subplots(figsize=(8, 4))
        _create_shaded_line_plot(predictions=predictions,
                                 target=target,
                                 axis=ax,
                                 interval_mid_range=interval_mid_range,
                                 digits=digits)

    # Save and return the figure
    plt.savefig("predicted_vs_actual_plot.png", dpi=300, bbox_inches="tight")
    plt.close()
    return fig
