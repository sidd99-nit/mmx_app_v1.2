import seaborn as sns
import matplotlib.pyplot as plt
import jax.numpy as jnp
from sklearn import metrics
import arviz

# Updated _create_shaded_line_plot function with explicit Matplotlib handling
def _create_shaded_line_plot(predictions: jnp.ndarray,
                             target: jnp.ndarray,
                             axis: plt.Axes,
                             title_prefix: str = "",
                             interval_mid_range: float = 0.9,
                             digits: int = 3) -> None:
    """Creates a plot of ground truth, predicted value, and credibility interval."""
    sns.set_theme(style="whitegrid")  # Apply Seaborn's clean theme

    if predictions.shape[1] != len(target):
        raise ValueError(
            "Predicted data and ground-truth data must have the same length."
        )

    upper_quantile = 1 - (1 - interval_mid_range) / 2
    lower_quantile = (1 - interval_mid_range) / 2
    upper_bound = jnp.quantile(a=predictions, q=upper_quantile, axis=0)
    lower_bound = jnp.quantile(a=predictions, q=lower_quantile, axis=0)

    r2, _ = arviz.r2_score(y_true=target, y_pred=predictions)
    mape = 100 * metrics.mean_absolute_percentage_error(
        y_true=target, y_pred=predictions.mean(axis=0))

    # Plot true values and predictions using explicit Matplotlib calls
    axis.plot(jnp.arange(target.shape[0]), target, label="True KPI", color="blue", linewidth=2)
    axis.plot(jnp.arange(target.shape[0]), predictions.mean(axis=0), label="Predicted KPI", color="green", linewidth=2)
    axis.fill_between(
        jnp.arange(target.shape[0]),
        lower_bound,
        upper_bound,
        color="green",
        alpha=0.2,  # Set transparency explicitly here
        label="Credibility Interval"
    )

    # Grid styling
    axis.yaxis.grid(color="gray", linestyle="dashed", alpha=0.3)
    axis.xaxis.grid(color="gray", linestyle="dashed", alpha=0.3)

    # Legend settings
    axis.legend(loc="upper left", fontsize=8, frameon=True, shadow=False)

    # Title and axis labels
    title = f"{title_prefix} True and Predicted KPI. R2 = {r2:.{digits}f}, MAPE = {mape:.{digits}f}%"
    axis.set_title(title, fontsize=12, fontweight="bold")
    axis.set_xlabel("Time", fontsize=10)
    axis.set_ylabel("KPI Value", fontsize=10)
    axis.tick_params(axis="both", which="major", labelsize=8)

# Updated _call_fit_plotter function
def _call_fit_plotter(
    predictions: jnp.ndarray,
    target: jnp.ndarray,
    interval_mid_range: float,
    digits: int,
) -> plt.Figure:
    """Calls the shaded line plot once for national and N times for geo models."""
    sns.set_context("notebook", font_scale=1)  # Adjust context for better visuals

    if predictions.ndim == 3:  # Multiple plots for geo models
        figure, axes = plt.subplots(
            predictions.shape[-1],
            figsize=(8, 3 * predictions.shape[-1]),  # Adjust height for compactness
            constrained_layout=True  # Better spacing between subplots
        )
        for i, ax in enumerate(axes):
            _create_shaded_line_plot(
                predictions=predictions[..., i],
                target=target[..., i],
                axis=ax,
                title_prefix=f"Geo {i}:",
                interval_mid_range=interval_mid_range,
                digits=digits
            )
    else:  # Single plot for national model
        figure, ax = plt.subplots(1, 1, figsize=(8, 4))  # Compact single plot
        _create_shaded_line_plot(
            predictions=predictions,
            target=target,
            axis=ax,
            interval_mid_range=interval_mid_range,
            digits=digits
        )

    # Save figure as PNG for reporting
    plt.savefig("predicted_vs_actual_plot.png", dpi=300, bbox_inches="tight")
    plt.close()
    return figure
