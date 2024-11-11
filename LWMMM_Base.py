import os
import pandas as pd
import warnings
import logging
import matplotlib.pyplot as plt
import jax.numpy as jnp
import numpyro
from lightweight_mmm import lightweight_mmm, plot, preprocessing
import seaborn as sns
import arviz
from sklearn import metrics
import matplotlib
matplotlib.use('Agg')

plt.rcParams['figure.figsize'] = [20,8]


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

warnings.filterwarnings('ignore')

SEED = 105

def create_shaded_line_plot(predictions: jnp.ndarray,
                            target: jnp.ndarray,
                            axis: plt.Axes,
                            title_prefix: str = "",
                            interval_mid_range: float = 0.9,
                            digits: int = 3) -> None:
    try:
        if predictions.shape[1] != len(target):
            raise ValueError("Predicted data and ground-truth data must have the same length.")
        
        upper_quantile = 1 - (1 - interval_mid_range) / 2
        lower_quantile = (1 - interval_mid_range) / 2
        upper_bound = jnp.quantile(a=predictions, q=upper_quantile, axis=0)
        lower_bound = jnp.quantile(a=predictions, q=lower_quantile, axis=0)

        r2, _ = arviz.r2_score(y_true=target, y_pred=predictions)
        mape = 100 * metrics.mean_absolute_percentage_error(y_true=target, y_pred=predictions.mean(axis=0))

        # Plot the true and predicted values
        axis.plot(jnp.arange(target.shape[0]), target, c="#34495e", lw=2, alpha=0.9, label="True KPI")
        axis.plot(jnp.arange(target.shape[0]), predictions.mean(axis=0), c="#2ecc71", lw=2, alpha=0.9, label="Predicted KPI")
        axis.fill_between(jnp.arange(target.shape[0]), lower_bound, upper_bound, alpha=0.2, color="#2ecc71", label="Confidence Interval")

        # Add custom gridlines
        axis.grid(True, which='both', color='gray', linestyle='dashed', linewidth=0.5, alpha=0.3)
        
        # Customize tick styles
        axis.tick_params(axis='both', which='major', labelsize=10)
        axis.minorticks_on()
        axis.tick_params(axis='both', which='minor', length=4, color='black', alpha=0.5)

        # Legend and title styling
        axis.legend(loc="best", fontsize=10, frameon=True, fancybox=True, shadow=True)
        title = f"{title_prefix} True and predicted KPI. R2 = {r2:.{digits}f}, MAPE = {mape:.{digits}f}%"
        axis.set_title(title, fontsize=14, weight='bold', color="#2c3e50")

        plt.close()
        logging.info("Shaded line plot created successfully.")
        
    except Exception as e:
        logging.error(f"An error occurred while creating the shaded line plot: {e}")
        raise

def call_fit_plotter(predictions: jnp.ndarray,
                     target: jnp.ndarray,
                     interval_mid_range: float,
                     digits: int) -> plt.Figure:
    try:
        figure_size = (12, 4)  # Common figure size
        
        # Use seaborn style for a more polished look
        with plt.style.context('seaborn-darkgrid'):  
            if predictions.ndim == 3:  # Multiple plots for geo model
                figure, axes = plt.subplots(predictions.shape[-1], figsize=figure_size)
                for i, ax in enumerate(axes):
                    create_shaded_line_plot(predictions=predictions[..., i],
                                            target=target[..., i],
                                            axis=ax,
                                            title_prefix=f"Geo {i}:",
                                            interval_mid_range=interval_mid_range,
                                            digits=digits)
            else:  # Single plot for national model
                figure, ax = plt.subplots(1, 1, figsize=figure_size)
                create_shaded_line_plot(predictions=predictions,
                                        target=target,
                                        axis=ax,
                                        interval_mid_range=interval_mid_range,
                                        digits=digits)
        
        logging.info("Fit plot created successfully.")
        return figure

    except Exception as e:
        logging.error(f"An error occurred while creating the fit plot: {e}")
        raise

def plot_model_fit(media_mix_model,
                   target_scaler=None,
                   interval_mid_range: float = 0.9,
                   digits: int = 3) -> plt.Figure:
    try:
        if not hasattr(media_mix_model, "trace"):
            logging.error("Model is not fitted yet. Cannot plot fit.")
            raise ValueError("Model needs to be fit first before attempting to plot its fit.")
        
        target_train = media_mix_model._target
        posterior_pred = media_mix_model.trace["mu"]

        if target_scaler:
            posterior_pred = target_scaler.inverse_transform(posterior_pred)
            target_train = target_scaler.inverse_transform(target_train)

        figure = call_fit_plotter(predictions=posterior_pred,
                                  target=target_train,
                                  interval_mid_range=interval_mid_range,
                                  digits=digits)

        logging.info("Model fit plot generated successfully.")
        return figure

    except Exception as e:
        logging.error(f"An error occurred while plotting model fit: {e}")
        raise

def create_directory(path):
    """Create directory if it doesn't exist."""
    try:
        if not os.path.exists(path):
            os.makedirs(path)
            logging.info(f"Created directory: {path}")
        else:
            logging.info(f"Directory already exists: {path}")
    except OSError as e:
        logging.error(f"Error creating directory {path}: {e}")
        raise

def load_data(file_path):
    """Load data from Excel file and handle file not found errors."""
    try:
        logging.info(f"Loading data from {file_path}")
        final_data = pd.read_excel(file_path, index_col=0)
        logging.info(f"Data loaded with shape: {final_data.shape}")
        return final_data
    except FileNotFoundError:
        logging.error(f"File not found: {file_path}")
        raise
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        raise

def set_device_count(count):
    """Set the number of devices (CPUs) for parallel processing."""
    try:
        logging.info(f"Setting device count to {count}")
        numpyro.set_host_device_count(count)
    except Exception as e:
        logging.error(f"Error setting device count: {e}")
        raise

def prepare_data(final_data, mdsp_cols, control_vars, sales_cols):
    """Prepare the data for modeling and handle missing columns."""
    try:
        logging.info("Preparing data for modeling")
        media_data = final_data[mdsp_cols].to_numpy()
        extra_features = final_data[control_vars].to_numpy()
        target = final_data[sales_cols].to_numpy().squeeze()
        costs = final_data[mdsp_cols].sum().to_numpy()
        return media_data, extra_features, target, costs
    except KeyError as e:
        logging.error(f"Column not found in data: {e}")
        raise
    except Exception as e:
        logging.error(f"Error preparing data: {e}")
        raise

def scale_data(media_data, extra_features, target, costs):
    """Scale the data using predefined scalers and handle errors."""
    try:
        logging.info("Scaling data")
        media_scaler = preprocessing.CustomScaler(divide_operation=jnp.mean, multiply_by=0.01)
        extra_features_scaler = preprocessing.CustomScaler(divide_operation=jnp.mean)
        target_scaler = preprocessing.CustomScaler(divide_operation=jnp.mean, multiply_by=1.5)
        cost_scaler = preprocessing.CustomScaler(divide_operation=jnp.mean, multiply_by=1)

        media_data = media_scaler.fit_transform(media_data)
        extra_features = extra_features_scaler.fit_transform(extra_features)
        target = target_scaler.fit_transform(target)
        costs = cost_scaler.fit_transform(costs)

        return media_data, extra_features, target, costs, media_scaler, target_scaler, cost_scaler
    except Exception as e:
        logging.error(f"Error scaling data: {e}")
        raise

def train_model(media_data, costs, target, extra_features, mdsp_cols):
    """Train the Hill-adstock model and handle potential errors."""
    try:
        logging.info("Training model using Hill-adstock")
        mmm_hill_adstock = lightweight_mmm.LightweightMMM(model_name="hill_adstock")

        # custom_priors = {
        #     "intercept": numpyro.distributions.HalfNormal(scale=100.),
        #     "lag_weight": numpyro.distributions.Beta(concentration1=2., concentration0=1.),
        #     "half_max_effective_concentration": numpyro.distributions.Gamma(concentration=2., rate=1.5)
        # }

        mmm_hill_adstock.fit(
            media=media_data,
            media_prior=costs,
            target=target,
            extra_features=extra_features,
            number_warmup=1000,
            number_samples=1000,
            number_chains=2,
            media_names=mdsp_cols,
            weekday_seasonality=False,
            seasonality_frequency=52, 
            seed=SEED
        )

        logging.info("Training model using Hill-adstock done")
        return mmm_hill_adstock
    except Exception as e:
        logging.error(f"Error training model: {e}")
        raise

def visualize_fit(mmm_hill_adstock, target_scaler):
    """Visualize model fit and save the plot, handling errors."""
    try:
        logging.info("Visualizing model fit")
        fig = plot.plot_model_fit(mmm_hill_adstock, target_scaler=target_scaler)

        # Define save path with absolute path handling
        save_path = os.path.abspath("static/images/LWMMM_Base")
        create_directory(save_path)

        # Save the plot
        plt_path = os.path.join(save_path, "model_fit.png")
        fig.savefig(plt_path, bbox_inches='tight')
        logging.info(f"Model fit plot saved to {plt_path}")
    except Exception as e:
        logging.error(f"Error visualizing or saving model fit plot: {e}")
        raise

def extract_metrics(mmm_hill_adstock, target_scaler, cost_scaler, channel_cols):
    """Extract media effectiveness and ROI, handle potential errors."""
    try:
        logging.info("Extracting media effectiveness and ROI estimation")
        media_effect_hat_hill, roi_hat_hill = mmm_hill_adstock.get_posterior_metrics(
            target_scaler=target_scaler, 
            cost_scaler=cost_scaler
        )
        roi_hat_yr_hill = pd.DataFrame(roi_hat_hill, columns=channel_cols)
        logging.info(f"Media contribution for each channel: {media_effect_hat_hill}")
        logging.info(f"ROI hat for each channel: {roi_hat_hill}")

        return media_effect_hat_hill, roi_hat_hill, roi_hat_yr_hill
    except Exception as e:
        logging.error(f"Error extracting metrics: {e}")
        raise

def save_roi_to_excel(roi_hat_yr_hill, file_name):
    """Save ROI estimates to Excel and handle file errors."""
    try:
        output_path = "/static/output_data/LWMMM_Base"
        create_directory(output_path)

        file_path = os.path.join(output_path, file_name)
        roi_hat_yr_hill.to_excel(file_path, index=False)
        logging.info(f"ROI estimation saved to {file_path}")
    except Exception as e:
        logging.error(f"Error saving ROI to Excel: {e}")
        raise

def visualize_roi(roi_hat_hill, channel_cols):
    """Visualize ROI as a bar chart and handle plotting errors."""
    try:
        logging.info("Visualizing ROI")
        fig = plot.plot_bars_media_metrics(
            metric=roi_hat_hill, 
            metric_name="ROI hat", 
            channel_names=channel_cols
        )

        # Define save path with absolute path handling
        save_path = os.path.abspath("static/images/LWMMM_Base")
        create_directory(save_path)

        # Save the ROI plot
        plt_path = os.path.join(save_path, "roi_bar_chart.png")
        fig.savefig(plt_path, bbox_inches='tight')
        logging.info(f"ROI bar chart saved to {plt_path}")
    except Exception as e:
        logging.error(f"Error visualizing or saving ROI bar chart: {e}")
        raise

def visualize_contributions(mmm_hill_adstock, target_scaler):
    """Visualize media & baseline contributions and handle errors."""
    try:
        logging.info("Visualizing media & baseline contribution over time")
        fig = plot.plot_media_baseline_contribution_area_plot(
            media_mix_model=mmm_hill_adstock, 
            target_scaler=target_scaler,
            fig_size=(12, 4),
            legend_outside=True
        )

        # Define save path with absolute path handling
        save_path = os.path.abspath("static/images/LWMMM_Base")
        create_directory(save_path)

        # Save the contributions plot
        plt_path = os.path.join(save_path, "media_baseline_contribution.png")
        fig.savefig(plt_path, bbox_inches='tight')
        logging.info(f"Media & baseline contribution plot saved to {plt_path}")
    except Exception as e:
        logging.error(f"Error visualizing or saving media & baseline contribution plot: {e}")
        raise

def calculate_average_roi(roi_hat_yr_hill, channel_cols):
    """Calculate average ROI and handle errors."""
    try:
        logging.info("Calculating average ROI for each channel")
        roi_avg_hill = pd.DataFrame([roi_hat_yr_hill.mean()], columns=channel_cols)
        return roi_avg_hill
    except Exception as e:
        logging.error(f"Error calculating average ROI: {e}")
        raise

def save_average_roi_to_excel(roi_avg_hill, file_name):
    """Save average ROI to Excel and handle file errors."""
    try:
        output_path = "/static/output_data/LWMMM_Base"
        create_directory(output_path)

        file_path = os.path.join(output_path, file_name)
        roi_avg_hill.to_excel(file_path, index=False)
        logging.info(f"Average ROI saved to {file_path}")
    except Exception as e:
        logging.error(f"Error saving average ROI to Excel: {e}")
        raise

def run_lwmmm_base(media_channels, control_vars, target_col, data_path):
    try:
        # Load data
        final_data = load_data(data_path)

        # Print data head
        print(final_data.head())

        print("media_channels:", media_channels)
        print("base_variables:", control_vars)
        print("target_variable:", target_col)

        # Set number of CPUs for parallel processing
        set_device_count(20)

        # Define columns
        mdsp_cols = media_channels
        channel_cols = media_channels
        sales_cols = target_col

        # Prepare data
        media_data, extra_features, target, costs = prepare_data(final_data, mdsp_cols, control_vars, sales_cols)

        # Scale data
        media_data, extra_features, target, costs, media_scaler, target_scaler, cost_scaler = scale_data(
            media_data, extra_features, target, costs)

        # Train model
        mmm_hill_adstock = train_model(media_data, costs, target, extra_features, mdsp_cols)

        # Visualize model fit
        visualize_fit(mmm_hill_adstock, target_scaler)

        # Extract metrics
        media_effect_hat_hill, roi_hat_hill, roi_hat_yr_hill = extract_metrics(
            mmm_hill_adstock, target_scaler, cost_scaler, channel_cols)

        # Save ROI to Excel
        #save_roi_to_excel(roi_hat_yr_hill, "RICOLA_EVERYDAY_roi.xlsx")

        # Visualize ROI
        visualize_roi(roi_hat_hill, channel_cols)

        # Visualize media & baseline contributions
        visualize_contributions(mmm_hill_adstock, target_scaler)

        # Calculate and save average ROI
        roi_avg_hill = calculate_average_roi(roi_hat_yr_hill, channel_cols)

        # Creating the Average ROI Plot
        # Create a plot of average ROIs and save at the location
        plt.figure(figsize=(12,4))

        # Define a custom color palette with the given hex codes
        custom_palette = ['#240750', '#344C64', '#577B8D', '#57A6A1']

        sns.lineplot(x=roi_avg_hill.columns, y=roi_avg_hill.iloc[0], marker='o', markersize=10, linewidth=2.5, color=custom_palette[3])
        plt.title('ROIs', fontsize=18, color=custom_palette[2])
        plt.xlabel('Media Channels', fontsize=14, color=custom_palette[1])
        plt.ylabel('ROI', fontsize=14, color=custom_palette[1])
        plt.xticks(fontsize=12, color=custom_palette[0], rotation = 70)
        plt.yticks(fontsize=12, color=custom_palette[0])
        plt.grid(False, linestyle='--', linewidth=0.6, alpha=0.7)
        plt.box(False)

        plt_path = os.path.join("static/images/LWMMM_Base","roi_plot.png")
        plt.savefig(plt_path, bbox_inches='tight', dpi=300)

        print("roi_avg_hill",roi_avg_hill)    
        return roi_avg_hill

    except Exception as e:
        logging.error(f"An error occurred: {e}")
