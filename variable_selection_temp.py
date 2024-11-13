import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

np.set_printoptions(suppress=True)

def feature_selection(data, target, media_channels, organic_channels, control_features, mandatory_features):
    # Combine all features
    features = control_features
    dropped_features = []  # List to store dropped features as tuples (feature, reason)

    # Function for singularity check
    def singularity_check(df, features):
        X = df[features]
        corr_matrix = X.corr()
        singular_vars = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                if abs(corr_matrix.iloc[i, j]) > 0.9:  # Threshold for singularity
                    singular_vars.append(corr_matrix.columns[j])
        return singular_vars

    # Function for VIF calculation
    def calculate_vif(df, features):
        X = df[features]
        vif_data = pd.DataFrame()
        vif_data["feature"] = X.columns
        vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
        return vif_data

    # Function for p-values checking using OLS
    def check_p_values(df, features, target):
        X = df[features]
        y = df[target]
        model = sm.OLS(y, X, hasconst=False).fit()
        p_values = model.pvalues
        return p_values

    # Initial selected features
    selected_features = features

    # Add mandatory features to selected features
    selected_features += [feature for feature in mandatory_features if feature not in selected_features]

    # Step 1: Singularity check
    singular_vars = singularity_check(data, selected_features)
    for feature in singular_vars:
        dropped_features.append((feature, "Singularity"))
    selected_features = [feature for feature in selected_features if feature not in singular_vars]

    # Step 2: VIF calculation
    if len(selected_features) > 0:
        vif_data = calculate_vif(data, selected_features)
        while vif_data['VIF'].max() > 15:  # Threshold for VIF
            max_vif_index = vif_data['VIF'].idxmax()
            feature_to_drop = vif_data.loc[max_vif_index, 'feature']
            dropped_features.append((feature_to_drop, "VIF"))
            selected_features.remove(feature_to_drop)
            vif_data = calculate_vif(data, selected_features)

    # Step 3: p-values checking
    if len(selected_features) > 0:
        p_values = check_p_values(data, selected_features, target)
        for feature, p_value in p_values.items():
            if p_value >= 0.2:  # Threshold for p-values
                dropped_features.append((feature, "p-value"))
        selected_features = [feature for feature in selected_features if p_values[feature] < 0.2]

    return selected_features, dropped_features

# Example usage:
def main_variable_selection(data, target_variable, media_channels, organic_channels, all_base_variables, seasonality_type):
    # Set control features based on the seasonality type
    control_features_1 = all_base_variables + ['seasonality_Bayesian']
    control_features_2 = all_base_variables + ['seasonality_ProphetM']
    control_features_3 = all_base_variables + ['seasonality_ProphetA']

    # Run feature selection for each control set
    selected_features_1, dropped_features_1 = feature_selection(
        data, target_variable, media_channels, organic_channels, control_features_1, mandatory_features=[]
    )
    selected_features_2, dropped_features_2 = feature_selection(
        data, target_variable, media_channels, organic_channels, control_features_2, mandatory_features=[]
    )
    selected_features_3, dropped_features_3 = feature_selection(
        data, target_variable, media_channels, organic_channels, control_features_3, mandatory_features=[]
    )

    # Return selected and dropped features based on seasonality type
    if seasonality_type == 'Bayesian':
        return selected_features_1, dropped_features_1
    elif seasonality_type == 'ProphetM':
        return selected_features_2, dropped_features_2
    else:
        return selected_features_3, dropped_features_3

# Example usage of the function
# Assuming `data`, `target_variable`, `media_channels`, `organic_channels`, and `all_base_variables` are defined

selected_features_B, dropped_features_B = main_variable_selection(
    data, target_variable, media_channels, organic_channels, all_base_variables, 'Bayesian'
)
selected_features_M, dropped_features_M = main_variable_selection(
    data, target_variable, media_channels, organic_channels, all_base_variables, 'ProphetM'
)
selected_features_A, dropped_features_A = main_variable_selection(
    data, target_variable, media_channels, organic_channels, all_base_variables, 'ProphetA'
)

# Print results
print("Selected Features for Bayesian:", selected_features_B)
print("Dropped Features for Bayesian:", [(feature, reason) for feature, reason in dropped_features_B])
print("Selected Features for ProphetM:", selected_features_M)
print("Dropped Features for ProphetM:", [(feature, reason) for feature, reason in dropped_features_M])
print("Selected Features for ProphetA:", selected_features_A)
print("Dropped Features for ProphetA:", [(feature, reason) for feature, reason in dropped_features_A])
