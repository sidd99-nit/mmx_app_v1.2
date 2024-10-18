from flask import Flask, request, jsonify, render_template,url_for,session, send_from_directory
from werkzeug.utils import secure_filename
from flask_session import Session
import pandas as pd
import os
import json
from datetime import timedelta
import plotly.graph_objects as go
import mpld3
import time


from routes.preqc.modal2 import create_plot
from routes.preqc.modal4 import create_plot_sales
from routes.preqc.modal3_bayesian import seasonality_bayesian
from routes.preqc.modal3_prophetA import seasonality_prophetA
from routes.preqc.modal3_prophetM import seasonality_prophetM
from routes.helper_modules.add_holidays import add_holidays
from routes.preqc.variable_selection import main_variable_selection 

from routes.models.LWMMM_Base import run_lwmmm_base
from routes.models.LWMMM_Custom import run_lwmmm_custom
from routes.models.Bayesian_Base import bayesian_base
from routes.models.Bayesian_Custom import bayesian_custom

import matplotlib
matplotlib.use('Agg')

app = Flask(__name__)

app.secret_key = 'supersecretkey'

app.config['SESSION_TYPE'] = 'filesystem'
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(days=1)

Session(app)

@app.route('/')
def welcome():
    return render_template('welcome.html')

@app.route('/files_upload')
def files_upload():

    # having loading effect
    # time.sleep(1.8)
    # Clear all session variables
    session.clear()

    return render_template('files_upload.html')

@app.route('/select_features')
def select_features():
    return render_template('select_features.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    # Check if the request contains both files
    if 'file_1' not in request.files or 'file_2' not in request.files:
        return jsonify({"error": "Both files are required"}), 400
    
    # Save the first file
    file_1 = request.files['file_1']
    file_1_name = secure_filename(file_1.filename)
    file_1_path = os.path.join('static','uploads', file_1_name)
    file_1.save(file_1_path)

    # Save the second file
    file_2 = request.files['file_2']
    file_2_name = secure_filename(file_2.filename)
    file_2_path = os.path.join('static','uploads', file_2_name)
    file_2.save(file_2_path)

    # Create an HTML file that will be shown in the iframe
    html_content = f"""
    <html>
    <head><title>File Upload Results</title></head>
    <body>
        <h1 style="colour:white">Uploaded Files</h1>
        <p style="colour:white">File 1: {file_1_name}</p>
        <p style="colour:white">File 2: {file_2_name}</p>
    </body>
    </html>
    """
    html_file_name = "file_display.html"
    html_file_path = os.path.join('static','reports', html_file_name)

    # Save the HTML content to the file
    with open(html_file_path, 'w') as html_file:
        html_file.write(html_content)

    # Return the file path to be used in the iframe
    return jsonify({"file_path": f"/get_html/{html_file_name}"}), 200

# route to serve the generated report HTML file
def get_html(filename):
    try:
        return send_from_directory(os.path.join('static','reports'),filename)    
    except FileNotFoundError:
        print("html prop")
        return jsonify({"error":"File not found"}), 404

@app.route('/pre-qc', methods=['POST'])
def save_data():
    file_1 = request.files['file_1']
    file_2 = request.files['file_2']
    data = request.form['data']

    print(file_1.filename)
    print(file_2.filename)

    # Save the file
    file_path_1 = os.path.join('static','uploads', file_1.filename)
    file_path_2 = os.path.join('static','uploads', file_2.filename)

    try:
        file_1.save(file_path_1)
        file_2.save(file_path_2)

        print("Media and Sales files are save successfully")
    except Exception as ex:
        print(f"Error saving Media and Sales files: {ex}")

    # Open file
    sales_df = pd.read_excel(os.path.join('static','uploads', file_1.filename))
    media_df = pd.read_excel(os.path.join('static','uploads', file_2.filename))

    combined_data = pd.merge(sales_df,media_df,on='TIME',how='inner')

    # Add holidays to the data
    combined_data = add_holidays(data=combined_data)

    combined_data.to_excel(os.path.join('static','uploads','combined_data.xlsx'))

    # Process the JSON data
    data_dict = json.loads(data)
    # Do something with the data

    # store the chosen variables in session
    session['user_options'] = data_dict

    print(data_dict)

     # Define the URL to redirect to
    redirect_url = url_for('preqc_report')  # Replace 'new_route' with the name of the route you want to redirect to

    # Return JSON with a message and a URL
    return jsonify({"message": "Data saved successfully!", "redirect_url": redirect_url}), 200


@app.route('/preqc_report')
def preqc_report():

    # reading the combined data including sales and media info
    combined_data = pd.read_excel(os.path.join('static','uploads','combined_data.xlsx'))

    # fetching the options chosen by user in dictionary form
    user_options_dict = session['user_options']

    # checking if user options are being fetched properly
    if(user_options_dict):
        print("session working fine and extracted user_options_dict")

    # extracting the media channels from dictionary in list form
    media_channels = user_options_dict['marketingVariables']
    
    # extracting the organic channels from dictionary in list form
    organic_channels = user_options_dict['organicVariables']

    # extracting the base variables from the dictionary in list form (used for variable selection process)
    all_base_variables = user_options_dict['mandatoryBaseVariables']

    # combing the media and organic channel lists to send to front end
    all_channels = media_channels + organic_channels 

    # fetching the target variable
    target_variable = user_options_dict['targetVariable'][0]

    # save seasonalities plots which also has seasonality data appended to the original data
    img_path_B,data_B = seasonality_bayesian(combined_data)
    img_path_A,data_A = seasonality_prophetA(combined_data, target=target_variable)
    img_path_M,data_M = seasonality_prophetM(combined_data, target=target_variable)

    # testing
    print("inside of data_B",data_M)
    # testing

    folder_name_seasonality = 'seasonality_data'
    if not os.path.exists(folder_name_seasonality):
        os.makedirs(folder_name_seasonality)
    
    # saved data with seasonality data appended
    data_B.to_excel(os.path.join(folder_name_seasonality,'data_B.xlsx'),index = False)
    data_A.to_excel(os.path.join(folder_name_seasonality,'data_A.xlsx'),index = False)
    data_M.to_excel(os.path.join(folder_name_seasonality,'data_M.xlsx'),index = False)

    # Running variable selection process
    control_features_B , dropped_features_B = main_variable_selection(data_B, target_variable, media_channels , organic_channels, all_base_variables, seasonality_type = 'Bayesian')
    control_features_A , dropped_features_A = main_variable_selection(data_A, target_variable, media_channels , organic_channels, all_base_variables, seasonality_type = 'ProphetA')
    control_features_M , dropped_features_M = main_variable_selection(data_M, target_variable, media_channels , organic_channels, all_base_variables, seasonality_type = 'ProphetM')

    # testing
    print("printing control ones",control_features_A,dropped_features_A)
    # testing
    
    # putting them in session variables
    session['control_features_A'] = control_features_A
    session['dropped_features_A'] = dropped_features_A
    session['control_features_B'] = control_features_B
    session['dropped_features_B'] = dropped_features_B
    session['control_features_M'] = control_features_M
    session['dropped_features_M'] = dropped_features_M

    # printing taget variable for cross checking
    print(target_variable)

    # For modal-2 , making media and organic plots
    fig2 = create_plot(combined_data,all_channels[0])

    # For modal-4 , making target variable i.e sales plot
    fig4 = create_plot_sales(combined_data,target_variable)


    # For modal-3 , seasonality plots of all type
    seasonalities = ['Bayesian','Prophet Additive','Prophet Multiplicative']
    fig3 = os.path.join('static', 'images', 'seasonality_Bayesian.png')


    # For modal-1 , to show the selected and dropped features 
    options_var_selection = ['Bayesian Seasonality','Prophet Multiplicative Seasonality','Prophet Additive Seasonality']
    # Create a single dictionary to hold 'selected' and 'dropped'
    variables_table = {"selected": control_features_B, "dropped": dropped_features_B}

    print(variables_table)

    return render_template('preqc_report.html', channel_names = all_channels, chart2 = fig2.to_html(full_html=False), chart4 = fig4.to_html(full_html=False), variables_table=variables_table,seasonalities = seasonalities ,chart3 = fig3, options_var_selection=options_var_selection)

# Route to update the Media Spend Distribution plot (Modal 2)
@app.route('/update_plot')
def update_plot():
    # Read the combined data including sales and media info
    combined_data = pd.read_excel(os.path.join('static', 'uploads', 'combined_data.xlsx'))

    # Get the selected column (channel) from the dropdown
    selected_channel = request.args.get('channel')

    # Create the Plotly figure based on the selected channel
    fig = create_plot(combined_data, selected_channel)

    # Convert the Plotly figure to JSON and return
    return fig.to_json()


#for saving the varibales that got dropped from variable selection but user selected them back
@app.route('/save_selected_variables', methods=['POST'])
def save_selected_variables():
    data = request.get_json()
    selected_option = data.get('selected_option')
    selected_variables = data.get('selected_variables')
    dropped_variables = data.get('dropped_variables')

    # Process and save the selected option and variables (customize this logic as needed)
    print("Selected Option:", selected_option)
    print("Selected Variables:", selected_variables)
    print("Dropped Variables:", dropped_variables)

    # save them in session variables as they'll be used in modelling
    session['selected_option_seasonality'] = selected_option
    session['selected_variables'] = selected_variables
    session['dropped_variables_to_include'] = dropped_variables

    # Simulate saving the data
    try:
        # Add logic here to save the data to a database or process it as needed
        return jsonify({"success": True , "redirect_url" : url_for('modelling')})
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"success": False})

#for changing the feature selection table according to seasonality
@app.route('/change_feature_table', methods = ['POST'])
def change_feature_table():
    data = request.get_json()
    selected_seasonality = data.get('selected_option')

    # extracting the dropped and selected features
    control_features_A = session['control_features_A']
    dropped_features_A = session['dropped_features_A']  
    control_features_B = session['control_features_B'] 
    dropped_features_B = session['dropped_features_B'] 
    control_features_M = session['control_features_M'] 
    dropped_features_M = session['dropped_features_M']  

    if selected_seasonality == 'Bayesian Seasonality':
        variables_table = {"selected": control_features_B, "dropped": dropped_features_B}
        print(selected_seasonality,variables_table)
    elif selected_seasonality ==  'Prophet Additive Seasonality':
        variables_table = {"selected": control_features_A, "dropped": dropped_features_A}
        print(selected_seasonality,variables_table)
    elif selected_seasonality == 'Prophet Multiplicative Seasonality':
        variables_table = {"selected": control_features_M, "dropped": dropped_features_M}
        print(selected_seasonality,variables_table)
    else:
        variables_table = {"selected": [], "dropped": []}
        print(selected_seasonality,variables_table)

    return jsonify({"variables_table":variables_table})


# Route to update the Seasonality plot (Modal 3)
@app.route('/update_seasonality')
def update_seasonality():
    # Fetch the user options stored in session
    user_options_dict = session.get('user_options', {})

    # Read the combined data including sales and media info
    combined_data = pd.read_excel(os.path.join('static', 'uploads', 'combined_data.xlsx'))

    # Extract the target column from the user options
    target = user_options_dict.get('targetVariable', [None])[0]

    # Get the selected seasonality type from the dropdown
    selected_seasonality = request.args.get('seasonality')

    if selected_seasonality == 'Bayesian':
        img_path = os.path.join('static', 'images', 'seasonality_Bayesian.png')
    elif selected_seasonality == 'Prophet Multiplicative':
        img_path = os.path.join('static', 'images', 'seasonality_prophetM.png')
    else:
        img_path = os.path.join('static', 'images', 'seasonality_prophetA.png')

    # Return the image path in JSON response
    return jsonify({'image_path': img_path})

# Modelling options
@app.route('/modelling')
def modelling():

    #Extract the variables selected
    seasonality_type = session['selected_option_seasonality'] 
    control_variables = session['selected_variables'] 
    mandatory_control_varibales = session['dropped_variables_to_include']

    #Extract media variables
    user_options_dict = session['user_options']
    # extracting the media channels from dictionary in list form
    media_channels = user_options_dict['marketingVariables']
    # extracting the organic channels from dictionary in list form
    organic_channels = user_options_dict['organicVariables']
    # fetching the target variable
    target_variable = user_options_dict['targetVariable'][0]

    # Creating final variables
    media_channels = media_channels + organic_channels
    session['media_channels_final'] = media_channels

    base_variables = control_variables + mandatory_control_varibales
    session['base_variables_final'] = base_variables

    target_variable = target_variable
    session['target_variable_final'] = target_variable

    return render_template("modelling.html",media_channels=media_channels, base_variables=base_variables, target_variable=target_variable)

# Actually running Light Weight MMM Base Model
@app.route('/run_model', methods=['POST'])
def run_model():
    data = request.get_json()

    # Extract the variables
    media_channels = session['media_channels_final']
    base_variables = session['base_variables_final']
    target_variable = session['target_variable_final']

    # Checking seasonality type
    seasonality_type_final = session['selected_option_seasonality']

    # converting target_variable to list
    target_variable = [target_variable]

    # accordingly selecting data i.e seasonality
    if seasonality_type_final == 'Bayesian Seasonality':
        data_path = os.path.join("seasonality_data","data_B.xlsx")
    elif seasonality_type_final == 'Prophet Multiplicative Seasonality':
        data_path = os.path.join("seasonality_data","data_M.xlsx")
    else:
        data_path = os.path.join("seasonality_data","data_A.xlsx")

    # Checking all variables once
    print("media_channels :",media_channels)
    print("base_variables :",base_variables) 
    print("target_variable :",target_variable) 

    # Running the Base LWMMM Model
    roi_df = run_lwmmm_base(media_channels,base_variables,target_variable,data_path)

    ### ROI data ###

    roi_df = roi_df.applymap(lambda x: round(x,2) if isinstance(x,(float,int)) else x )

    # transposing ROI
    roi_df_transposed = roi_df.T

    # Rename columns to 'feature' and 'value'
    roi_df_transposed.columns = ['value']
    roi_df_transposed['feature'] = roi_df_transposed.index

    # Reorder columns to match the desired structure
    roi_df_transposed = roi_df_transposed[['feature', 'value']]

    # Convert the DataFrame to a list of dictionaries
    table = roi_df_transposed.to_dict(orient='records')
    ### ROI data ###

    # Run your machine learning model here
    # For the example, we assume we get the following:
    images = {
        'model_fit': '/static/images/LWMMM_Base/model_fit.png',
        'roi_hat': '/static/images/LWMMM_Base/roi_bar_chart.png',
        'media_attribution': '/static/images/LWMMM_Base/media_baseline_contribution.png',
        'roi_plot': '/static/images/LWMMM_Base/roi_plot.png'
    }

    # table = [
    #     {'feature': 'Feature 1', 'value': 'Value 1'},
    #     {'feature': 'Feature 2', 'value': 'Value 2'}
    # ]

    # Return the result as JSON
    return jsonify({'images': images, 'table': table})

# Running LightWeight with custom priors
@app.route('/lightweight_custom_prior', methods=['POST'])
def lightweight_custom_prior():
    data = request.get_json()

    # Extract the variables
    media_channels = session['media_channels_final']
    base_variables = session['base_variables_final']
    target_variable = session['target_variable_final']
    scale =  (data.get('scale'))
    concentration1 = data.get('concentration1')
    concentration0 =  data.get('concentration0')
    concentration =  data.get('concentration')
    rate =  data.get('rate')

    print("checkinggggggggggg",scale,concentration1,concentration0,concentration,rate)

    # Checking seasonality type
    seasonality_type_final = session['selected_option_seasonality']

    # converting target_variable to list
    target_variable = [target_variable]

    # accordingly selecting data i.e seasonality
    if seasonality_type_final == 'Bayesian Seasonality':
        data_path = os.path.join("seasonality_data","data_B.xlsx")
    elif seasonality_type_final == 'Prophet Multiplicative Seasonality':
        data_path = os.path.join("seasonality_data","data_M.xlsx")
    else:
        data_path = os.path.join("seasonality_data","data_A.xlsx")

    # Checking all variables once
    print("media_channels :",media_channels)
    print("base_variables :",base_variables) 
    print("target_variable :",target_variable) 

    # Running the Custom LWMMM Model
    roi_df = run_lwmmm_custom(media_channels,base_variables,target_variable,data_path, scale , concentration1 , concentration0, concentration , rate)

    ### ROI data ###

    roi_df = roi_df.applymap(lambda x: round(x,2) if isinstance(x,(float,int)) else x )

    # transposing ROI
    roi_df_transposed = roi_df.T

    # Rename columns to 'feature' and 'value'
    roi_df_transposed.columns = ['value']
    roi_df_transposed['feature'] = roi_df_transposed.index

    # Reorder columns to match the desired structure
    roi_df_transposed = roi_df_transposed[['feature', 'value']]

    # Convert the DataFrame to a list of dictionaries
    table = roi_df_transposed.to_dict(orient='records')
    ### ROI data ###

    # Run your machine learning model here
    # For the example, we assume we get the following:
    images = {
        'model_fit': '/static/images/LWMMM_Custom/model_fit.png',
        'roi_hat': '/static/images/LWMMM_Custom/roi_bar_chart.png',
        'media_attribution': '/static/images/LWMMM_Custom/media_baseline_contribution.png',
        'roi_plot': '/static/images/LWMMM_Custom/roi_plot.png'
    }

    # table = [
    #     {'feature': 'Feature 1', 'value': 'Value 1'},
    #     {'feature': 'Feature 2', 'value': 'Value 2'}
    # ]

    # Return the result as JSON
    return jsonify({'images': images, 'table': table})

# Actually running Bayesian Base Model
@app.route('/run_bayesian_base', methods=['POST'])
def run_bayesian_base():
    data = request.get_json()

    # Extract the variables
    media_channels = session['media_channels_final']
    base_variables = session['base_variables_final']
    target_variable = session['target_variable_final']

    # Checking seasonality type
    seasonality_type_final = session['selected_option_seasonality']

    # converting target_variable to list
    target_variable = [target_variable]

    # accordingly selecting data i.e seasonality
    if seasonality_type_final == 'Bayesian Seasonality':
        data_path = os.path.join("seasonality_data","data_B.xlsx")
    elif seasonality_type_final == 'Prophet Multiplicative Seasonality':
        data_path = os.path.join("seasonality_data","data_M.xlsx")
    else:
        data_path = os.path.join("seasonality_data","data_A.xlsx")

    # Checking all variables once
    print("media_channels :",media_channels)
    print("base_variables :",base_variables) 
    print("target_variable :",target_variable) 

    # Running the Base LWMMM Model
    roi_df = bayesian_base(media_channels,base_variables,target_variable,data_path)
    roi_df['ROI'] = roi_df['ROI'].round(2)
    roi_df.columns = ['feature','value']

    ### ROI data ###

    # Convert the DataFrame to a list of dictionaries
    table = roi_df.to_dict(orient='records')
    ### ROI data ###

    # Run your machine learning model here
    # For the example, we assume we get the following:
    images = {
        'roi_plot': '/static/images/Bayesian_Base/roi_plot.png'
    }

    # table = [
    #     {'feature': 'Feature 1', 'value': 'Value 1'},
    #     {'feature': 'Feature 2', 'value': 'Value 2'}
    # ]

    # Return the result as JSON
    return jsonify({'images': images, 'table': table})
    # return jsonify({'table': table})

# Running LightWeight with custom priors
@app.route('/bayesian_final_iteration', methods=['POST'])
def bayesian_final_iteration():
    data = request.get_json()

    # Extract the variables
    media_channels = session['media_channels_final']
    base_variables = session['base_variables_final']
    target_variable = session['target_variable_final']
    runs =  data.get('runs')

    print("checkinggggggggggg",runs)

    # Checking seasonality type
    seasonality_type_final = session['selected_option_seasonality']

    # converting target_variable to list
    target_variable = [target_variable]

    # accordingly selecting data i.e seasonality
    if seasonality_type_final == 'Bayesian Seasonality':
        data_path = os.path.join("seasonality_data","data_B.xlsx")
    elif seasonality_type_final == 'Prophet Multiplicative Seasonality':
        data_path = os.path.join("seasonality_data","data_M.xlsx")
    else:
        data_path = os.path.join("seasonality_data","data_A.xlsx")

    # Checking all variables once
    print("media_channels :",media_channels)
    print("base_variables :",base_variables) 
    print("target_variable :",target_variable) 

    # Running the Custom LWMMM Model
    roi_df = bayesian_custom(media_channels,base_variables,target_variable,data_path,runs)
    roi_df['ROI'] = roi_df['ROI'].round(2)
    roi_df.columns = ['feature','value']

    ### ROI data ###

    # Convert the DataFrame to a list of dictionaries
    table = roi_df.to_dict(orient='records')
    ### ROI data ###

    # Run your machine learning model here
    # For the example, we assume we get the following:
    images = {
        'roi_plot': '/static/images/Bayesian_Custom/roi_plot.png'
    }

    # table = [
    #     {'feature': 'Feature 1', 'value': 'Value 1'},
    #     {'feature': 'Feature 2', 'value': 'Value 2'}
    # ]

    # Return the result as JSON
    return jsonify({'images': images, 'table': table})
    # return jsonify({'table': table})

if __name__ == '__main__':
    app.run(debug=True)

    
