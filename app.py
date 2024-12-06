from flask import Flask, request, jsonify, session, send_from_directory, abort
import os
import shutil
import json
from flask_cors import CORS
from flask_session import Session
from datetime import timedelta


# Import helper modules or route modules
from routes.Gemini import get_gemini_response

app = Flask(__name__)
app.secret_key = 'supersecretkey'
CORS(app)

app.config['SESSION_TYPE'] = 'filesystem'
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(days=1)
Session(app)

# Directory configurations
UPLOAD_FOLDER = './uploads'
OUTPUT_JSONS = './output_jsons'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_JSONS, exist_ok=True)

app.config.update({
    'UPLOAD_FOLDER': UPLOAD_FOLDER,
    'OUTPUT_JSONS': OUTPUT_JSONS,
    'SESSION_TYPE': 'filesystem',
})

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}

# Global variables
approved_prods = set()
pending_prods = []


def allowed_file(filename):
    """Check if a file is an allowed image format."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def segregate_images(filenames):
    """
    Segregate image filenames into a dictionary based on the text
    before the last underscore and number.
    """
    segregated_dict = {}
    for file in filenames:
        key_part = '_'.join(file.rsplit('_', 1)[0:-1]) if "_" in file else file.rsplit('.', 1)[0]
        segregated_dict.setdefault(key_part, []).append(file)
    return segregated_dict


def get_first_values(data_dict):
    """Retrieve the first value of each list in the dictionary."""
    try:
        return [values[0] for values in data_dict.values() if values]
    except Exception as ex:
        print("Error in retrieving first image of each product:", ex)
        return []


@app.route('/upload', methods=['POST', 'GET'])
def upload_files():
    session.clear()

    if 'files' not in request.files:
        return jsonify({'error': 'No files part in the request'}), 400

    files = request.files.getlist('files')
    saved_files = []

    for file in files:
        if file.filename:
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)
            saved_files.append(file.filename)

    all_segregated_files = segregate_images(saved_files)
    all_products_first_image = get_first_values(all_segregated_files)

    # Store data in session
    session['saved_files']= saved_files
    session['segregated_files']= all_segregated_files
    session['all_products_first_image']= all_products_first_image

    return jsonify({
        'message': 'Files successfully uploaded',
        'files': saved_files,
        'product_thumbnail_images': all_products_first_image,
        'all_segregated_files': all_segregated_files
    }), 200


@app.route('/thumbnails/<filename>', methods=['POST', 'GET'])
def uploaded_file(filename):
    try:
        return send_from_directory(app.config['UPLOAD_FOLDER'], filename)
    except FileNotFoundError:
        abort(404)


@app.route('/first_info', methods=['GET', 'POST'])
def first_info():
    try:
        incoming_data = request.get_json()
        if not incoming_data:
            return jsonify({'error': 'No JSON data received'}), 400
        else:
            print(incoming_data.get('data'))

        all_products_first_image = session.get('all_products_first_image')
        segregated_files = session.get('segregated_files')

        return jsonify({
            'product_thumbnail_file_names': all_products_first_image,
            'all_segregated_files': segregated_files
        }), 200

    except Exception as ex:
        print('Error in product thumbnail images:', ex)
        return jsonify({'error': str(ex)}), 500


@app.route('/pending_approved', methods=['POST'])
def product_thumbnail():
    try:
        incoming_data = request.get_json()
        
        approved_prods.add('HARIBO_CANDY_1.jpg')  # Example placeholder

        print('INCOMING DATA',incoming_data.get('data'))

        return jsonify({
            'message': 'Hello boy'
        }), 200

    except Exception as ex:
        print('Error in product thumbnail images:', ex)
        return jsonify({'error': str(ex)}), 500


@app.route('/complete_info', methods=['POST', 'GET'])
def complete_info():
    # Example data for testing
    all_products_images = {
        'HARIBO_CANDY': ['HARIBO_CANDY_1.jpg', 'HARIBO_CANDY_2.jpg'],
        'rocher': ['rocher_1.png', 'rocher_2.png']
    }

    for key, images in all_products_images.items():
        try:
            df = get_gemini_response(images, os.path.join(os.getcwd(), "uploads"))
            json_data = df.to_dict(orient='records')[0]

            json_file_path = os.path.join(app.config['OUTPUT_JSONS'], f"{key}.json")
            with open(json_file_path, 'w') as json_file:
                json.dump(json_data, json_file, indent=4)

            print(f"Saved JSON for key: {key} at {json_file_path}")
        except Exception as ex:
            print(f"Error processing images for key {key}:", ex)

    return jsonify({'message': 'Processing complete'}), 200


if __name__ == '__main__':
    app.run(debug=True)
