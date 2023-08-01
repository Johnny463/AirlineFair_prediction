from flask import Flask, request, jsonify
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import io

# Load the pre-trained Keras model for image classification
model = load_model('model3.h5')

# Create a Flask application
app = Flask(__name__)

# Define a route "/Imageclassifier" to handle POST requests for image classification
@app.route('/Imageclassifier', methods=['POST'])
def recommend():
    try:
        # Check if the request contains the "image" part
        if 'image' not in request.files:
            return jsonify({'error': 'No image part in the request.'}), 400

        # Extract the image file from the request
        image_file = request.files['image']
        
        # Check if a file is selected
        if image_file.filename == '':
            return jsonify({'error': 'No selected file.'}), 400

        # Load and preprocess the image
        image_data = io.BytesIO(image_file.read())
        test_image = image.load_img(image_data, target_size=(64, 64))
        test_image = image.img_to_array(test_image)
        test_image = test_image / 255.0
        test_image = np.expand_dims(test_image, axis=0)
        
        # Make predictions using the pre-trained model
        result = model.predict(test_image)

        # Determine if the image contains a cat or a dog based on the prediction
        if result[0] <= 0.5:
            return jsonify({'result': 'Image is Cat'}), 200
        else:
            return jsonify({'result': 'Image is Dog'}), 200

    # Handle exceptions and return an error response with status code 500
    except Exception as e:
        return jsonify({'error': f'Error occurred: {str(e)}'}), 500

# Start the Flask application in debug mode if the script is directly executed
if __name__ == '__main__':
    app.run(debug=True)
