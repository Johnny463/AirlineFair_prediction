import os
from flask import Flask, request, jsonify
import numpy as np
from PIL import Image
import cv2
app = Flask(__name__)

def save_image(image_file, filename):
    # Get the current working directory (where the Python file is located)
    base_directory = os.getcwd()

    # Create the directory for saving images if it does not exist
    image_directory = os.path.join(base_directory, "images")
    os.makedirs(image_directory, exist_ok=True)

    # Save the image to the specified filename in the images directory
    file_path = os.path.join(image_directory, filename)
    image_file.save(file_path)
def greyscale(image_file, filename):
    
    image = Image.open(image_file)

    # Convert the PIL image to a NumPy array
    image_array = np.array(image)

    # Convert the image to grayscale+
    gray_image = np.dot(image_array[..., :3], [0.2989, 0.5870, 0.1140])

    # Convert the values to integers between 0 and 255 (8-bit grayscale)
    gray_image = np.round(gray_image).astype(np.uint8)
    
    
    pil_image = Image.fromarray(gray_image)
    
    
    base_directory = os.getcwd()
    image_directory = os.path.join(base_directory, "images")
    os.makedirs(image_directory, exist_ok=True)
    file_path = os.path.join(image_directory, filename)
    pil_image.save(file_path)
def cartoon_effect(image_array):
    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)

    # Apply bilateral filter to smoothen the image while preserving edges
    smoothed_image = cv2.bilateralFilter(gray_image, d=9, sigmaColor=75, sigmaSpace=75)

    # Detect edges using adaptive thresholding
    edges = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 9)

    # Combine edges with the smoothed image to get the cartoon effect
    cartoon_image = cv2.bitwise_and(smoothed_image, smoothed_image, mask=edges)

    # Convert the values to integers between 0 and 255 (8-bit grayscale)
    cartoon_image = np.round(cartoon_image).astype(np.uint8)

    return cartoon_image

@app.route('/upload_image', methods=['POST'])
def upload_image(): 
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image part in the request.'}), 400

        image_file = request.files['image']
        if image_file.filename == '':
            return jsonify({'error': 'No selected file.'}), 400

        filename = image_file.filename

        # Save the original image
        save_image(image_file, filename)

        # Convert the received image to a NumPy array using PIL
        image = Image.open(image_file)
        image_array = np.array(image)

        # Apply the cartoon effect
        cartoon_image = cartoon_effect(image_array)

        # Save the cartoon image with a prefix 'cartoon_'
        cartoon_filename = 'cartoon_' + filename
        save_image(Image.fromarray(cartoon_image), cartoon_filename)

        return jsonify({'message': 'Image , GreyScale and Cartoon successfully saved.'}), 200
        
    except Exception as e:
        return jsonify({'error': f'Error occurred: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True)
