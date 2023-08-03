import json
import pdfplumber
import pytesseract
from PIL import Image
import os
import pandas as pd
import tabula

# Set the full path to the Tesseract executable (if not in PATH)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def extract_text_with_pdfplumber(pdf_file_path):
    # Extract text from the PDF file using pdfplumber
    text = ""
    with pdfplumber.open(pdf_file_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text()

    return text

def extract_text_with_ocr(image_path):
    # Perform OCR on the image using Tesseract
    image = Image.open(image_path)
    extracted_text = pytesseract.image_to_string(image)
    return extracted_text

def extract_info_from_pdf(pdf_file_path):
    # Initialize variables to store extracted information
    extracted_data = {
        'pages': []
    }

    # Extract text from the PDF file using pdfplumber
    with pdfplumber.open(pdf_file_path) as pdf:
        for page_number, page in enumerate(pdf.pages):
            page_data = {
                'page_number': page_number + 1,
                'text': page.extract_text(),
                'tables': []
            }

            # Save each page as an image file
            image_path = f"page_{page_number + 1}.png"
            page.to_image(resolution=150).save(image_path)

            # Perform OCR on the image using pytesseract
            ocr_result = extract_text_with_ocr(image_path)
            page_data['ocr_result'] = ocr_result
            # Perform post-processing or save the OCR result as needed
            # Example: You can extract additional information from OCR result

            # Remove the temporary image file
            os.remove(image_path)

            # Extract tables from the page using tabula-py
            tables = []
            try:
                tables = tabula.read_pdf(pdf_file_path, pages=page_number + 1, multiple_tables=True)
            except:
                pass

            if tables:
                # Convert DataFrames to a list of dictionaries
                json_tables = [table.to_dict(orient='records') for table in tables if isinstance(table, pd.DataFrame)]
                page_data['tables'] = json_tables

            extracted_data['pages'].append(page_data)

    return extracted_data

if __name__ == "__main__":
    # Replace 'your_pdf_file.pdf' with the actual path to your PDF file
    pdf_file_path = 'data.pdf'

    # Extract text using pdfplumber and OCR
    extracted_data = extract_info_from_pdf(pdf_file_path)

    # Save the extracted data to a JSON file
    with open('extracted_data1.json', 'w', encoding='utf-8') as json_file:
        json.dump(extracted_data, json_file, indent=2)
