import json
import re
import pdfplumber
import tabula
import pytesseract
from PIL import Image
import os
from pandas import DataFrame

# Set the full path to the Java executable (for tabula-py)
tabula.environment_info.java_path = r'C:\Program Files (x86)\Java\jre-1.8\bin\java.exe'

# Set the full path to the Tesseract executable (if not in PATH)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def extract_text_with_pdfplumber(pdf_file_path):
    # Extract text from the PDF file using pdfplumber
    text = ""
    with pdfplumber.open(pdf_file_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text()

    return text

def clean_up_text(text):
    # Remove excessive newlines and trailing whitespaces
    text = re.sub(r'\n+', '\n', text).strip()
    return text

def extract_info_from_pdf(pdf_file_path):
    # Initialize variables to store extracted information
    name = "NAME AND ADDRESS WITHHELD"
    company = None
    phone_number = None
    date_of_report = None

    # Extract text from the PDF file using pdfplumber
    with pdfplumber.open(pdf_file_path) as pdf:
        text = ""
        for page_number, page in enumerate(pdf.pages):
            text += page.extract_text()

            # Save each page as an image file
            image_path = f"page_{page_number + 1}.png"
            page.to_image(resolution=150).save(image_path)

            # Perform OCR on the image using pytesseract
            extracted_text = pytesseract.image_to_string(Image.open(image_path))
            #print(f"Page {page_number + 1} OCR Result:")
            #print(extracted_text)
            # Perform post-processing or save the OCR result as needed
            # Example: You can extract additional information from OCR result

            # Remove the temporary image file
            os.remove(image_path)

    # Use regex patterns to find Company, Phone, and Date of Report
    company_pattern = r"(?<=\n)[A-Za-z\s]+(?=\nHead Drug Safety Surveillance)"
    phone_pattern = r"Phone:\s*(\d{3}\s\d{3}\s\d{4})"
    date_pattern = r"\d{2}-(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)-\d{4}"

    # Search for Company, Phone, and Date of Report in the text
    company_match = re.search(company_pattern, text)
    phone_match = re.search(phone_pattern, text)
    date_match = re.search(date_pattern, text)

    # Extract information if found
    if company_match:
        company = company_match.group(0)
    if phone_match:
        phone_number = phone_match.group(0).replace("Phone:", "").strip()
    if date_match:
        date_of_report = date_match.group(0)

    # Create a dictionary to store the extracted information
    extracted_info = {
        'name': name,
        'company': company,
        'phone_number': phone_number,
        'date_of_report': date_of_report
    }

    # Step 1: Table Extraction
    # Use tabula-py to extract tables from the PDF
    tables = tabula.read_pdf(pdf_file_path, pages="all", multiple_tables=True)

    # Convert DataFrames to JSON serializable format
    json_tables = [table.to_dict(orient='records') for table in tables if isinstance(table, DataFrame)]

    # Add table data to the extracted_info dictionary
    extracted_info['tables'] = json_tables

    return extracted_info, text
# def text_to_json(text):
#     # Define regular expression patterns to extract key-value pairs
#     key_value_pattern = r"(?P<key>\w+):\s*(?P<value>.+)"
#     key_value_pairs = re.findall(key_value_pattern, text)

#     # Create a dictionary from the extracted key-value pairs
#     json_data = {}
#     for key, value in key_value_pairs:
#         json_data[key] = value

#     return json_data
def text_to_json(text):
    # Split the text into lines
    lines = text.split('\n')

    # Create a dictionary to store key-value pairs
    json_data = {}

    # Iterate through each line and create key-value pairs
    for line in lines:
        if ':' in line:
            key, value = map(str.strip, line.split(':', 1))
            json_data[key] = value

    return json_data



if __name__ == "__main__":
    # Replace 'your_pdf_file.pdf' with the actual path to your PDF file
    pdf_file_path = 'data2.pdf'

    # Extract text using pdfplumber
    extracted_text = extract_text_with_pdfplumber(pdf_file_path)
    cleaned_text = clean_up_text(extracted_text)

    #print("Extracted Text:")
    #print(cleaned_text)

    # Extract information from the PDF
    extracted_info, _ = extract_info_from_pdf(pdf_file_path)

    # Display the extracted information in JSON format in the terminal
    json_info_data = json.dumps(extracted_info, indent=2)
    #print(json_info_data)

    
    # Save the cleaned extracted text to a text file
    with open('extracted_text.txt', 'w', encoding='utf-8') as text_file:
        text_file.write(cleaned_text)
    text_file_path = 'extracted_text.txt'

    # Read the text from the file
    with open(text_file_path, 'r', encoding='utf-8') as text_file:
        text = text_file.read()

    # Convert the text to JSON format
    json_data = text_to_json(text)

    # Save the JSON data to a JSON file
    with open('extracted_data.json', 'w', encoding='utf-8') as json_file:
        json.dump(json_data, json_file, indent=2)       
