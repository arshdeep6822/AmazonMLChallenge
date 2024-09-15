# src/predictor.py
import os
import pandas as pd
import pytesseract
import cv2
import re
from src.constants import entity_unit_map
from src.constants import allowed_units

def preprocess_image(image_path):
    """
    Preprocess the image to make text extraction easier.
    """
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
    return thresh

def extract_text_from_image(image_path):
    """
    Extract text from the image using OCR.
    """
    preprocessed_image = preprocess_image(image_path)
    extracted_text = pytesseract.image_to_string(preprocessed_image)
    return extracted_text


def extract_entity_value(extracted_text, entity_name):
    """
    Extract the value for the given entity (e.g., weight, dimensions, etc.)
    """
    allowed_units = entity_unit_map.get(entity_name, set())
    
    if entity_name in ["item_weight", "maximum_weight_recommendation"]:
        pattern = r'(\d+\.?\d*)\s?(gram|kilogram|microgram|milligram|ounce|pound|ton)'
    elif entity_name in ["width", "depth", "height"]:
        pattern = r'(\d+\.?\d*)\s?(centimetre|foot|inch|metre|millimetre|yard)'
    elif entity_name == "voltage":
        pattern = r'(\d+\.?\d*)\s?(volt|kilovolt|millivolt)'
    elif entity_name == "wattage":
        pattern = r'(\d+\.?\d*)\s?(watt|kilowatt)'
    elif entity_name == "item_volume":
        pattern = r'(\d+\.?\d*)\s?(cubic foot|cubic inch|cup|decilitre|centilitre|fluid ounce|gallon|imperial gallon|litre|microlitre|millilitre|pint|quart)'

    match = re.search(pattern, extracted_text, re.IGNORECASE)
    if match:
        value = match.group(1)
        unit = match.group(2)
        if unit.lower() in [u.lower() for u in allowed_units]:
            return f"{value} {unit}"
    return ""

def predict(image_path, entity_name):
    """
    Predict the entity value from the image.

    Parameters:
    - image_path: Path to the image file
    - entity_name: Name of the entity to predict (e.g., "item_weight", "width")

    Returns:
    - Predicted entity value (e.g., "10 gram", "15 cm")
    """
    extracted_text = extract_text_from_image(image_path)
    predicted_value = extract_entity_value(extracted_text, entity_name)
    return predicted_value
