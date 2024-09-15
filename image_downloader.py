import pandas as pd
import os
from src.utils import download_image

csv_path = '../dataset/test.csv'

df = pd.read_csv(csv_path)

image_save_folder = '../downloaded_images/'

if not os.path.exists(image_save_folder):
    os.makedirs(image_save_folder)
    
for index, row in df.iterrows():
    image_link = row['image_link']
    
    try:
        download_image(image_link, image_save_folder)  # Assuming download_images(url, save_location)
        print(f"Downloaded")
    except Exception as e:
        print(f"Error downloading {image_link}: {e}")
