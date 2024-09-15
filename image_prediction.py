import os
import pandas as pd
from predictor import predict
from src.utils import download_image

csv_path = '/Users/arsh/Downloads/student_resource 3/dataset/test.csv'
image_save_folder = '/Users/arsh/Downloads/student_resource 3/downloaded_images'
df = pd.read_csv(csv_path)



# Path to the folder with downloaded images
IMAGE_FOLDER = image_save_folder

# Path to the test CSV file
test_csv_path = '/Users/arsh/Downloads/student_resource 3/dataset/test.csv'

# Output file for predictions
output_csv_path = '/Users/arsh/Downloads/student_resource 3/test_out.csv'

# Read the test CSV
test_df = pd.read_csv(test_csv_path)

# Create a list to store predictions
predictions = []

# Loop through each row in the test dataset
count = 0
for index, row in test_df.iterrows():
    count+=1
    print(count)
    url = row['image_link']
    image_filename = url.split('/')[-1] #abcd.jpg
    image_path = os.path.join(IMAGE_FOLDER, image_filename)
    
    # Get the entity name (e.g., "item_weight")
    entity_name = row['entity_name']
    
    # Call the predict function to get the predicted value for this image and entity
    prediction = predict(image_path, entity_name)
    
    # Append the result to the predictions list
    predictions.append({
        'index': row['index'],
        'prediction': prediction
    })

# Create a DataFrame from the predictions list
predictions_df = pd.DataFrame(predictions)

# Save the predictions to a CSV file
predictions_df.to_csv(output_csv_path, index=False)

print(f"Predictions saved to {output_csv_path}")