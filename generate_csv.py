import os
import csv

def generate_csv(dataset_dir, output_csv):
    """
    Generates a CSV file that maps image filenames to class labels.
    
    Args:
    - dataset_dir (str): Directory path where images are stored.
    - output_csv (str): Path where the CSV file will be saved.
    """
    with open(output_csv, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Filename', 'ClassId'])  # Writing the header

        # Loop through all class folders (subdirectories)
        for class_id in os.listdir(dataset_dir):
            class_dir = os.path.join(dataset_dir, class_id)
            if os.path.isdir(class_dir):
                # Loop through all image files in the class folder
                for img_file in os.listdir(class_dir):
                    if img_file.endswith('.ppm'):  # Only process .ppm images
                        # Write the relative path to the image and the class label (folder name)
                        writer.writerow([os.path.join(class_id, img_file), int(class_id)])

    print(f"CSV file generated and saved at: {output_csv}")

# Specify the path to the Training folder and the output CSV file
dataset_dir = r"C:\Users\gyane\Desktop\road-sign-detection-ai\dataset\GTSRB\Training"
output_csv = r"C:\Users\gyane\Desktop\road-sign-detection-ai\dataset\GT-final_train.csv"

# Generate the CSV
generate_csv(dataset_dir, output_csv)
