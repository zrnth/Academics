import os
import shutil


def create_dataset_structure(root_path):
    # Define paths for the new dataset folders
    dataset_path = os.path.join("./", "dataset")
    dataset_0_path = os.path.join(dataset_path, "0")
    dataset_1_path = os.path.join(dataset_path, "1")

    # Create dataset directories if they don't exist
    os.makedirs(dataset_0_path, exist_ok=True)
    os.makedirs(dataset_1_path, exist_ok=True)

    # Traverse through the root folder to process patient folders
    for patient_id in os.listdir(root_path):
        patient_path = os.path.join(root_path, patient_id)
        print()

        if os.path.isdir(patient_path):  # Check if it's a directory
            for folder_name in ["0", "1"]:
                source_folder = os.path.join(patient_path, folder_name)

                if os.path.isdir(source_folder):  # Ensure folder exists
                    destination_folder = dataset_0_path if folder_name == "0" else dataset_1_path

                    for file_name in os.listdir(source_folder):
                        source_file = os.path.join(source_folder, file_name)
                        destination_file = os.path.join(destination_folder, file_name)

                        # Copy the file to the respective folder in the dataset
                        if os.path.isfile(source_file):
                            shutil.copy(source_file, destination_file)

    # Count and print the number of files in each dataset folder
    count_0 = len(os.listdir(dataset_0_path))
    count_1 = len(os.listdir(dataset_1_path))

    print(f"Number of files in '0': {count_0}")
    print(f"Number of files in '1': {count_1}")

# Example usage
root_path = "./1"  # Replace with your actual root directory path
create_dataset_structure(root_path)