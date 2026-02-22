import requests
import zipfile
from pathlib import Path
import os



def retrieve_data():
    """
    Retrieves a zip file containing only images of pizza, steak, and sushi
    """
    
    #Setup path to a folder
    data_path = Path("Data/")


    #Creating data folder
    if data_path.is_dir():
        print(f"{data_path} directory already exists, skipping download")
    else:
        print(f"{data_path} does not exist, creating directory")
        data_path.mkdir(parents=True, exist_ok=True)


        #Downloading Data Zip file, wb means write binary
        with open(data_path/"pizza_steak_sushi.zip","wb") as a:
            request = requests.get("https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi.zip")
            print("Downloading image data...")
            a.write(request.content)
        
        
        #Unzipping Files in Zip
        with zipfile.ZipFile(data_path/"pizza_steak_sushi.zip", "r") as images:
            print("Unzipping Image Data")
            images.extractall(data_path)


        # Delete the zip file
        zip_file_path = data_path / "pizza_steak_sushi.zip"
        try:
            os.remove(zip_file_path)
            print(f"{zip_file_path} deleted successfully.")
        except OSError:
            print(f"Error: OSError")