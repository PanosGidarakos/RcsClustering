
import pandas as pd
import os

# Get the project root directory (parent of utils/)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def read_lidar_data(path: str) -> pd.DataFrame:
    """Reads and processes LiDAR data from an Excel file."""
    temp = pd.read_excel(path)
    temp = temp.iloc[:, 2:-1]
    temp.index = pd.read_excel(os.path.join(PROJECT_ROOT, 'lidar_data/height.xlsx'), header=None)[0]
    temp.columns = pd.read_excel(os.path.join(PROJECT_ROOT, 'lidar_data/channels.xlsx'), header=None)[0]
    temp = temp.iloc[:, ::-1]
    temp.index.name = 'heights'

    return temp

    

def groundtruth(path: str) -> pd.DataFrame:
    """Reads and processes ground truth data from an Excel file."""
    
    pollen_data = pd.read_excel(path, header=None)
    pollen_data.set_index(pollen_data[0], inplace=True, drop=True)
    pollen_data = pollen_data.drop([0], axis=1)
    pollen_data = pollen_data.iloc[:, ::-1]

    return pollen_data