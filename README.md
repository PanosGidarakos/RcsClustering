# RcsClustering

A project for clustering and analyzing real atmospheric LiDAR data.

## Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/PanosGidarakos/RcsClustering.git
cd RcsClustering
```

### 2. Create a Virtual Environment
```bash
python3 -m venv venv
```

### 3. Activate the Virtual Environment
```bash
source venv/bin/activate
```

### 4. Install Dependencies
```bash
pip install -r requirements.txt
```

### 5. Run the Notebook
Open the notebook in VS Code or Jupyter:
```bash
jupyter notebook demo_notebook.ipynb
```

Or simply open `demo_notebook.ipynb` in VS Code and run the cells.

## Project Structure
- `demo_notebook.ipynb` - Main analysis notebook
- `utils/` - Helper functions for data reading and processing
- `lidar_data/` - LiDAR data files
- `groundtruth_data/` - Ground truth data
- `requirements.txt` - Python dependencies

## Usage
The notebook demonstrates how to read and process LiDAR data. Simply run the cells in order.
