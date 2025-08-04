# Data Processing Guide

## Dataset Overview

This project uses multiple robotic navigation datasets for training neural world models. The datasets contain visual-inertial navigation trajectories with RGB images and corresponding trajectory data.

### Available Datasets

| Dataset | Trajectories | Description |
|---------|-------------|-------------|
| **recon** | 11,835 | Indoor navigation data from jackal robot |
| **navware** | 60 | Navigation scenarios with various robots |
| **sacson** | 2,955 | Structured environment navigation |
| **scand** | 615 | Scandinavian indoor environments |
| **tartan_drive** | 178 | Outdoor driving scenarios |

**Total**: 15,643 trajectories across all datasets

## Data Structure

### Directory Layout
```
data/
├── recon/
│   ├── jackal_2019-10-31-14-43-12_2_r06/
│   │   ├── 0.jpg, 1.jpg, ..., N.jpg    # RGB images (640x480)
│   │   └── traj_data.pkl               # Trajectory data
│   └── ...
├── sacson/
├── scand/
├── tartan_drive/
└── navware/
```

### Trajectory Data Format
Each `traj_data.pkl` contains:
- **position**: `(N, 2)` array - [x, y] coordinates in meters
- **yaw**: `(N,)` array - orientation in radians

### Data Splits
```
data_splits/
├── recon/
│   ├── train/traj_names.txt    # Training trajectories
│   └── test/traj_names.txt     # Test trajectories (2,367 samples)
├── sacson/
│   ├── train/traj_names.txt
│   └── test/traj_names.txt     # Test trajectories (591 samples)
└── ...
```

## Data Processing Pipeline

### 1. Image Processing
- **Input**: RGB images (640×480)
- **Output**: Preprocessed tensors (224×224)
- **Transforms**: Resize → ToTensor → Normalize
- **Normalization**: ImageNet stats (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

### 2. Trajectory Processing
- **Coordinate System**: Local coordinates relative to starting position
- **Action Computation**: Waypoint differences [Δx, Δy, Δyaw]
- **Metric Spacing**: Dataset-specific waypoint spacing (see config)

### 3. Data Normalization
Action statistics for normalization:
```yaml
action_stats:
  min: [-2.5, -4, -3.14]    # [min_dx, min_dy, min_dyaw]
  max: [5, 4, 3.14]         # [max_dx, max_dy, max_dyaw]
```

**Important**: All action dimensions (dx, dy, dyaw) are normalized to [-1, 1] range to ensure:
- ✅ **Balanced weights**: Prevent any single dimension from dominating loss calculations
- ✅ **Training stability**: Consistent gradient scales across all dimensions  
- ✅ **Numerical stability**: Avoid gradient explosion/vanishing issues
- ✅ **Diffusion model compatibility**: Standardized input distribution

### 4. Dataset Configuration
```yaml
recon:
  metric_waypoint_spacing: 0.25    # meters
scand:
  metric_waypoint_spacing: 0.38
tartan_drive:
  metric_waypoint_spacing: 0.72
go_stanford:
  metric_waypoint_spacing: 0.12
sacson:
  metric_waypoint_spacing: 0.255
```

## Key Functions

### Core Processing Functions (`misc.py`)
- `normalize_data(data, stats)`: Normalize action data to [-1, 1] range
- `unnormalize_data(ndata, stats)`: Convert back to original scale
- `to_local_coords(positions, origin, yaw)`: Transform to local coordinate system
- `angle_difference(ref_yaw, yaw_array)`: Compute relative yaw angles

### Dataset Class (`datasets.py`)
- `BaseDataset`: Main dataset class for loading and processing
- Supports variable trajectory lengths and context sizes
- Handles goal conditioning and action prediction

## Data Statistics

### Sample Trajectory Analysis
- **Trajectory Length**: 12-78 waypoints (varies by dataset)
- **Average Step Size**: 0.25-0.72 meters (dataset dependent)
- **Position Range**: Varies by environment scale
- **Yaw Range**: [-π, π] radians

### Processing Verification
✅ All datasets successfully loaded  
✅ Image preprocessing pipeline functional  
✅ Trajectory data parsing correct  
✅ Normalization functions working  
✅ Data splits properly configured  
✅ Batch processing verified  

## Usage

### Loading Data
```python
from datasets import BaseDataset
from torchvision import transforms

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                        std=[0.229, 0.224, 0.225])
])

dataset = BaseDataset(
    data_folder="/path/to/data/recon",
    data_split_folder="/path/to/data_splits/recon",
    dataset_name="recon",
    image_size=(224, 224),
    len_traj_pred=5,
    context_size=1,
    transform=transform,
    traj_names="train/traj_names.txt"
)
```

### Configuration Files
- `config/data_config.yaml`: Action normalization and dataset parameters
- `config/eval_config.yaml`: Evaluation settings and dataset selections

## Notes

- Image files are named sequentially (0.jpg, 1.jpg, ...)
- Trajectory data is synchronized with image timestamps
- Missing images or corrupted data are handled gracefully
- All coordinate transformations preserve metric accuracy
- Normalization supports both 2D (x,y) and 3D (x,y,yaw) action spaces
