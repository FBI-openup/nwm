# WorldMem Installation and Test Suite

## Overview

This comprehensive Python script provides automated installation, configuration, and testing for the WorldMem (World Memory) components of the Navigation World Models (NWM) project. It handles dependency management, environment validation, and functional testing in a single, easy-to-use script.

## Features

### 🚀 Installation Features
- **Automatic Environment Detection**: Detects workspace structure and Python environment
- **Dependency Management**: Installs all required Python packages from `requirements.txt`
- **Conda Integration**: Handles conda/mamba specific dependencies (ffmpeg, etc.)
- **Cache Cleanup**: Removes all Python cache files (`__pycache__`, `.pyc`, `.pyo`)
- **Path Management**: Automatically configures Python paths for WorldMem modules

### 🧪 Testing Features
- **Import Testing**: Verifies all required packages can be imported
- **Algorithm Testing**: Tests WorldMem core algorithms (`WorldMemMinecraft`, `PosePrediction`)
- **Memory Integration**: Validates memory configuration and integration components
- **GPU/CUDA Testing**: Checks GPU availability and basic CUDA operations
- **Comprehensive Reporting**: Detailed test results with pass/fail status

### 📊 Monitoring & Logging
- **Colored Terminal Output**: Easy-to-read status indicators
- **Detailed Logging**: Complete setup log saved to `worldmem_setup.log`
- **Progress Tracking**: Step-by-step progress with clear status messages
- **Error Handling**: Comprehensive error catching and reporting

## Usage

### Quick Start
```bash
# Navigate to your workspace directory
cd "/home/zhangboyuan/VScode workspace"

# Run the setup and test script
python worldmem_setup_and_test.py
```

### Custom Workspace Path
```bash
# If your workspace is in a different location
python worldmem_setup_and_test.py --workspace /path/to/your/workspace
```

## What the Script Does

### 1. Environment Setup Phase
```
🔍 CHECKING PYTHON ENVIRONMENT
├── Validates Python version (3.10+)
├── Checks conda/mamba environment
├── Verifies base packages (torch, numpy, pandas)
└── Reports environment status
```

### 2. Installation Phase
```
📦 INSTALLING WORLDMEM REQUIREMENTS
├── Locates WorldMem/requirements.txt
├── Installs all Python dependencies via pip
├── Handles conda-specific packages (ffmpeg)
└── Reports installation status
```

### 3. Cache Cleanup Phase
```
🧹 CLEANING PYTHON CACHE
├── Removes __pycache__ directories
├── Deletes .pyc and .pyo files
├── Cleans .pytest_cache directories
└── Reports cleanup results
```

### 4. Testing Phase
```
🧪 COMPREHENSIVE WORLDMEM TEST
├── Tests core library imports (torch, lightning, etc.)
├── Validates WorldMem algorithm imports
├── Checks memory integration functionality
├── Tests GPU/CUDA availability
└── Generates comprehensive test report
```

## Directory Structure Expected

The script expects this workspace structure:
```
VScode workspace/
├── Mine/
│   └── nwm/
│       ├── WorldMem/              # Main WorldMem directory
│       │   ├── requirements.txt   # Python dependencies
│       │   ├── algorithms/        # Core algorithms
│       │   ├── main.py           # Main training script
│       │   └── app.py            # Gradio app
│       └── config/
│           └── memory_config.yaml # Memory configuration
└── Simon/
    └── nwm/                       # Alternative NWM setup
```

## Dependencies Installed

### Core ML/AI Libraries
- **torch**: PyTorch deep learning framework
- **torchvision**: Computer vision utilities
- **lightning**: PyTorch Lightning training framework
- **diffusers**: Hugging Face diffusion models
- **timm**: PyTorch image models
- **einops**: Tensor operations made easy

### WorldMem Specific
- **hydra-core**: Configuration management
- **omegaconf**: Configuration handling
- **wandb**: Experiment tracking
- **torchmetrics**: ML metrics
- **rotary_embedding_torch**: Rotary embeddings

### Utilities
- **opencv-python**: Computer vision
- **matplotlib**: Plotting
- **pandas**: Data manipulation
- **tqdm**: Progress bars
- **moviepy**: Video processing
- **gradio**: Web interface
- **ffmpeg**: Video encoding (via conda)

## Output Interpretation

### Success Indicators
- ✅ **Green checkmarks**: Successful operations
- 🎉 **Celebration emoji**: Complete success
- 📊 **High success rate**: 80%+ tests passing

### Warning Indicators
- ⚠️ **Yellow warnings**: Non-critical issues
- 🔧 **Wrench emoji**: Fixable problems
- 📝 **Note emoji**: Information messages

### Error Indicators
- ❌ **Red X marks**: Failed operations
- 🚨 **Alert emoji**: Critical errors
- 📋 **Clipboard emoji**: Check logs for details

## Advanced Usage

### Testing Only (Skip Installation)
To run only the test suite without installing dependencies:

```python
from worldmem_setup_and_test import WorldMemSetup

setup = WorldMemSetup()
setup.run_comprehensive_test()
```

### Custom Configuration
```python
setup = WorldMemSetup(workspace_path="/custom/path")
setup.full_setup()
```

### Individual Test Components
```python
setup = WorldMemSetup()

# Test specific components
setup.test_worldmem_import()
setup.test_memory_integration()
setup.test_gpu_availability()
```

## Troubleshooting

### Common Issues

**1. Workspace Not Found**
```
Error: Could not detect workspace with Mine/nwm and Simon/nwm directories
```
**Solution**: Ensure you're running from the correct directory or specify the workspace path.

**2. Conda Environment Issues**
```
Warning: Could not install ffmpeg via conda
```
**Solution**: Ensure conda/mamba is properly installed and activated.

**3. GPU/CUDA Issues**
```
Warning: CUDA not available - will run on CPU
```
**Solution**: Install CUDA toolkit or run on CPU (script will continue).

**4. Import Errors**
```
Error: Failed to import WorldMem algorithms
```
**Solution**: Check that all dependencies are installed correctly.

### Debug Mode
For detailed debugging, check the log file:
```bash
tail -f worldmem_setup.log
```

## Integration with WorldMem

After successful setup, you can:

### 1. Run the Gradio App
```bash
cd "WorldMem"
python app.py
```

### 2. Start Training
```bash
cd "WorldMem"
python main.py
```

### 3. Run Inference
```bash
cd "WorldMem"
bash infer.sh
```

### 4. Use in Python Scripts
```python
import sys
sys.path.append('/path/to/WorldMem')

from algorithms.worldmem import WorldMemMinecraft
model = WorldMemMinecraft()
```

## Memory Integration Features

The script specifically tests:

1. **Memory Configuration Loading**: Validates `memory_config.yaml`
2. **Memory Capacity Settings**: Checks memory buffer configurations
3. **Long-term Memory**: Tests long-term memory threshold settings
4. **Memory Attention**: Validates memory attention mechanisms
5. **Temporal Embeddings**: Tests timestamp embedding functionality

## Performance Optimization

The script includes optimizations for:
- **Mixed Precision**: Enables automatic mixed precision training
- **Compilation**: Supports torch.compile for faster inference
- **Memory Management**: Optimizes GPU memory usage
- **Caching**: Manages model and data caching efficiently

## Security & Privacy

- **No Data Collection**: Script does not collect or transmit personal data
- **Local Execution**: All operations run locally on your machine
- **Safe Installation**: Only installs from official package repositories
- **Log Safety**: Logs contain no sensitive information

## Contributing

To extend the script:

1. **Add New Tests**: Extend the `test_*` methods in `WorldMemSetup`
2. **Add Dependencies**: Update the installation methods
3. **Improve Detection**: Enhance workspace/environment detection
4. **Add Platforms**: Support additional operating systems

## License

This script is part of the Navigation World Models project. Please refer to the main project license for usage terms.

## Support

For issues related to:
- **Script Functionality**: Check the log file and troubleshooting section
- **WorldMem Specific**: Refer to WorldMem documentation
- **NWM Project**: Visit the main NWM repository

---

**Created for the Navigation World Models Project**  
*Making WorldMem setup and testing effortless* 🌍🧠
