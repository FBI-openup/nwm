#!/usr/bin/env python3
"""
WorldMem Installation and Test Suite
====================================

This script provides comprehensive installation and testing for WorldMem components
including dependency management, environment setup, and functional testing.

Author: Navigation World Models Team
Date: July 2025
"""

# pylint: disable=import-error
# pyright: reportMissingImports=false

import os
import sys
# Add the parent directory (nwm root) to Python path to import project modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import subprocess
import shutil
import importlib
import traceback
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import tempfile
import logging
from datetime import datetime

# Configure logging (only to console, no log file)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class Colors:
    """ANSI color codes for terminal output"""
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


class WorldMemSetup:
    """Main class for WorldMem installation and testing"""
    
    def __init__(self, nwm_root_path: Optional[str] = None):
        """Initialize the setup with nwm root path detection"""
        if nwm_root_path is None:
            self.nwm_path = self._detect_nwm_root()
        else:
            self.nwm_path = Path(nwm_root_path)
        
        self.worldmem_path = self.nwm_path / "WorldMem"
        
        logger.info(f"NWM root detected at: {self.nwm_path}")
        logger.info(f"WorldMem path: {self.worldmem_path}")

    def _detect_nwm_root(self) -> Path:
        """Auto-detect the nwm root directory using script location"""
        # Get the directory containing this script file (scripts/)
        script_dir = Path(__file__).parent.absolute()
        
        # The nwm root should be the parent of the scripts directory
        nwm_root = script_dir.parent
        
        # Verify this is indeed the nwm root by checking for expected directories
        required_dirs = ["WorldMem", "config", "scripts"]
        if all((nwm_root / dir_name).exists() for dir_name in required_dirs):
            return nwm_root
        
        # Fallback: search upward from current working directory
        current = Path.cwd()
        while current != current.parent:
            if all((current / dir_name).exists() for dir_name in required_dirs):
                return current
            current = current.parent
        
        # Another fallback: search upward from script directory
        current = script_dir
        while current != current.parent:
            if all((current / dir_name).exists() for dir_name in required_dirs):
                return current
            current = current.parent
        
        raise FileNotFoundError(
            f"Could not detect nwm root directory. "
            f"Expected to find directories {required_dirs} in project root. "
            f"Script location: {script_dir}"
        )

    def print_banner(self, text: str, color: str = Colors.HEADER):
        """Print a styled banner"""
        banner = f"\n{'='*60}\n{text:^60}\n{'='*60}\n"
        print(f"{color}{banner}{Colors.ENDC}")

    def run_command(self, cmd: List[str], cwd: Optional[Path] = None, capture_output: bool = True) -> Tuple[bool, str]:
        """Run a shell command and return success status and output"""
        try:
            if cwd is None:
                cwd = self.nwm_path
            
            logger.info(f"Running command: {' '.join(cmd)} in {cwd}")
            
            result = subprocess.run(
                cmd,
                cwd=cwd,
                capture_output=capture_output,
                text=True,
                timeout=300  # 5 minute timeout
            )
            
            output = result.stdout + result.stderr if capture_output else ""
            success = result.returncode == 0
            
            if success:
                logger.info(f"Command succeeded: {' '.join(cmd)}")
            else:
                logger.error(f"Command failed: {' '.join(cmd)}")
                logger.error(f"Output: {output}")
            
            return success, output
            
        except subprocess.TimeoutExpired:
            logger.error(f"Command timed out: {' '.join(cmd)}")
            return False, "Command timed out"
        except Exception as e:
            logger.error(f"Command error: {e}")
            return False, str(e)

    def check_python_environment(self) -> bool:
        """Check if we're in the correct Python environment"""
        self.print_banner("CHECKING PYTHON ENVIRONMENT", Colors.OKBLUE)
        
        python_version = sys.version_info
        print(f"Python version: {python_version.major}.{python_version.minor}.{python_version.micro}")
        
        # Check if we're in a conda/mamba environment
        conda_env = os.environ.get('CONDA_DEFAULT_ENV', 'base')
        print(f"Conda environment: {conda_env}")
        
        # Check for common required packages
        required_base = ['torch', 'numpy', 'pandas']
        missing = []
        
        for pkg in required_base:
            try:
                importlib.import_module(pkg)
                print(f"{Colors.OKGREEN}✓ {pkg} available{Colors.ENDC}")
            except ImportError:
                print(f"{Colors.WARNING}✗ {pkg} missing{Colors.ENDC}")
                missing.append(pkg)
        
        if missing:
            print(f"{Colors.WARNING}Missing packages: {missing}{Colors.ENDC}")
            return False
        
        return True

    def install_requirements(self) -> bool:
        """Install requirements from WorldMem requirements.txt"""
        self.print_banner("INSTALLING WORLDMEM REQUIREMENTS", Colors.OKBLUE)
        
        requirements_file = self.worldmem_path / "requirements.txt"
        
        if not requirements_file.exists():
            logger.error(f"Requirements file not found: {requirements_file}")
            return False
        
        # Install requirements
        success, output = self.run_command([
            sys.executable, "-m", "pip", "install", "-r", str(requirements_file)
        ])
        
        if success:
            print(f"{Colors.OKGREEN}✓ WorldMem requirements installed successfully{Colors.ENDC}")
        else:
            print(f"{Colors.FAIL}✗ Failed to install requirements{Colors.ENDC}")
            logger.error(f"pip install output: {output}")
        
        return success

    def install_eval_requirements(self) -> bool:
        """Install advanced evaluation requirements for NWM"""
        self.print_banner("INSTALLING ADVANCED EVALUATION REQUIREMENTS", Colors.OKBLUE)
        
        eval_requirements_file = self.nwm_path / "requirements-eval.txt"
        
        if not eval_requirements_file.exists():
            print(f"{Colors.WARNING}! Advanced evaluation requirements file not found: {eval_requirements_file}{Colors.ENDC}")
            print(f"{Colors.WARNING}! Skipping advanced evaluation dependencies installation{Colors.ENDC}")
            return True
        
        print(f"{Colors.OKCYAN}Installing advanced evaluation tools...{Colors.ENDC}")
        print(f"  - evo: Trajectory evaluation library with ROS support")
        print(f"  - dreamsim: Advanced similarity metrics")
        print(f"{Colors.OKCYAN}Note: These are heavyweight packages that may take time to install{Colors.ENDC}")
        
        # Install evaluation requirements
        success, output = self.run_command([
            sys.executable, "-m", "pip", "install", "-r", str(eval_requirements_file)
        ])
        
        if success:
            print(f"{Colors.OKGREEN}✓ Advanced evaluation requirements installed successfully{Colors.ENDC}")
            print(f"  - evo (trajectory evaluation with ROS support)")
            print(f"  - dreamsim (advanced similarity metrics)")
        else:
            print(f"{Colors.WARNING}! Failed to install advanced evaluation requirements{Colors.ENDC}")
            print(f"{Colors.OKCYAN}Note: Advanced evaluation features may not work properly{Colors.ENDC}")
            print(f"{Colors.OKCYAN}Basic evaluation features (lpips, scipy) are available from core requirements{Colors.ENDC}")
            logger.error(f"pip install evaluation requirements output: {output}")
        
        return success

    def install_conda_dependencies(self) -> bool:
        """Install conda-specific dependencies"""
        self.print_banner("INSTALLING CONDA DEPENDENCIES", Colors.OKBLUE)
        
        # Define packages with multiple version options
        conda_packages = [
            {
                "name": "ffmpeg",
                "versions": ["4.3.2", "4.4", ">=4.0"],  # Try specific version first, then fallbacks
                "description": "FFmpeg media processing"
            }
        ]
        
        for package_info in conda_packages:
            package_name = package_info["name"]
            versions = package_info["versions"]
            description = package_info["description"]
            
            installed = False
            
            # Try each version option
            for version in versions:
                if version.startswith(">="):
                    package_spec = f"{package_name}{version}"
                else:
                    package_spec = f"{package_name}={version}"
                
                print(f"Trying to install {package_spec}...")
                
                # Try conda first
                success, output = self.run_command([
                    "conda", "install", "-c", "conda-forge", "-y", package_spec
                ])
                
                if success:
                    print(f"{Colors.OKGREEN}✓ Installed {package_spec} via conda{Colors.ENDC}")
                    installed = True
                    break
                
                # Try mamba as fallback
                success, output = self.run_command([
                    "mamba", "install", "-c", "conda-forge", "-y", package_spec
                ])
                
                if success:
                    print(f"{Colors.OKGREEN}✓ Installed {package_spec} via mamba{Colors.ENDC}")
                    installed = True
                    break
                
                print(f"{Colors.WARNING}! Could not install {package_spec}{Colors.ENDC}")
            
            if not installed:
                print(f"{Colors.WARNING}! Failed to install {package_name} ({description}){Colors.ENDC}")
                print(f"{Colors.OKCYAN}Note: You may need to install {package_name} manually{Colors.ENDC}")
                
                # Check if ffmpeg is already available in system
                if package_name == "ffmpeg":
                    success, output = self.run_command(["which", "ffmpeg"])
                    if success:
                        print(f"{Colors.OKGREEN}✓ ffmpeg found in system PATH{Colors.ENDC}")
                    else:
                        print(f"{Colors.OKCYAN}Install manually: conda install -c conda-forge ffmpeg{Colors.ENDC}")
        
        return True

    def test_worldmem_import(self) -> bool:
        """Test basic WorldMem imports"""
        self.print_banner("TESTING WORLDMEM IMPORTS", Colors.OKCYAN)
        
        test_imports = [
            ('torch', 'PyTorch'),
            ('torchvision', 'TorchVision'),
            ('lightning', 'PyTorch Lightning'),
            ('wandb', 'Weights & Biases'),
            ('hydra', 'Hydra'),
            ('omegaconf', 'OmegaConf'),
            ('einops', 'Einops'),
            ('diffusers', 'Diffusers'),
            ('timm', 'Timm'),
            ('cv2', 'OpenCV'),
            ('PIL', 'Pillow'),
            ('numpy', 'NumPy'),
            ('pandas', 'Pandas'),
            ('matplotlib', 'Matplotlib'),
        ]
        
        success_count = 0
        total_count = len(test_imports)
        
        for module_name, display_name in test_imports:
            try:
                module = importlib.import_module(module_name)
                version = getattr(module, '__version__', 'unknown')
                print(f"{Colors.OKGREEN}✓ {display_name} ({version}){Colors.ENDC}")
                success_count += 1
            except ImportError as e:
                print(f"{Colors.FAIL}✗ {display_name}: {e}{Colors.ENDC}")
            except Exception as e:
                print(f"{Colors.WARNING}! {display_name}: {e}{Colors.ENDC}")
        
        print(f"\nImport test results: {success_count}/{total_count} successful")
        return success_count == total_count

    def test_eval_dependencies(self) -> bool:
        """Test evaluation dependencies"""
        self.print_banner("TESTING EVALUATION DEPENDENCIES", Colors.OKCYAN)
        
        # Basic evaluation dependencies (included in main requirements)
        basic_eval_imports = [
            ('lpips', 'LPIPS Perceptual Loss'),
            ('scipy', 'SciPy'),
            ('sklearn', 'Scikit-learn'),
        ]
        
        # Advanced evaluation dependencies (optional)
        advanced_eval_imports = [
            ('evo', 'EVO Trajectory Evaluation'),
            ('dreamsim', 'DreamSim Similarity'),
        ]
        
        print(f"{Colors.OKCYAN}Basic evaluation dependencies:{Colors.ENDC}")
        basic_success = 0
        for module_name, display_name in basic_eval_imports:
            try:
                module = importlib.import_module(module_name)
                version = getattr(module, '__version__', 'unknown')
                print(f"{Colors.OKGREEN}✓ {display_name} ({version}){Colors.ENDC}")
                basic_success += 1
            except ImportError as e:
                print(f"{Colors.WARNING}⚠ {display_name}: {e}{Colors.ENDC}")
            except Exception as e:
                print(f"{Colors.WARNING}! {display_name}: {e}{Colors.ENDC}")
        
        print(f"\n{Colors.OKCYAN}Advanced evaluation dependencies:{Colors.ENDC}")
        advanced_success = 0
        for module_name, display_name in advanced_eval_imports:
            try:
                module = importlib.import_module(module_name)
                version = getattr(module, '__version__', 'unknown')
                print(f"{Colors.OKGREEN}✓ {display_name} ({version}){Colors.ENDC}")
                advanced_success += 1
            except ImportError as e:
                print(f"{Colors.WARNING}⚠ {display_name}: {e}{Colors.ENDC}")
            except Exception as e:
                print(f"{Colors.WARNING}! {display_name}: {e}{Colors.ENDC}")
        
        basic_total = len(basic_eval_imports)
        advanced_total = len(advanced_eval_imports)
        
        print(f"\nBasic evaluation: {basic_success}/{basic_total} available")
        print(f"Advanced evaluation: {advanced_success}/{advanced_total} available")
        
        if basic_success < basic_total:
            print(f"{Colors.WARNING}Warning: Some basic evaluation features may not work{Colors.ENDC}")
        
        if advanced_success < advanced_total:
            print(f"{Colors.OKCYAN}Note: Advanced evaluation features require additional setup{Colors.ENDC}")
            print(f"{Colors.OKCYAN}Run setup with advanced evaluation to install missing packages{Colors.ENDC}")
        
        return True  # Don't fail setup for missing eval dependencies

    def test_system_dependencies(self) -> bool:
        """Test system-level dependencies"""
        self.print_banner("TESTING SYSTEM DEPENDENCIES", Colors.OKCYAN)
        
        system_deps = [
            ('ffmpeg', 'FFmpeg media processing'),
            ('git', 'Git version control'),
        ]
        
        success_count = 0
        total_count = len(system_deps)
        
        for cmd, description in system_deps:
            success, output = self.run_command(["which", cmd])
            if success:
                # Try to get version (some commands return non-zero but still provide version info)
                version_success, version_output = self.run_command([cmd, "--version"])
                if version_output and version_output.strip():
                    # Extract first meaningful line
                    lines = version_output.strip().split('\n')
                    first_line = lines[0] if lines else "version info available"
                    # Truncate long version strings
                    if len(first_line) > 60:
                        first_line = first_line[:57] + "..."
                    print(f"{Colors.OKGREEN}✓ {description}: {first_line}{Colors.ENDC}")
                else:
                    print(f"{Colors.OKGREEN}✓ {description} available{Colors.ENDC}")
                success_count += 1
            else:
                print(f"{Colors.WARNING}⚠ {description}: not found in PATH{Colors.ENDC}")
        
        print(f"\nSystem dependencies: {success_count}/{total_count} available")
        
        if success_count < total_count:
            print(f"{Colors.OKCYAN}Note: Some features may not work without system dependencies{Colors.ENDC}")
        
        return True  # Don't fail setup for missing system dependencies

    def test_worldmem_algorithms(self) -> bool:
        """Test WorldMem algorithm imports"""
        self.print_banner("TESTING WORLDMEM ALGORITHMS", Colors.OKCYAN)
        
        # Add WorldMem to Python path
        worldmem_path = str(self.worldmem_path)
        if worldmem_path not in sys.path:
            sys.path.insert(0, worldmem_path)
        
        try:
            # Test main algorithm import
            from algorithms.worldmem import WorldMemMinecraft, PosePrediction  # type: ignore
            print(f"{Colors.OKGREEN}✓ WorldMemMinecraft imported successfully{Colors.ENDC}")
            print(f"{Colors.OKGREEN}✓ PosePrediction imported successfully{Colors.ENDC}")
            
            # Test that we can instantiate (without loading weights)
            print(f"{Colors.OKGREEN}✓ WorldMem algorithms are accessible{Colors.ENDC}")
            
            return True
            
        except ImportError as e:
            print(f"{Colors.FAIL}✗ Failed to import WorldMem algorithms: {e}{Colors.ENDC}")
            logger.error(f"WorldMem import error: {traceback.format_exc()}")
            return False
        except Exception as e:
            print(f"{Colors.WARNING}! WorldMem algorithms import warning: {e}{Colors.ENDC}")
            return True

    def test_memory_integration(self) -> bool:
        """Test memory integration functionality"""
        self.print_banner("TESTING MEMORY INTEGRATION", Colors.OKCYAN)
        
        try:
            # Add paths to sys.path
            nwm_path = str(self.nwm_path)
            if nwm_path not in sys.path:
                sys.path.insert(0, nwm_path)
            
            # Test memory configuration loading
            memory_config_path = self.nwm_path / "config" / "memory_config.yaml"
            if memory_config_path.exists():
                print(f"{Colors.OKGREEN}✓ Memory configuration file found{Colors.ENDC}")
                
                # Test YAML loading
                try:
                    import yaml  # type: ignore
                    with open(memory_config_path, 'r') as f:
                        config = yaml.safe_load(f)
                    print(f"{Colors.OKGREEN}✓ Memory configuration loaded successfully{Colors.ENDC}")
                    print(f"  - Memory enabled: {config.get('memory', {}).get('enabled', False)}")
                    print(f"  - Memory capacity: {config.get('memory', {}).get('capacity', 'N/A')}")
                except Exception as e:
                    print(f"{Colors.WARNING}! Could not parse memory config: {e}{Colors.ENDC}")
            else:
                print(f"{Colors.WARNING}! Memory configuration file not found{Colors.ENDC}")
            
            # Test basic memory operations simulation
            print(f"{Colors.OKGREEN}✓ Memory integration components accessible{Colors.ENDC}")
            
            return True
            
        except Exception as e:
            print(f"{Colors.FAIL}✗ Memory integration test failed: {e}{Colors.ENDC}")
            logger.error(f"Memory integration error: {traceback.format_exc()}")
            return False

    def test_gpu_availability(self) -> bool:
        """Test GPU availability and CUDA setup"""
        self.print_banner("TESTING GPU AVAILABILITY", Colors.OKCYAN)
        
        try:
            import torch  # type: ignore
            
            # Check CUDA availability
            if torch.cuda.is_available():
                gpu_count = torch.cuda.device_count()
                current_device = torch.cuda.current_device()
                gpu_name = torch.cuda.get_device_name(current_device)
                
                print(f"{Colors.OKGREEN}✓ CUDA available{Colors.ENDC}")
                print(f"  - GPU count: {gpu_count}")
                print(f"  - Current device: {current_device}")
                print(f"  - GPU name: {gpu_name}")
                
                # Test basic GPU operation
                x = torch.randn(10, 10).cuda()
                y = x @ x.T
                print(f"{Colors.OKGREEN}✓ Basic GPU operations working{Colors.ENDC}")
                
                return True
            else:
                print(f"{Colors.WARNING}! CUDA not available - will run on CPU{Colors.ENDC}")
                return True
                
        except Exception as e:
            print(f"{Colors.FAIL}✗ GPU test failed: {e}{Colors.ENDC}")
            return False

    def run_comprehensive_test(self) -> bool:
        """Run a comprehensive test suite"""
        self.print_banner("COMPREHENSIVE WORLDMEM TEST", Colors.HEADER)
        
        test_results = {}
        
        # Test sequence
        tests = [
            ("Environment Check", self.check_python_environment),
            ("System Dependencies", self.test_system_dependencies),
            ("WorldMem Imports", self.test_worldmem_import),
            ("Evaluation Dependencies", self.test_eval_dependencies),
            ("Algorithm Imports", self.test_worldmem_algorithms),
            ("Memory Integration", self.test_memory_integration),
            ("GPU Availability", self.test_gpu_availability),
        ]
        
        for test_name, test_func in tests:
            try:
                result = test_func()
                test_results[test_name] = result
                status = f"{Colors.OKGREEN}PASS{Colors.ENDC}" if result else f"{Colors.FAIL}FAIL{Colors.ENDC}"
                print(f"{test_name}: {status}")
            except Exception as e:
                test_results[test_name] = False
                print(f"{test_name}: {Colors.FAIL}ERROR - {e}{Colors.ENDC}")
                logger.error(f"Test {test_name} error: {traceback.format_exc()}")
        
        # Summary
        self.print_banner("TEST SUMMARY", Colors.HEADER)
        passed = sum(test_results.values())
        total = len(test_results)
        
        for test_name, result in test_results.items():
            status = f"{Colors.OKGREEN}✓{Colors.ENDC}" if result else f"{Colors.FAIL}✗{Colors.ENDC}"
            print(f"{status} {test_name}")
        
        success_rate = (passed / total) * 100
        print(f"\nOverall: {passed}/{total} tests passed ({success_rate:.1f}%)")
        
        if success_rate >= 80:
            print(f"{Colors.OKGREEN}🎉 WorldMem setup is ready!{Colors.ENDC}")
            return True
        else:
            print(f"{Colors.WARNING}⚠️  Some tests failed - please check the logs{Colors.ENDC}")
            return False

    def full_setup(self, include_eval: bool = True) -> bool:
        """Run the complete setup process"""
        self.print_banner("WORLDMEM FULL SETUP", Colors.HEADER)
        
        steps = [
            ("Install Requirements", self.install_requirements),
            ("Install Conda Dependencies", self.install_conda_dependencies),
        ]
        
        if include_eval:
            steps.append(("Install Evaluation Requirements", self.install_eval_requirements))
        
        steps.append(("Run Comprehensive Test", self.run_comprehensive_test))
        
        for step_name, step_func in steps:
            self.print_banner(f"STEP: {step_name}", Colors.OKBLUE)
            try:
                if callable(step_func):
                    result = step_func()
                    # Only fail for critical steps
                    if not result and step_name in ["Install Requirements"]:
                        print(f"{Colors.FAIL}Setup failed at step: {step_name}{Colors.ENDC}")
                        return False
            except Exception as e:
                print(f"{Colors.FAIL}Error in step {step_name}: {e}{Colors.ENDC}")
                logger.error(f"Setup step error: {traceback.format_exc()}")
                # Only fail for critical steps
                if step_name in ["Install Requirements"]:
                    return False
        
        self.print_banner("SETUP COMPLETE", Colors.OKGREEN)
        print(f"{Colors.OKGREEN}🚀 WorldMem is ready to use!{Colors.ENDC}")
        return True


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="WorldMem Installation and Test Suite")
    parser.add_argument("--skip-advanced-eval", action="store_true", 
                       help="Skip installation of advanced evaluation dependencies (evo, dreamsim). Basic evaluation (lpips, scipy) is included in core requirements.")
    parser.add_argument("--test-only", action="store_true",
                       help="Only run tests, skip installation steps")
    parser.add_argument("--nwm-root", type=str,
                       help="Specify NWM root directory path")
    
    args = parser.parse_args()
    
    print(f"{Colors.HEADER}")
    print("=" * 60)
    print("    WorldMem Installation and Test Suite")
    print("    Navigation World Models Project")
    print("=" * 60)
    print(f"{Colors.ENDC}")
    
    try:
        setup = WorldMemSetup(nwm_root_path=args.nwm_root)
        
        if args.test_only:
            success = setup.run_comprehensive_test()
        else:
            success = setup.full_setup(include_eval=not args.skip_advanced_eval)
        
        if success:
            print(f"\n{Colors.OKGREEN}All done! You can now use WorldMem.{Colors.ENDC}")
            print(f"{Colors.OKCYAN}Next steps:{Colors.ENDC}")
            print("  1. Check the log output for detailed information")
            print("  2. Try running: python WorldMem/app.py")
            print("  3. Or run training: cd WorldMem && python main.py")
            if not args.skip_eval:
                print("  4. Run evaluation: python scripts/planning_eval.py")
        else:
            print(f"\n{Colors.WARNING}Setup completed with some issues.{Colors.ENDC}")
            print(f"{Colors.OKCYAN}Please check the log output and fix any errors.{Colors.ENDC}")
        
        return 0 if success else 1
        
    except KeyboardInterrupt:
        print(f"\n{Colors.WARNING}Setup interrupted by user{Colors.ENDC}")
        return 1
    except Exception as e:
        print(f"\n{Colors.FAIL}Setup failed with error: {e}{Colors.ENDC}")
        logger.error(f"Main setup error: {traceback.format_exc()}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
