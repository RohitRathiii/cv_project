"""
Setup Validation Script

This script validates the project setup without requiring all dependencies
to be installed. It checks file structure, syntax, and basic functionality.
"""

import os
import sys
import ast
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def check_python_syntax(file_path: Path) -> bool:
    """Check if a Python file has valid syntax"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            source = f.read()
        ast.parse(source)
        return True
    except SyntaxError as e:
        logger.error(f"Syntax error in {file_path}: {e}")
        return False
    except Exception as e:
        logger.error(f"Error reading {file_path}: {e}")
        return False


def validate_project_structure():
    """Validate project directory structure"""
    logger.info("Validating project structure...")
    
    project_root = Path(__file__).parent.parent
    
    # Required directories
    required_dirs = [
        'src', 'src/models', 'src/data', 'src/pipeline', 'src/utils',
        'scripts', 'config', 'tests', 'models', 'datasets', 'results'
    ]
    
    missing_dirs = []
    for dir_path in required_dirs:
        full_path = project_root / dir_path
        if not full_path.exists():
            missing_dirs.append(dir_path)
    
    if missing_dirs:
        logger.error(f"Missing directories: {missing_dirs}")
        return False
    
    logger.info("✅ All required directories present")
    return True


def validate_python_files():
    """Validate Python file syntax"""
    logger.info("Validating Python file syntax...")
    
    project_root = Path(__file__).parent.parent
    
    # Core Python files to check
    python_files = [
        'src/__init__.py',
        'src/models/__init__.py',
        'src/models/apple_detector.py',
        'src/models/quality_classifier.py',
        'src/models/apple_tracker.py',
        'src/data/__init__.py',
        'src/data/data_processing.py',
        'src/pipeline/__init__.py',
        'src/pipeline/apple_pipeline.py',
        'src/utils/__init__.py',
        'src/utils/gradio_interface.py',
        'src/utils/evaluation.py',
        'scripts/train_detection.py',
        'scripts/train_quality.py',
        'scripts/train_pipeline.py'
    ]
    
    syntax_errors = []
    for file_path in python_files:
        full_path = project_root / file_path
        if full_path.exists():
            if not check_python_syntax(full_path):
                syntax_errors.append(file_path)
        else:
            logger.warning(f"File not found: {file_path}")
            syntax_errors.append(file_path)
    
    if syntax_errors:
        logger.error(f"Files with syntax errors: {syntax_errors}")
        return False
    
    logger.info("✅ All Python files have valid syntax")
    return True


def validate_config_files():
    """Validate configuration files"""
    logger.info("Validating configuration files...")
    
    project_root = Path(__file__).parent.parent
    
    try:
        import yaml
        
        # Check main config
        config_path = project_root / 'config' / 'config.yaml'
        if config_path.exists():
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            # Validate config structure
            required_sections = ['model', 'training', 'data']
            for section in required_sections:
                if section not in config:
                    logger.error(f"Missing config section: {section}")
                    return False
        
        # Check dataset config
        dataset_config_path = project_root / 'config' / 'dataset.yaml'
        if dataset_config_path.exists():
            with open(dataset_config_path, 'r') as f:
                dataset_config = yaml.safe_load(f)
            
            required_keys = ['nc', 'names']
            for key in required_keys:
                if key not in dataset_config:
                    logger.error(f"Missing dataset config key: {key}")
                    return False
        
        logger.info("✅ Configuration files are valid")
        return True
        
    except ImportError:
        logger.warning("⚠️ PyYAML not installed, skipping config validation")
        return True
    except Exception as e:
        logger.error(f"Error validating config files: {e}")
        return False


def validate_requirements():
    """Validate requirements.txt"""
    logger.info("Validating requirements.txt...")
    
    project_root = Path(__file__).parent.parent
    requirements_path = project_root / 'requirements.txt'
    
    if not requirements_path.exists():
        logger.error("requirements.txt not found")
        return False
    
    try:
        with open(requirements_path, 'r') as f:
            requirements = f.read().strip()
        
        # Check for essential packages
        essential_packages = [
            'torch', 'torchvision', 'ultralytics', 'opencv-python',
            'gradio', 'numpy', 'pandas', 'matplotlib'
        ]
        
        missing_packages = []
        for package in essential_packages:
            if package not in requirements:
                missing_packages.append(package)
        
        if missing_packages:
            logger.warning(f"Missing essential packages in requirements: {missing_packages}")
        
        logger.info("✅ Requirements.txt is present and contains key packages")
        return True
        
    except Exception as e:
        logger.error(f"Error reading requirements.txt: {e}")
        return False


def validate_documentation():
    """Validate documentation files"""
    logger.info("Validating documentation...")
    
    project_root = Path(__file__).parent.parent
    
    # Check README
    readme_path = project_root / 'README.md'
    if not readme_path.exists():
        logger.error("README.md not found")
        return False
    
    try:
        with open(readme_path, 'r') as f:
            readme_content = f.read()
        
        # Check for essential sections
        essential_sections = [
            'Installation', 'Usage', 'Features', 'Architecture'
        ]
        
        missing_sections = []
        for section in essential_sections:
            if section.lower() not in readme_content.lower():
                missing_sections.append(section)
        
        if missing_sections:
            logger.warning(f"README missing sections: {missing_sections}")
        
        logger.info("✅ Documentation files are present")
        return True
        
    except Exception as e:
        logger.error(f"Error reading documentation: {e}")
        return False


def check_file_sizes():
    """Check for reasonable file sizes"""
    logger.info("Checking file sizes...")
    
    project_root = Path(__file__).parent.parent
    
    large_files = []
    
    for file_path in project_root.rglob('*.py'):
        if file_path.stat().st_size > 100000:  # 100KB
            large_files.append(f"{file_path.relative_to(project_root)} ({file_path.stat().st_size // 1024}KB)")
    
    if large_files:
        logger.info(f"Large Python files detected: {large_files}")
    
    logger.info("✅ File size check completed")
    return True


def run_validation():
    """Run all validation checks"""
    logger.info("="*60)
    logger.info("APPLE DETECTION PIPELINE - SETUP VALIDATION")
    logger.info("="*60)
    
    checks = [
        ("Project Structure", validate_project_structure),
        ("Python File Syntax", validate_python_files),
        ("Configuration Files", validate_config_files),
        ("Requirements File", validate_requirements),
        ("Documentation", validate_documentation),
        ("File Sizes", check_file_sizes)
    ]
    
    results = {}
    
    for check_name, check_func in checks:
        logger.info(f"\n--- {check_name} ---")
        try:
            results[check_name] = check_func()
        except Exception as e:
            logger.error(f"Check {check_name} failed with error: {e}")
            results[check_name] = False
    
    # Summary
    logger.info("\n" + "="*60)
    logger.info("VALIDATION RESULTS")
    logger.info("="*60)
    
    passed = 0
    total = len(checks)
    
    for check_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        logger.info(f"{check_name:<25} {status}")
        if result:
            passed += 1
    
    logger.info("-" * 60)
    logger.info(f"TOTAL: {passed}/{total} checks passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        logger.info("🎉 ALL CHECKS PASSED! Project setup is complete.")
        logger.info("\nNext steps:")
        logger.info("1. Install dependencies: pip install -r requirements.txt")
        logger.info("2. Download datasets or prepare your own")
        logger.info("3. Train models or download pre-trained weights")
        logger.info("4. Run the pipeline or web interface")
    else:
        logger.warning(f"⚠️ {total-passed} checks failed. Please address the issues above.")
    
    logger.info("="*60)
    
    return passed == total


if __name__ == "__main__":
    success = run_validation()
    sys.exit(0 if success else 1)