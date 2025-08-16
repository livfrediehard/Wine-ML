#!/usr/bin/env python3
"""
Setup script for Wine ML Learning Project
This script helps you set up your machine learning environment.
"""

import subprocess
import sys
import os

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"\n🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error during {description}: {e}")
        print(f"Error output: {e.stderr}")
        return False

def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8 or higher is required!")
        print(f"Current version: {version.major}.{version.minor}.{version.micro}")
        return False
    print(f"✅ Python version {version.major}.{version.minor}.{version.micro} is compatible")
    return True

def main():
    print("🍷 Wine Quality ML Learning Project Setup")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        return
    
    # Check if virtual environment exists
    venv_path = "wine_ml_env"
    if os.path.exists(venv_path):
        print(f"\n📁 Virtual environment '{venv_path}' already exists")
        response = input("Do you want to recreate it? (y/N): ").lower()
        if response == 'y':
            print("🗑️ Removing existing virtual environment...")
            if os.name == 'nt':  # Windows
                run_command(f"rmdir /s /q {venv_path}", "Removing existing virtual environment")
            else:  # Unix/Linux/Mac
                run_command(f"rm -rf {venv_path}", "Removing existing virtual environment")
        else:
            print("Using existing virtual environment")
    
    # Create virtual environment if it doesn't exist
    if not os.path.exists(venv_path):
        if not run_command(f"python3 -m venv {venv_path}", "Creating virtual environment"):
            return
    
    # Determine activation command
    if os.name == 'nt':  # Windows
        activate_cmd = f"{venv_path}\\Scripts\\activate"
        pip_cmd = f"{venv_path}\\Scripts\\pip"
    else:  # Unix/Linux/Mac
        activate_cmd = f"source {venv_path}/bin/activate"
        pip_cmd = f"{venv_path}/bin/pip"
    
    # Install requirements
    if not run_command(f"{pip_cmd} install --upgrade pip", "Upgrading pip"):
        return
    
    if not run_command(f"{pip_cmd} install -r requirements.txt", "Installing dependencies"):
        return
    
    print("\n🎉 Setup completed successfully!")
    print("\n📋 Next steps:")
    print("1. Activate your virtual environment:")
    if os.name == 'nt':  # Windows
        print(f"   {venv_path}\\Scripts\\activate")
    else:  # Unix/Linux/Mac
        print(f"   source {venv_path}/bin/activate")
    
    print("2. Start Jupyter Notebook:")
    print("   jupyter notebook")
    
    print("3. Open the notebooks in this order:")
    print("   - 01_data_exploration.ipynb")
    print("   - 02_basic_ml.ipynb")
    print("   - 03_neural_networks.ipynb")
    
    print("\n📚 Learning Resources:")
    print("- PyTorch Tutorials: https://pytorch.org/tutorials/")
    print("- scikit-learn Documentation: https://scikit-learn.org/stable/")
    print("- Machine Learning Mastery: https://machinelearningmastery.com/")
    
    print("\n🚀 Happy learning!")

if __name__ == "__main__":
    main()
