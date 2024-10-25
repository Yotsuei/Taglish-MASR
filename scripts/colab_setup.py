# colab_setup.py
import os
import subprocess
from google.colab import drive

def mount_drive():
    """Mounts Google Drive for accessing and saving files."""
    drive.mount('/content/drive')
    print("Google Drive mounted at /content/drive")

def install_requirements(requirements_path='requirements.txt'):
    """Installs necessary packages from requirements.txt."""
    subprocess.check_call(['pip', 'install', '-r', requirements_path])
    print("Dependencies installed successfully.")

def set_data_paths():
    """Sets paths for data directories (adjust paths as needed)."""
    paths = {
        'training_data': '/content/drive/MyDrive/Taglish-MASR/data/training/',
        'evaluation_data': '/content/drive/MyDrive/Taglish-MASR/data/evaluation/',
        'common_voice_data': '/content/drive/MyDrive/Taglish-MASR/data/common-voice/'
    }
    return paths

def setup_environment():
    """Performs all setup steps for Colab environment."""
    print("Setting up Colab environment...")
    mount_drive()
    install_requirements()
    paths = set_data_paths()
    print("Setup complete. Data paths configured:", paths)
    return paths

if __name__ == '__main__':
    setup_environment()
