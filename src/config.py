# Loads and validates configuration from files
# src/config.py
import yaml
from pathlib import Path

def load_config(config_path: Path = Path("config/main_config.yaml")) -> dict:
    """Loads a YAML configuration file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config

# Load the main configuration
config = load_config()