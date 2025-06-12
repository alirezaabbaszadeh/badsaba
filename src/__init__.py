import pickle
import logging
from pathlib import Path
import os

# Create logs directory if it doesn't exist
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)

# Basic logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(log_dir, "app.log")),
        logging.StreamHandler() # Also print logs to console
    ]
)

# Get a logger instance for this module
logger = logging.getLogger(__name__)

def save_object(file_path: Path, obj: object):
    """
    Saves a Python object to a specified file path using pickle.

    Args:
        file_path (Path): The path where the object will be saved.
        obj (object): The Python object to save.
    """
    try:
        # Ensure the directory for the file exists
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)
        logger.info(f"✅ Object saved successfully to: {file_path}")
    except Exception as e:
        logger.error(f"❌ Error saving object to {file_path}: {e}")
        raise

def load_object(file_path: Path) -> object:
    """
    Loads a Python object from a file path using pickle.

    Args:
        file_path (Path): The path of the file to load.

    Returns:
        object: The loaded Python object.
    """
    try:
        with open(file_path, "rb") as file_obj:
            obj = pickle.load(file_obj)
        logger.info(f"✅ Object loaded successfully from: {file_path}")
        return obj
    except FileNotFoundError:
        logger.error(f"❌ File not found at: {file_path}")
        raise
    except Exception as e:
        logger.error(f"❌ Error loading object from {file_path}: {e}")
        raise