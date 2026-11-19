import os
import logging
import yaml

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Path to config file
CONFIG_FILE = os.path.join(os.path.dirname(__file__), "..", "config", "config.yaml")


def load_config():
    """
    Load the entire configuration from config.yaml
    """
    try:
        with open(CONFIG_FILE, "r") as f:
            config = yaml.safe_load(f)
            if config is None:
                logger.warning("Config file is empty.")
                return {}
            return config
    except FileNotFoundError:
        logger.error(f"Config file not found at: {CONFIG_FILE}")
        return {}
    except Exception as e:
        logger.error(f"Error loading configuration: {e}")
        return {}


def get_project():
    """Get project metadata"""
    config = load_config()
    return config.get("project", {})


def get_paths():
    """Get all path configurations"""
    config = load_config()
    return config.get("paths", {})


def get_columns():
    """Get column related configurations"""
    config = load_config()
    return config.get("columns", {})


def get_preprocessing():
    """Get preprocessing configurations"""
    config = load_config()
    return config.get("preprocessing", {})


def get_modeling():
    """Get modeling configurations"""
    config = load_config()
    return config.get("modeling", {})


def get_training():
    """Get training configurations"""
    config = load_config()
    return config.get("training", {})


def get_evaluation():
    """Get evaluation configurations"""
    config = load_config()
    return config.get("evaluation", {})


def get_logging():
    """Get logging configurations"""
    config = load_config()
    return config.get("logging", {})


def get_reproducibility():
    """Get reproducibility settings"""
    config = load_config()
    return config.get("reproducibility", {})


def get_mlflow():
    """Get MLflow configurations (if enabled)"""
    config = load_config()
    return config.get("mlflow", {})


# Optional helper to get nested path easily
def get_data_path(key: str):
    """
    Helper to quickly get a specific data path.
    Example: get_data_path("raw")
    """
    paths = get_paths()
    return paths.get("data", {}).get(key, None)


def get_artifact_path(key: str):
    """
    Helper to quickly get a specific artifact path.
    Example: get_artifact_path("models")
    """
    paths = get_paths()
    return paths.get("artifacts", {}).get(key, None)