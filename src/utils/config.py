import os

from dotenv import load_dotenv

load_dotenv()


class Config:
    """Centralized configuration for GaitGuard."""

    # Hardware
    DEVICE_TYPE = os.getenv("DEVICE_TYPE", "raspberry_pi_5")
    POSE_BACKEND = os.getenv("POSE_BACKEND", "mediapipe")

    # Baseline & Drift
    BASELINE_WINDOW_HOURS = int(
        os.getenv("BASELINE_WINDOW_HOURS", "4")
    )
    DRIFT_WINDOW_MINUTES = int(
        os.getenv("DRIFT_WINDOW_MINUTES", "10")
    )
    FATIGUE_RISK_THRESHOLD = float(
        os.getenv("FATIGUE_RISK_THRESHOLD", "0.7")
    )

    # Logging
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

    # GPU
    ENABLE_GPU = os.getenv("ENABLE_GPU", "false").lower() == "true"

    # Paths
    BASE_DIR = os.path.dirname(
        os.path.dirname(
            os.path.dirname(os.path.abspath(__file__))
        )
    )

    DATA_DIR = os.path.join(BASE_DIR, "data")
    LOGS_DIR = os.path.join(BASE_DIR, "logs")

    # Ensure required directories exist
    os.makedirs(LOGS_DIR, exist_ok=True)


config = Config()
