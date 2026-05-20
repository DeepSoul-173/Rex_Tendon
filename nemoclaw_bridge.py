import subprocess
import logging

logger = logging.getLogger(__name__)

def bend_tentacle(pitch: float, yaw: float, extension: float):
    """
    Bend the tentacle using the CLI.
    """
    cmd = [
        "poetry", "run", "python", "-m", "rex_tendon", "control", "bend",
        "--pitch", str(pitch),
        "--yaw", str(yaw),
        "--extension", str(extension)
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        logger.info(f"bend_tentacle output: {result.stdout}")
        return result.stdout
    except subprocess.CalledProcessError as e:
        logger.error(f"Error bending tentacle: {e.stderr}")
        raise

def trigger_grasp():
    """
    Trigger grasp using the CLI.
    """
    cmd = ["poetry", "run", "python", "-m", "rex_tendon", "control", "grasp"]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        logger.info(f"trigger_grasp output: {result.stdout}")
        return result.stdout
    except subprocess.CalledProcessError as e:
        logger.error(f"Error triggering grasp: {e.stderr}")
        raise

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("NemoClaw Bridge initialized. Ready for standalone control.")
