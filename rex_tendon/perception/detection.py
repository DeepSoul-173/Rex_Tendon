"""Object detection using YOLO models for real-time perception."""

import logging
from pathlib import Path
from typing import Optional, Tuple, Union, List, Dict, Any
import numpy as np

from ..configs.loaders import get_perception_config

logger = logging.getLogger(__name__)


class YOLODetector:
    """YOLO object detector with configurable model and parameters."""

    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        device: str = "cpu",
        confidence_threshold: float = 0.3,
        config=None,
    ):
        """Initialize YOLO detector.

        Args:
            model_path: Path to YOLO model (.pt or .onnx). If None, uses config.
            device: Device for inference ('cpu', '0', etc.)
            confidence_threshold: Minimum confidence for detections
            config: Optional config object to use
        """
        # Lazy import to avoid loading ultralytics when not needed
        try:
            from ultralytics import YOLO
        except ImportError:
            raise ImportError(
                "ultralytics is required for YOLO detection. Install with: pip install ultralytics"
            )

        if config is None:
            config = get_perception_config()

        if model_path is None:
            model_path = config.yolo_model_path

        if model_path is None:
            raise ValueError("No YOLO model path specified")

        self.model_path = Path(model_path)
        self.device = device
        self.confidence_threshold = confidence_threshold
        self.model = YOLO(str(model_path), task="detect")

    def detect(
        self,
        frame: np.ndarray,
        return_all: bool = False,
    ) -> Union[
        Tuple[Optional[np.ndarray], float, Optional[np.ndarray]],
        List[Tuple[np.ndarray, float, np.ndarray]],
    ]:
        """Detect objects in frame.

        Args:
            frame: Input image as numpy array
            return_all: If True, return all detections; if False, return best detection

        Returns:
            If return_all=False: (center, confidence, bbox) or (None, 0.0, None)
            If return_all=True: List of (center, confidence, bbox) tuples
        """
        try:
            results = self.model.predict(
                frame,
                conf=self.confidence_threshold,
                verbose=False,
                device=self.device,
            )

            if not results or len(results[0].boxes) == 0:
                return [] if return_all else (None, 0.0, None)

            detections = []
            for box in results[0].boxes:
                # Get bounding box coordinates
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                bbox = np.array([x1, y1, x2, y2])

                # Calculate center
                center = np.array([(x1 + x2) / 2, (y1 + y2) / 2])

                # Get confidence
                confidence = float(box.conf[0].cpu().numpy())

                detections.append((center, confidence, bbox))

            if return_all:
                return detections
            else:
                # Return detection with highest confidence
                if detections:
                    best_detection = max(detections, key=lambda x: x[1])
                    return best_detection
                else:
                    return (None, 0.0, None)

        except Exception as e:
            logger.error(f"YOLO detection error: {e}")
            return [] if return_all else (None, 0.0, None)

