"""Dashboard visualization for stereo perception and 3D tracking."""

from typing import Optional, Dict, Any, Tuple
import numpy as np
import cv2
import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
from collections import deque

from .hand_tracking import draw_hand_landmarks


def setup_dashboard_figure(
    figure_width: int = 6,
    figure_height: int = 6,
    coordinate_limits: Optional[Dict[str, Dict[str, float]]] = None,
) -> Dict[str, Any]:
    """Setup matplotlib figure and axes for 3D trajectory dashboard.

    Args:
        figure_width: Figure width in inches
        figure_height: Figure height in inches
        coordinate_limits: Coordinate limits for each axis

    Returns:
        Dictionary containing figure and plot handles
    """
    if coordinate_limits is None:
        coordinate_limits = {
            "X": {"clip_min": -0.20, "clip_max": 0.20},
            "Y": {"clip_min": 0.0, "clip_max": 0.43},
            "Z": {"clip_min": -0.4, "clip_max": -0.10},
        }

    fig = plt.figure(figsize=(figure_width, figure_height), dpi=100, facecolor="black")
    gs = GridSpec(2, 2, height_ratios=[1, 1])

    # Create subplots for different projections
    ax_xz = fig.add_subplot(gs[0, 0])  # X-Z projection
    ax_zy = fig.add_subplot(
        gs[0, 1]
    )  # Z-Y projection (note: Y-Z would be more intuitive)
    ax_xy = fig.add_subplot(gs[1, 0])  # X-Y projection

    axes_3d_plots = [ax_xz, ax_zy, ax_xy]

    # Style all axes
    for ax in axes_3d_plots:
        ax.set_facecolor("#111111")
        if hasattr(ax, "spines"):
            for spine in ax.spines.values():
                spine.set_color("#777777")
        ax.tick_params(colors="#777777")
        ax.autoscale(enable=False)

    # Set axis limits
    ax_xz.set_xlim(
        coordinate_limits["X"]["clip_min"], coordinate_limits["X"]["clip_max"]
    )
    ax_xz.set_ylim(
        coordinate_limits["Z"]["clip_min"], coordinate_limits["Z"]["clip_max"]
    )

    ax_zy.set_xlim(
        coordinate_limits["Y"]["clip_min"], coordinate_limits["Y"]["clip_max"]
    )
    ax_zy.set_ylim(
        coordinate_limits["Z"]["clip_min"], coordinate_limits["Z"]["clip_max"]
    )

    ax_xy.set_xlim(
        coordinate_limits["X"]["clip_min"], coordinate_limits["X"]["clip_max"]
    )
    ax_xy.set_ylim(
        coordinate_limits["Y"]["clip_min"], coordinate_limits["Y"]["clip_max"]
    )

    # Set axis labels
    ax_xz.set_xlabel("X", color="white", fontsize=12, fontweight="bold")
    ax_xz.set_ylabel("Z", color="white", fontsize=12, fontweight="bold")

    ax_zy.set_xlabel("Y", color="white", fontsize=12, fontweight="bold")
    ax_zy.set_ylabel("Z", color="white", fontsize=12, fontweight="bold")

    ax_xy.set_xlabel("X", color="white", fontsize=12, fontweight="bold")
    ax_xy.set_ylabel("Y", color="white", fontsize=12, fontweight="bold")

    # Initialize plot elements for tip trajectory
    scat_xz = ax_xz.scatter([], [], s=40, c="#e74c3c")  # Red for current point
    (line_xz,) = ax_xz.plot([], [], lw=1.2, c="#3498db")  # Blue for trajectory

    scat_zy = ax_zy.scatter([], [], s=40, c="#e74c3c")
    (line_zy,) = ax_zy.plot([], [], lw=1.2, c="#3498db")

    scat_xy = ax_xy.scatter([], [], s=40, c="#e74c3c")
    (line_xy,) = ax_xy.plot([], [], lw=1.2, c="#3498db")

    # Initialize plot elements for finger trajectory
    scat_xz_finger = ax_xz.scatter([], [], s=40, c="#2ecc71")  # Green for finger
    (line_xz_finger,) = ax_xz.plot([], [], lw=1.2, c="#2ecc71")

    scat_zy_finger = ax_zy.scatter([], [], s=40, c="#2ecc71")
    (line_zy_finger,) = ax_zy.plot([], [], lw=1.2, c="#2ecc71")

    scat_xy_finger = ax_xy.scatter([], [], s=40, c="#2ecc71")
    (line_xy_finger,) = ax_xy.plot([], [], lw=1.2, c="#2ecc71")

    plot_handles = {
        "fig": fig,
        "ax_xz": ax_xz,
        "ax_zy": ax_zy,
        "ax_xy": ax_xy,
        "scat_xz": scat_xz,
        "line_xz": line_xz,
        "scat_xz_finger": scat_xz_finger,
        "line_xz_finger": line_xz_finger,
        "scat_zy": scat_zy,
        "line_zy": line_zy,
        "scat_zy_finger": scat_zy_finger,
        "line_zy_finger": line_zy_finger,
        "scat_xy": scat_xy,
        "line_xy": line_xy,
        "scat_xy_finger": scat_xy_finger,
        "line_xy_finger": line_xy_finger,
    }

    return plot_handles


def update_dashboard_plots(
    plot_handles: Dict[str, Any], tip_trajectory: deque, finger_trajectory: deque
) -> np.ndarray:
    """Update dashboard plots with new trajectory data.

    Args:
        plot_handles: Plot handles from setup_dashboard_figure
        tip_trajectory: Deque of 3D tip positions
        finger_trajectory: Deque of 3D finger positions

    Returns:
        Dashboard image as RGB numpy array
    """
    fig = plot_handles["fig"]
    ax_xz, ax_zy, ax_xy = (
        plot_handles["ax_xz"],
        plot_handles["ax_zy"],
        plot_handles["ax_xy"],
    )

    # Get plot elements
    scat_xz, line_xz = plot_handles["scat_xz"], plot_handles["line_xz"]
    scat_xz_finger, line_xz_finger = (
        plot_handles["scat_xz_finger"],
        plot_handles["line_xz_finger"],
    )

    scat_zy, line_zy = plot_handles["scat_zy"], plot_handles["line_zy"]
    scat_zy_finger, line_zy_finger = (
        plot_handles["scat_zy_finger"],
        plot_handles["line_zy_finger"],
    )

    scat_xy, line_xy = plot_handles["scat_xy"], plot_handles["line_xy"]
    scat_xy_finger, line_xy_finger = (
        plot_handles["scat_xy_finger"],
        plot_handles["line_xy_finger"],
    )

    # Add grid lines
    for ax in [ax_xz, ax_zy, ax_xy]:
        ax.grid(True, color="#333333", linestyle="-", linewidth=0.5)
        ax.axhline(y=0, color="#333333", linestyle="-", linewidth=0.5)
        ax.axvline(x=0, color="#333333", linestyle="-", linewidth=0.5)

    # Update tip trajectory
    if tip_trajectory:
        xs, ys, zs = np.stack(tip_trajectory).T

        # Update current point (last point)
        scat_xz.set_offsets(np.c_[xs[-1:], zs[-1:]])
        scat_zy.set_offsets(np.c_[ys[-1:], zs[-1:]])
        scat_xy.set_offsets(np.c_[xs[-1:], ys[-1:]])

        # Update trajectory lines
        line_xz.set_data(xs, zs)
        line_zy.set_data(ys, zs)
        line_xy.set_data(xs, ys)

    # Update finger trajectory
    if finger_trajectory:
        xs_f, ys_f, zs_f = np.stack(finger_trajectory).T

        # Update current point (last point)
        scat_xz_finger.set_offsets(np.c_[xs_f[-1:], zs_f[-1:]])
        scat_zy_finger.set_offsets(np.c_[ys_f[-1:], zs_f[-1:]])
        scat_xy_finger.set_offsets(np.c_[xs_f[-1:], ys_f[-1:]])

        # Update trajectory lines
        line_xz_finger.set_data(xs_f, zs_f)
        line_zy_finger.set_data(ys_f, zs_f)
        line_xy_finger.set_data(xs_f, ys_f)

    # Render to numpy array
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    buf = fig.canvas.buffer_rgba()
    img = np.asarray(buf, dtype=np.uint8).reshape(h, w, 4)

    # Convert RGBA to RGB and BGR (for OpenCV)
    return img[:, :, :3][:, :, ::-1]


def draw_detection_overlays(
    frame: np.ndarray,
    object_center: Optional[np.ndarray] = None,
    object_bbox: Optional[list] = None,
    finger_tip: Optional[np.ndarray] = None,
    mediapipe_results: Optional[Any] = None,
    object_color: Tuple[int, int, int] = (0, 255, 0),
    finger_color: Tuple[int, int, int] = (0, 255, 255),
    bbox_color: Tuple[int, int, int] = (0, 255, 0),
    point_radius: int = 6,
    bbox_thickness: int = 2,
) -> np.ndarray:
    """Draw detection overlays on frame.

    Args:
        frame: Input image
        object_center: Object center point (x, y)
        object_bbox: Object bounding box [x1, y1, x2, y2]
        finger_tip: Finger tip position (x, y)
        mediapipe_results: MediaPipe results for hand landmarks
        object_color: Color for object center point (BGR)
        finger_color: Color for finger tip point (BGR)
        bbox_color: Color for bounding box (BGR)
        point_radius: Radius for drawn points
        bbox_thickness: Thickness for bounding box

    Returns:
        Frame with overlays drawn
    """
    output_frame = frame.copy()

    # Draw object detection
    if object_center is not None:
        cv2.circle(
            output_frame, tuple(map(int, object_center)), point_radius, object_color, -1
        )

    if object_bbox is not None:
        x1, y1, x2, y2 = map(int, object_bbox)
        cv2.rectangle(output_frame, (x1, y1), (x2, y2), bbox_color, bbox_thickness)

    # Draw finger tip
    if finger_tip is not None:
        cv2.circle(
            output_frame, tuple(map(int, finger_tip)), point_radius, finger_color, -1
        )

    # Draw hand landmarks
    if mediapipe_results is not None:
        output_frame = draw_hand_landmarks(output_frame, mediapipe_results)

    return output_frame


def assemble_dashboard_view(
    left_frame: np.ndarray,
    right_frame: np.ndarray,
    plot_handles: Dict[str, Any],
    tip_trajectory: deque,
    finger_trajectory: deque,
    max_display_size: Tuple[int, int] = (1280, 720),
) -> np.ndarray:
    """Combine stereo frames and 3D plots into single dashboard view.

    Args:
        left_frame: Processed left camera frame
        right_frame: Processed right camera frame
        plot_handles: Plot handles from setup_dashboard_figure
        tip_trajectory: Deque of 3D tip positions
        finger_trajectory: Deque of 3D finger positions
        max_display_size: Maximum display size (width, height)

    Returns:
        Combined dashboard image
    """
    # Combine stereo frames horizontally
    stereo_display = np.hstack((left_frame, right_frame))

    # Generate 3D plot image
    plot_image = update_dashboard_plots(plot_handles, tip_trajectory, finger_trajectory)

    if plot_image.shape[0] == 0:
        return stereo_display

    # Resize plot to match stereo display height
    plot_height, plot_width = plot_image.shape[:2]
    target_height = stereo_display.shape[0]
    scale = target_height / plot_height
    plot_resized = cv2.resize(plot_image, (int(plot_width * scale), target_height))

    # Combine horizontally
    final_dashboard = np.hstack((stereo_display, plot_resized))

    # Resize to fit display limits if needed
    max_width, max_height = max_display_size
    h, w = final_dashboard.shape[:2]

    if h > max_height or w > max_width:
        scale = min(max_height / h, max_width / w)
        final_dashboard = cv2.resize(final_dashboard, (int(w * scale), int(h * scale)))

    return final_dashboard

