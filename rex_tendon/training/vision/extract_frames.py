"""K-means based frame extraction for smart frame selection."""

import logging
from pathlib import Path

import cv2
import numpy as np
import typer
from tqdm import tqdm
from rich.console import Console

from sklearn.cluster import KMeans

logger = logging.getLogger(__name__)
console = Console()
app = typer.Typer(help="Frame extraction utilities")


def extract_features(frame: np.ndarray, size: tuple = (64, 64)) -> np.ndarray:
    """Resize then flatten to a vector in [0,1]. Cheap but surprisingly solid."""
    small = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
    return small.reshape(-1).astype(np.float32) / 255.0


def pick_representatives(features: np.ndarray, n_clusters: int) -> list:
    """Run k-means and return indices of frames closest to each centroid."""
    k = min(n_clusters, len(features))
    km = KMeans(n_clusters=k, random_state=0, n_init="auto").fit(features)
    dists = ((features - km.cluster_centers_[km.labels_]) ** 2).sum(axis=1)
    reps = []
    for c in range(k):
        cluster_idx = np.where(km.labels_ == c)[0]
        reps.append(cluster_idx[np.argmin(dists[cluster_idx])])
    return sorted(reps)


def process_video(video_path: Path, output_dir: Path, n_frames: int, step: int) -> int:
    """Process single video with k-means frame selection."""
    cap = cv2.VideoCapture(str(video_path))
    feats, frames, idx = [], [], 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % step == 0:
            feats.append(extract_features(frame))
            frames.append(frame)
        idx += 1
    cap.release()

    if not feats:
        return 0

    feats = np.asarray(feats)
    chosen = pick_representatives(feats, n_frames)
    basename = video_path.stem

    for rank, i in enumerate(chosen):
        fname = output_dir / f"{basename}_f{step * i:06d}_k{rank}.jpg"
        cv2.imwrite(str(fname), frames[i])

    return len(chosen)


def get_video_files(input_path: Path) -> list[Path]:
    """Get list of video files from input path (file or directory)."""
    if input_path.is_file():
        # Check if it's a video file
        if input_path.suffix.lower() in [".mp4", ".avi", ".mov", ".mkv", ".webm"]:
            return [input_path]
        else:
            console.print(
                f"[red]Error: {input_path} is not a supported video file[/red]"
            )
            raise typer.Exit(1)
    elif input_path.is_dir():
        # Get all video files from directory
        video_extensions = ["*.mp4", "*.avi", "*.mov", "*.mkv", "*.webm"]
        video_files = []
        for ext in video_extensions:
            video_files.extend(input_path.glob(ext))
            video_files.extend(input_path.glob(ext.upper()))
        return sorted(video_files)
    else:
        console.print(f"[red]Error: Input path not found: {input_path}[/red]")
        raise typer.Exit(1)


def extract_frames_kmeans(
    video_path: str, output_dir: str, num_frames: int, step: int = 5
) -> None:
    """Extract representative frames using K-means clustering.

    This function provides the interface expected by the main module.

    Args:
        video_path: Path to video file or directory containing videos
        output_dir: Directory to save extracted frames
        num_frames: Number of frames to keep per video
        step: Analyze every step-th frame
    """
    input_path = Path(video_path)
    output_path = Path(output_dir)

    if not input_path.exists():
        logger.error(f"Error: Input path not found: {input_path}")
        return

    output_path.mkdir(parents=True, exist_ok=True)

    # Get list of video files to process
    video_files = get_video_files(input_path)

    if not video_files:
        logger.info("No supported video files found.")
        return

    logger.info(f"Found {len(video_files)} video file(s) to process")

    # Process all videos with progress bar
    total_kept = 0
    for video_file in tqdm(video_files, desc="Processing videos"):
        try:
            kept = process_video(video_file, output_path, num_frames, step)
            logger.info(f"{video_file.name}: kept {kept} frames")
            total_kept += kept
        except Exception as e:
            logger.error(f"Error processing {video_file.name}: {e}")

    logger.info(f"Total frames extracted: {total_kept}")


@app.command()
def extract_frames(
    video_path: str = typer.Argument(..., help="Path to input video"),
    output_dir: str = typer.Argument(..., help="Output directory for frames"),
    num_frames: int = typer.Argument(..., help="Number of frames to extract"),
) -> None:
    """Extract representative frames using k-means."""
    extract_frames_kmeans(video_path, output_dir, num_frames)


if __name__ == "__main__":
    app()

