"""Generate synthetic training data by compositing objects onto backgrounds."""

import os
import random
import multiprocessing as mp
from pathlib import Path
from typing import List, Tuple, Optional

import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError
import typer
from typing_extensions import Annotated
from tqdm import tqdm
from rich.console import Console

from .augmentation import AugmentationPipeline

console = Console()
app = typer.Typer(help="Synthetic data generation utilities")

SUPPORTED_IMAGE_FORMATS = (".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".gif")


def generate_synthetic_data(
    objects_dir: str,
    backgrounds_dir: str,
    num_images: int = 1000,
) -> None:
    """Wrapper function to generate synthetic data.

    Args:
        objects_dir: Directory containing object images
        backgrounds_dir: Directory containing background images
        num_images: Number of synthetic images to generate
    """
    # Convert strings to Path objects
    objects_path = Path(objects_dir)
    backgrounds_path = Path(backgrounds_dir)

    # Call the typer command function directly with defaults
    generate(
        objects_dir=objects_path,
        backgrounds_dir=backgrounds_path,
        output_dir=Path("./synthetic_dataset"),
        num_images=num_images,
        num_workers=4,
        class_names=["object"],
        object_size_range=(0.1, 0.7),
        background_size=(640, 640),
        augmentation_probability=0.85,
        validation_split=0.2,
    )


def load_images_from_folder(
    folder_path: Path, image_type: str = "object"
) -> List[Image.Image]:
    """Load images from folder and convert to appropriate format.

    Args:
        folder_path: Path to folder containing images
        image_type: Type of images ("object" or "background")

    Returns:
        List of loaded PIL images
    """
    images = []
    if not folder_path.exists():
        console.print(f"[yellow]Warning: Folder not found: {folder_path}[/yellow]")
        return images

    for filename in os.listdir(folder_path):
        if filename.lower().endswith(SUPPORTED_IMAGE_FORMATS):
            try:
                img_path = folder_path / filename
                img = Image.open(img_path)
                if image_type == "object":
                    images.append(img.convert("RGBA"))
                else:
                    images.append(img.convert("RGB"))
            except UnidentifiedImageError:
                console.print(
                    f"[yellow]Warning: Could not identify image file: {filename}[/yellow]"
                )
            except Exception as e:
                console.print(f"[red]Error loading image {filename}: {e}[/red]")

    if not images:
        console.print(f"[yellow]Warning: No images loaded from {folder_path}[/yellow]")
    return images


def list_image_paths(folder_path: Path) -> List[Path]:
    """Return list of image file paths without loading them into memory.

    Args:
        folder_path: Path to folder containing images

    Returns:
        List of image file paths
    """
    image_paths = []
    if not folder_path.exists():
        console.print(f"[yellow]Warning: Folder not found: {folder_path}[/yellow]")
        return image_paths

    for filename in os.listdir(folder_path):
        if filename.lower().endswith(SUPPORTED_IMAGE_FORMATS):
            image_paths.append(folder_path / filename)

    return image_paths


def bbox_to_yolo_format(
    bbox: Tuple[int, int, int, int],
    image_width: int,
    image_height: int,
    class_id: int = 0,
) -> Tuple[int, float, float, float, float]:
    """Convert bounding box from pixel coordinates to YOLO format.

    Args:
        bbox: Bounding box as (x, y, w, h)
        image_width: Image width in pixels
        image_height: Image height in pixels
        class_id: Class ID for the object

    Returns:
        YOLO format tuple (class_id, center_x, center_y, width, height)
    """
    x, y, w, h = bbox
    center_x = (x + w / 2) / image_width
    center_y = (y + h / 2) / image_height
    norm_width = w / image_width
    norm_height = h / image_height
    return class_id, center_x, center_y, norm_width, norm_height


def save_yolo_annotation(annotation_path: Path, yolo_bbox: Tuple):
    """Save YOLO format annotation to text file.

    Args:
        annotation_path: Path to save annotation file
        yolo_bbox: YOLO format bounding box tuple
    """
    class_id, center_x, center_y, width, height = yolo_bbox
    with open(annotation_path, "w") as f:
        f.write(f"{class_id} {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}\n")


def generate_single_image(args_tuple) -> Tuple[bool, str, int, str]:
    """Worker function to generate a single synthetic image.

    Args:
        args_tuple: Tuple containing all arguments for image generation

    Returns:
        Tuple of (success, error_message, image_index, split_type)
    """
    (
        image_index,
        object_images_pil,
        background_image_paths,
        object_size_range,
        background_size,
        augmentation_probability,
        validation_split,
        output_dirs,
        worker_id,
    ) = args_tuple

    try:
        # Set random seed for reproducibility
        random.seed(image_index + worker_id * 10000)
        np.random.seed(image_index + worker_id * 10000)

        # Select random object and background
        obj_pil = random.choice(object_images_pil).copy()
        bg_path = random.choice(background_image_paths)

        try:
            bg_pil = Image.open(bg_path).convert("RGB")
        except Exception as e:
            return (
                False,
                f"Failed to load background image '{bg_path}': {e}",
                image_index,
                None,
            )

        # Resize background
        bg_target_width, bg_target_height = background_size
        bg_pil = bg_pil.resize((bg_target_width, bg_target_height), Image.LANCZOS)

        # Transform object
        obj_pil = AugmentationPipeline.resize_object(
            obj_pil,
            target_size_ratio_range=object_size_range,
            bg_shape=(bg_target_width, bg_target_height),
        )
        obj_pil = AugmentationPipeline.rotate_object(obj_pil, angle_range=(-180, 180))

        # Augment background
        bg_cv = cv2.cvtColor(np.array(bg_pil), cv2.COLOR_RGB2BGR)
        augmented_bg_cv = AugmentationPipeline.augment_background(bg_cv)
        augmented_bg_pil = Image.fromarray(
            cv2.cvtColor(augmented_bg_cv, cv2.COLOR_BGR2RGB)
        )

        # Paste object onto background
        composite_pil, bbox = AugmentationPipeline.paste_object(
            augmented_bg_pil, obj_pil
        )

        # Apply final augmentations
        composite_cv = cv2.cvtColor(np.array(composite_pil), cv2.COLOR_RGB2BGR)
        augmented_composite_cv = AugmentationPipeline.apply_composite_augmentations(
            composite_cv,
            augmentation_prob=augmentation_probability,
        )
        final_image_pil = Image.fromarray(
            cv2.cvtColor(augmented_composite_cv, cv2.COLOR_BGR2RGB)
        )

        # Determine train/val split
        is_val = random.random() < validation_split
        split_type = "val" if is_val else "train"
        img_dir = output_dirs[f"{split_type}_images"]
        lbl_dir = output_dirs[f"{split_type}_labels"]

        # Save image and annotation
        output_filename = f"synthetic_image_{image_index:05d}.png"
        output_path = img_dir / output_filename
        annotation_filename = f"synthetic_image_{image_index:05d}.txt"
        annotation_path = lbl_dir / annotation_filename

        final_image_pil.save(output_path)

        # Create YOLO annotation
        image_width, image_height = final_image_pil.size
        yolo_bbox = bbox_to_yolo_format(bbox, image_width, image_height, class_id=0)
        save_yolo_annotation(annotation_path, yolo_bbox)

        return True, "", image_index, split_type

    except Exception as e:
        return (
            False,
            f"Error processing image {image_index}: {str(e)}",
            image_index,
            None,
        )


@app.command()
def generate(
    objects_dir: Annotated[
        Path,
        typer.Option(help="Directory containing object images (PNG with transparency)"),
    ],
    backgrounds_dir: Annotated[
        Path, typer.Option(help="Directory containing background images")
    ],
    output_dir: Annotated[
        Path, typer.Option(help="Directory to save generated dataset")
    ] = Path("./synthetic_dataset"),
    num_images: Annotated[
        int, typer.Option(help="Number of synthetic images to generate")
    ] = 1000,
    num_workers: Annotated[int, typer.Option(help="Number of worker processes")] = 4,
    class_names: Annotated[List[str], typer.Option(help="Class names for dataset")] = [
        "object"
    ],
    object_size_range: Annotated[
        Tuple[float, float], typer.Option(help="Object size range as min,max")
    ] = (0.1, 0.7),
    background_size: Annotated[
        Tuple[int, int], typer.Option(help="Background size as width,height")
    ] = (640, 640),
    augmentation_probability: Annotated[
        float, typer.Option(help="Probability of applying augmentations")
    ] = 0.85,
    validation_split: Annotated[
        float, typer.Option(help="Fraction of data for validation")
    ] = 0.2,
):
    """Generate synthetic training dataset."""

    console.print(f"Loading object images from: {objects_dir}")
    object_images_pil = load_images_from_folder(objects_dir, image_type="object")

    console.print(f"Scanning background images from: {backgrounds_dir}")
    background_image_paths = list_image_paths(backgrounds_dir)

    if not object_images_pil:
        console.print("[red]Error: No object images loaded. Exiting.[/red]")
        raise typer.Exit(1)
    if not background_image_paths:
        console.print("[red]Error: No background images loaded. Exiting.[/red]")
        raise typer.Exit(1)

    # Create output directories
    train_img_dir = output_dir / "train" / "images"
    train_lbl_dir = output_dir / "train" / "labels"
    val_img_dir = output_dir / "val" / "images"
    val_lbl_dir = output_dir / "val" / "labels"

    output_dirs = {
        "train_images": train_img_dir,
        "train_labels": train_lbl_dir,
        "val_images": val_img_dir,
        "val_labels": val_lbl_dir,
    }

    for directory in output_dirs.values():
        directory.mkdir(parents=True, exist_ok=True)

    console.print(
        f"Generating {num_images} synthetic images using {num_workers} workers..."
    )

    # Prepare worker arguments
    worker_args = []
    for i in range(num_images):
        worker_id = i % num_workers
        worker_args.append(
            (
                i,
                object_images_pil,
                background_image_paths,
                object_size_range,
                background_size,
                augmentation_probability,
                validation_split,
                output_dirs,
                worker_id,
            )
        )

    # Generate images
    if num_workers == 1:
        results = []
        for worker_arg in tqdm(worker_args, desc="Generating images"):
            results.append(generate_single_image(worker_arg))
    else:
        with mp.Pool(processes=num_workers) as pool:
            results = list(
                tqdm(
                    pool.imap(generate_single_image, worker_args),
                    total=len(worker_args),
                    desc="Generating images",
                )
            )

    # Process results
    successful = 0
    failed = 0
    train_count = 0
    val_count = 0

    for success, error_msg, image_index, split_type in results:
        if success:
            successful += 1
            if split_type == "train":
                train_count += 1
            elif split_type == "val":
                val_count += 1
        else:
            failed += 1
            console.print(
                f"[red]Failed to generate image {image_index}: {error_msg}[/red]"
            )

    # Create dataset.yaml
    dataset_yaml_path = output_dir / "dataset.yaml"
    with open(dataset_yaml_path, "w") as f_yaml:
        f_yaml.write(
            f"""# Auto-generated dataset config

path: {output_dir}
train: train/images
val: val/images

autodownload: false

names:
"""
        )
        for i, name in enumerate(class_names):
            f_yaml.write(f"  {i}: {name}\n")

    # Print summary
    console.print("\nSynthetic data generation completed:")
    console.print(f"  Successful: {successful} images")
    console.print(f"  Failed: {failed} images")
    console.print(f"  Train split: {train_count} images")
    console.print(f"  Val split: {val_count} images")
    console.print(f"Dataset saved in: {output_dir}")
    console.print(f"dataset.yaml written to: {dataset_yaml_path}")


@app.command()
def generate_data(
    objects_dir: str = typer.Argument(..., help="Directory containing object images"),
    backgrounds_dir: str = typer.Argument(
        ..., help="Directory containing background images"
    ),
    num_images: int = typer.Option(
        1000, "--num-images", help="Number of synthetic images to generate"
    ),
) -> None:
    """Generate synthetic training data."""
    generate_synthetic_data(objects_dir, backgrounds_dir, num_images)


if __name__ == "__main__":
    app()

