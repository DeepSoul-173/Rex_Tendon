"""Image augmentation pipelines for vision training."""

import cv2
import numpy as np
import random
from PIL import Image, ImageEnhance
import albumentations as A
from typing import Tuple, Optional


class AugmentationPipeline:
    """Manages image augmentation pipelines for training data."""

    @staticmethod
    def clean_alpha_channel(
        image_pil: Image.Image, alpha_threshold: int = 50
    ) -> Image.Image:
        """Clean alpha channel to remove semi-transparent artifacts.

        Args:
            image_pil: PIL image with alpha channel
            alpha_threshold: Threshold for alpha cleanup

        Returns:
            Image with cleaned alpha channel
        """
        if image_pil.mode != "RGBA":
            return image_pil

        r, g, b, alpha = image_pil.split()
        cleaned_alpha = alpha.point(lambda p: 0 if p < alpha_threshold else p)
        cleaned_alpha = cleaned_alpha.point(lambda p: 255 if p > alpha_threshold else 0)

        return Image.merge("RGBA", (r, g, b, cleaned_alpha))

    @staticmethod
    def get_tight_bbox(image_pil: Image.Image) -> Tuple[Optional[Tuple], Image.Image]:
        """Get tight bounding box by cleaning alpha channel first.

        Args:
            image_pil: PIL image

        Returns:
            Tuple of (bbox, cleaned_image)
        """
        if image_pil.mode == "RGBA":
            cleaned_image = AugmentationPipeline.clean_alpha_channel(image_pil)
            return cleaned_image.getbbox(), cleaned_image
        else:
            return image_pil.getbbox(), image_pil

    @staticmethod
    def resize_object(
        image_pil: Image.Image,
        target_size_ratio_range: Tuple[float, float] = (0.1, 0.8),
        bg_shape: Optional[Tuple[int, int]] = None,
    ) -> Image.Image:
        """Resize object as ratio of background's smallest dimension.

        Args:
            image_pil: Object image to resize
            target_size_ratio_range: Min/max size as ratio of background
            bg_shape: Background shape (width, height)

        Returns:
            Resized object image
        """
        if bg_shape is None:
            bg_shape = (640, 640)

        min_bg_dim = min(bg_shape[0], bg_shape[1])
        scale_factor = random.uniform(
            target_size_ratio_range[0], target_size_ratio_range[1]
        )
        target_object_size = int(min_bg_dim * scale_factor)

        original_width, original_height = image_pil.size
        aspect_ratio = original_width / original_height

        if original_width > original_height:
            new_width = target_object_size
            new_height = int(target_object_size / aspect_ratio)
        else:
            new_height = target_object_size
            new_width = int(target_object_size * aspect_ratio)

        new_width = max(1, new_width)
        new_height = max(1, new_height)

        resized_image = image_pil.resize((new_width, new_height), Image.LANCZOS)

        if resized_image.mode == "RGBA":
            resized_image = AugmentationPipeline.clean_alpha_channel(
                resized_image, alpha_threshold=50
            )

        return resized_image

    @staticmethod
    def rotate_object(
        image_pil: Image.Image, angle_range: Tuple[float, float] = (-180, 180)
    ) -> Image.Image:
        """Rotate object while keeping it fully visible.

        Args:
            image_pil: Object image to rotate
            angle_range: Min/max rotation angles in degrees

        Returns:
            Rotated object image
        """
        angle = random.uniform(angle_range[0], angle_range[1])

        if image_pil.mode != "RGBA":
            image_pil = image_pil.convert("RGBA")

        rotated_image = image_pil.rotate(
            angle,
            expand=True,
            resample=Image.BICUBIC,
            fillcolor=(0, 0, 0, 0),
        )

        rotated_image = AugmentationPipeline.clean_alpha_channel(
            rotated_image, alpha_threshold=50
        )
        return rotated_image

    @staticmethod
    def create_background_augmentation_pipeline() -> A.Compose:
        """Create albumentations pipeline for background augmentation.

        Returns:
            Albumentations composition for background augmentation
        """
        return A.Compose(
            [
                A.OneOf(
                    [
                        A.GaussianBlur(blur_limit=(3, 9), p=0.5),
                        A.MedianBlur(blur_limit=(3, 9), p=0.5),
                    ],
                    p=0.4,
                ),
                A.RandomBrightnessContrast(
                    brightness_limit=0.2, contrast_limit=0.2, p=0.5
                ),
                A.HueSaturationValue(
                    hue_shift_limit=10, sat_shift_limit=15, val_shift_limit=10, p=0.4
                ),
                A.CoarseDropout(
                    max_holes=5,
                    max_height=50,
                    max_width=50,
                    min_holes=1,
                    min_height=10,
                    min_width=10,
                    p=0.2,
                ),
                A.Downscale(
                    scale_min=0.5, scale_max=0.9, interpolation=cv2.INTER_NEAREST, p=0.1
                ),
            ]
        )

    @staticmethod
    def augment_background(image_cv2: np.ndarray) -> np.ndarray:
        """Apply augmentations to background image.

        Args:
            image_cv2: Background image as OpenCV array

        Returns:
            Augmented background image
        """
        transform = AugmentationPipeline.create_background_augmentation_pipeline()
        augmented = transform(image=image_cv2)
        return augmented["image"]

    @staticmethod
    def create_composite_augmentation_pipeline() -> A.Compose:
        """Create pipeline for augmenting final composite images.

        Returns:
            Albumentations composition for composite augmentation
        """
        return A.Compose(
            [
                A.OneOf(
                    [
                        A.GaussianBlur(blur_limit=(3, 7), p=0.6),
                        A.MedianBlur(blur_limit=(3, 7), p=0.4),
                        A.MotionBlur(blur_limit=(3, 7), p=0.3),
                    ],
                    p=0.5,
                ),
                A.RandomBrightnessContrast(
                    brightness_limit=(-0.2, 0.2), contrast_limit=(-0.2, 0.2), p=0.6
                ),
                A.HueSaturationValue(
                    hue_shift_limit=15, sat_shift_limit=20, val_shift_limit=15, p=0.5
                ),
                A.RandomGamma(gamma_limit=(80, 120), p=0.4),
                A.CLAHE(clip_limit=3.0, tile_grid_size=(8, 8), p=0.15),
                A.OneOf(
                    [
                        A.GaussNoise(var_limit=(10.0, 60.0), mean=0, p=0.6),
                        A.ISONoise(
                            color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=0.4
                        ),
                    ],
                    p=0.4,
                ),
                A.Sharpen(alpha=(0.1, 0.2), lightness=(0.9, 1.1), p=0.05),
                A.Emboss(alpha=(0.1, 0.3), strength=(0.2, 0.5), p=0.05),
                A.Downscale(
                    scale_min=0.7, scale_max=0.95, interpolation=cv2.INTER_LINEAR, p=0.1
                ),
                A.ImageCompression(quality_lower=60, quality_upper=95, p=0.15),
            ]
        )

    @staticmethod
    def apply_composite_augmentations(
        image_cv2: np.ndarray, augmentation_prob: float = 0.85
    ) -> np.ndarray:
        """Apply augmentations to final composite image.

        Args:
            image_cv2: Composite image as OpenCV array
            augmentation_prob: Probability of applying augmentations

        Returns:
            Augmented composite image
        """
        if random.random() > augmentation_prob:
            return image_cv2

        transform = AugmentationPipeline.create_composite_augmentation_pipeline()
        augmented = transform(image=image_cv2)
        return augmented["image"]

    @staticmethod
    def paste_object(
        background_pil: Image.Image, object_pil: Image.Image, position: str = "random"
    ) -> Tuple[Image.Image, Tuple[int, int, int, int]]:
        """Paste object onto background and return composite with bbox.

        Args:
            background_pil: Background image
            object_pil: Object image to paste
            position: Position strategy ("random" or (x, y) tuple)

        Returns:
            Tuple of (composite_image, bbox_xyxy)
        """
        bg_w, bg_h = background_pil.size

        tight_bbox, cleaned_object_pil = AugmentationPipeline.get_tight_bbox(object_pil)

        if tight_bbox is None:
            return background_pil.copy(), (0, 0, 0, 0)

        cropped_object_pil = cleaned_object_pil.crop(tight_bbox)
        obj_w, obj_h = cropped_object_pil.size

        if obj_w == 0 or obj_h == 0:
            return background_pil.copy(), (0, 0, 0, 0)

        # Resize if object is larger than background
        if obj_w > bg_w or obj_h > bg_h:
            aspect_ratio = obj_w / obj_h
            if obj_w > bg_w:
                obj_w = bg_w
                obj_h = int(obj_w / aspect_ratio)
            if obj_h > bg_h:
                obj_h = bg_h
                obj_w = int(obj_h * aspect_ratio)
            obj_w = max(1, obj_w)
            obj_h = max(1, obj_h)

            cropped_object_pil = cropped_object_pil.resize(
                (obj_w, obj_h), Image.LANCZOS
            )

            resized_bbox, cropped_object_pil = AugmentationPipeline.get_tight_bbox(
                cropped_object_pil
            )
            if resized_bbox is not None:
                cropped_object_pil = cropped_object_pil.crop(resized_bbox)
                obj_w, obj_h = cropped_object_pil.size

        # Determine position
        if position == "random":
            max_x = bg_w - obj_w
            max_y = bg_h - obj_h
            pos_x = random.randint(0, max(0, max_x))
            pos_y = random.randint(0, max(0, max_y))
        else:
            pos_x, pos_y = position
            pos_x = max(0, min(pos_x, bg_w - obj_w))
            pos_y = max(0, min(pos_y, bg_h - obj_h))

        # Paste object
        composite_image = background_pil.copy()

        if cropped_object_pil.mode == "RGBA":
            try:
                mask = cropped_object_pil.split()[3]
                composite_image.paste(cropped_object_pil, (pos_x, pos_y), mask)
            except IndexError:
                composite_image.paste(cropped_object_pil, (pos_x, pos_y))
        elif cropped_object_pil.mode == "LA":
            try:
                mask = cropped_object_pil.split()[1]
                rgb_object = cropped_object_pil.convert("RGBA")
                composite_image.paste(rgb_object, (pos_x, pos_y), mask)
            except IndexError:
                composite_image.paste(
                    cropped_object_pil.convert("RGBA"), (pos_x, pos_y)
                )
        else:
            composite_image.paste(cropped_object_pil.convert("RGBA"), (pos_x, pos_y))

        return composite_image, (pos_x, pos_y, obj_w, obj_h)

    @staticmethod
    def apply_lighting_effects(
        image_pil: Image.Image,
        brightness_factor_range: Tuple[float, float] = (0.5, 1.8),
        contrast_factor_range: Tuple[float, float] = (0.5, 1.8),
        gamma_range: Tuple[float, float] = (0.5, 2.0),
    ) -> Image.Image:
        """Apply lighting effects to PIL image.

        Args:
            image_pil: Input PIL image
            brightness_factor_range: Range for brightness adjustment
            contrast_factor_range: Range for contrast adjustment
            gamma_range: Range for gamma correction

        Returns:
            Image with lighting effects applied
        """
        # Apply brightness
        enhancer = ImageEnhance.Brightness(image_pil)
        factor = random.uniform(brightness_factor_range[0], brightness_factor_range[1])
        image_pil = enhancer.enhance(factor)

        # Apply contrast
        enhancer = ImageEnhance.Contrast(image_pil)
        factor = random.uniform(contrast_factor_range[0], contrast_factor_range[1])
        image_pil = enhancer.enhance(factor)

        # Apply gamma correction
        gamma = random.uniform(gamma_range[0], gamma_range[1])
        gamma = max(0.1, gamma)

        img_np = np.array(image_pil).astype(np.float32) / 255.0
        img_gamma_corrected_np = np.power(img_np, 1.0 / gamma)
        img_gamma_corrected_np = np.clip(img_gamma_corrected_np * 255.0, 0, 255).astype(
            np.uint8
        )

        return Image.fromarray(img_gamma_corrected_np)

