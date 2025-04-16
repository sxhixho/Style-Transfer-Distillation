import os, glob
from PIL import Image, ImageStat
from tqdm import tqdm

# Global configuration (can also be imported from a config module)
DARKNESS_THRESHOLD = 43.0
BRIGHTNESS_THRESHOLD = 240.0
LOW_CONTRAST_THRESHOLD = 15.0

class DataCleaner:
    """
    Handles image analysis and filtering based on brightness, contrast, or aspect ratio.
    """
    def calculate_brightness(self, image_path):
        try:
            img = Image.open(image_path).convert('L')
            stat = ImageStat.Stat(img)
            return stat.mean[0]
        except Exception as e:
            print(f"Warning: Could not calculate brightness for {image_path}: {e}")
            return None

    def is_image_bad(self, image_path, dark_thresh=DARKNESS_THRESHOLD, bright_thresh=BRIGHTNESS_THRESHOLD):
        brightness = self.calculate_brightness(image_path)
        if brightness is None:
            return True, "Error processing image"
        if brightness < dark_thresh:
            return True, f"Too dark (Brightness: {brightness:.1f})"
        if brightness > bright_thresh:
            return True, f"Too bright (Brightness: {brightness:.1f})"

        try:
            img = Image.open(image_path).convert('L')
            stat = ImageStat.Stat(img)
            if stat.stddev[0] < LOW_CONTRAST_THRESHOLD:
                return True, f"Low contrast (StdDev: {stat.stddev[0]:.1f})"
        except Exception:
            pass

        try:
            img = Image.open(image_path)
            width, height = img.size
            aspect_ratio = width / height
            if aspect_ratio > 3.0 or aspect_ratio < 0.33:
                return True, f"Extreme aspect ratio ({aspect_ratio:.2f})"
        except Exception:
            pass

        return False, "Image OK"

    def filter_content_images(self, content_dir, dry_run=True):
        print(f"--- Starting Image Filtering in '{content_dir}' ---")
        if not dry_run:
            print("WARNING: Files will be deleted!")
        image_paths = (glob.glob(os.path.join(content_dir, '*.jpg')) +
                       glob.glob(os.path.join(content_dir, '*.png')) +
                       glob.glob(os.path.join(content_dir, '*.jpeg')))
        if not image_paths:
            print("No images found to filter.")
            return
        removed_count = 0
        checked_count = 0
        for img_path in tqdm(image_paths, desc="Filtering Images"):
            checked_count += 1
            is_bad, reason = self.is_image_bad(img_path)
            if is_bad:
                action = "[Would Remove]" if dry_run else "[Removing]"
                print(f"{action} {os.path.basename(img_path)} - Reason: {reason}")
                if not dry_run:
                    try:
                        os.remove(img_path)
                        removed_count += 1
                    except Exception as e:
                        print(f"Error removing {img_path}: {e}")
        print("Filtering complete.")
        print(f"Checked: {checked_count} images. Removed (or to be removed): {removed_count}")

if __name__ == '__main__':
    # Test/demo for DataCleaner
    cleaner = DataCleaner()
    folder = "./"  # Change to your test folder
    cleaner.filter_content_images(folder, dry_run=True)
