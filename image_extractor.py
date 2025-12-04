import os
import shutil

# Set your source folder (where images actually are)
SOURCE_FOLDER = "archive/Images/Images"

# Output folder for dashboard
OUTPUT_FOLDER = "dataset"

# Valid image extensions
EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)

print("Scanning for images...")

image_paths = []

# Walk through ALL subfolders
for root, dirs, files in os.walk(SOURCE_FOLDER):
    for file in files:
        if file.lower().endswith(EXTS):
            full_path = os.path.join(root, file)
            image_paths.append(full_path)

print(f"Found {len(image_paths)} images.")

# Copy images into dataset/ and rename sequentially
for i, img_path in enumerate(image_paths, start=1):
    ext = os.path.splitext(img_path)[1].lower()
    new_filename = f"{i}{ext}"
    new_path = os.path.join(OUTPUT_FOLDER, new_filename)
    shutil.copy(img_path, new_path)

print(f"Copied all images to '{OUTPUT_FOLDER}/' successfully!")
