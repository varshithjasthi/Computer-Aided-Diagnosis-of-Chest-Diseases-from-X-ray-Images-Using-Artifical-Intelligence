import os
import pandas as pd
from PIL import Image

# ========== PATHS ==========
archive_dir = r"C:\Users\giriv\OneDrive\Desktop\working Projects\new x\archive"
csv_path = os.path.join(archive_dir, "Data_Entry_2017.csv")
output_dir = r"C:\Users\giriv\OneDrive\Desktop\working Projects\new x\dataset_224"
# ============================

MAX_IMAGES_PER_CLASS = 2000
IMAGE_SIZE = (224, 224)

os.makedirs(output_dir, exist_ok=True)

# Load CSV
df = pd.read_csv(csv_path)

# Remove duplicate rows
df = df.drop_duplicates(subset=["Image Index"])

# Keep ONLY single-label rows (no "|")
df = df[~df["Finding Labels"].str.contains("\|", regex=True)]

print("Total single-label unique images:", len(df))

# Create image lookup dictionary
image_paths = {}
for root, _, files in os.walk(archive_dir):
    for file in files:
        if file.endswith(".png"):
            image_paths[file] = os.path.join(root, file)

class_counts = {}
processed_images = set()   # Track already processed images

for _, row in df.iterrows():
    image_name = row["Image Index"]
    label = row["Finding Labels"]

    # Skip if image missing
    if image_name not in image_paths:
        continue

    # Skip if already processed
    if image_name in processed_images:
        continue

    if label not in class_counts:
        class_counts[label] = 0

    if class_counts[label] >= MAX_IMAGES_PER_CLASS:
        continue

    class_folder = os.path.join(output_dir, label)
    os.makedirs(class_folder, exist_ok=True)

    save_path = os.path.join(class_folder, image_name)

    # Skip if file already exists (important if script runs again)
    if os.path.exists(save_path):
        continue

    try:
        img = Image.open(image_paths[image_name]).convert("RGB")
        img = img.resize(IMAGE_SIZE)
        img.save(save_path)

        class_counts[label] += 1
        processed_images.add(image_name)

    except Exception as e:
        print(f"Error processing {image_name}: {e}")
        continue

print("\n✅ Dataset Created Successfully!")
print("\nImages per class:")
for cls, count in class_counts.items():
    print(f"{cls}: {count}")
