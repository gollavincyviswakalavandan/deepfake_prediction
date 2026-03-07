import os
import random
import shutil

source_dir = "extracted_frames/train_data"
dest_dir = "dataset"

print("Checking source path:", source_dir)

if not os.path.exists(source_dir):
    print("Source folder not found!")
    exit()

for class_name in ["real", "fake"]:
    class_path = os.path.join(source_dir, class_name)

    if not os.path.exists(class_path):
        print(f"{class_name} folder not found!")
        continue

    images = os.listdir(class_path)
    print(f"{class_name} images found:", len(images))

    random.shuffle(images)

    total = len(images)
    train_end = int(0.7 * total)
    val_end = train_end + int(0.15 * total)

    splits = {
        "train": images[:train_end],
        "val": images[train_end:val_end],
        "test": images[val_end:]
    }

    for split in splits:
        split_path = os.path.join(dest_dir, split, class_name)
        os.makedirs(split_path, exist_ok=True)

        for img in splits[split]:
            src = os.path.join(class_path, img)
            dst = os.path.join(split_path, img)
            shutil.copy(src, dst)

print("Dataset split completed!")