import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import shutil
import os
import yaml

# UPDATE THESE PATHS
hasty_coco_dir = ''  # update this to the absolute path of your hasty directory in coco format
yolo_dir = ''  # update this to the absolute path of your desired output folder

##############################################
# YOU SHOULDN'T NEED TO CHANGE ANYTHING BELOW
##############################################

# Input paths
coco_annotations_path = hasty_coco_dir + 'annotations.json'
coco_images_train_path = hasty_coco_dir + 'images/train/'
coco_images_val_path = hasty_coco_dir + 'images/val/'

# Output paths
yolo_images_train_path = yolo_dir + 'images/train/'
yolo_images_val_path = yolo_dir + 'images/val/'
yolo_labels_train_path = yolo_dir + 'labels/train/'
yolo_labels_val_path = yolo_dir + 'labels/val/'

# Ensure the output directories exist
os.makedirs(yolo_images_train_path, exist_ok=True)
os.makedirs(yolo_images_val_path, exist_ok=True)
os.makedirs(yolo_labels_train_path, exist_ok=True)
os.makedirs(yolo_labels_val_path, exist_ok=True)


def copy_all_files(src_dir, dst_dir):
    os.makedirs(dst_dir, exist_ok=True)  # create output dir
    files = os.listdir(src_dir)
    for file_name in files:
        src_file = os.path.join(src_dir, file_name)
        dst_file = os.path.join(dst_dir, file_name)
        if os.path.isfile(src_file):  # check if file or directory
            shutil.copy(src_file, dst_file)


def create_yaml_file(yaml_path, category_names):
    yaml_content = {
        "path": "",  # dataset root directory in ultralytics
        "train": "images/train",  # Relative path to training images
        "val": "images/val",  # Relative path to validation images
        "test": "",  # Optional, path to test images
        "names": {idx: name for idx, (_, name) in enumerate(sorted(category_names.items()))}
    }

    with open(yaml_path, 'w') as file:
        yaml.dump(yaml_content, file, sort_keys=False, default_flow_style=False)


def convert_coco_bbox_to_yolo(bbox, img_width, img_height):
    x_min, y_min, width, height = bbox
    x_center = x_min + width / 2
    y_center = y_min + height / 2
    x_center /= img_width
    y_center /= img_height
    width /= img_width
    height /= img_height
    return x_center, y_center, width, height


def draw_bbox(ax, bbox, img_width, img_height, category_name, color):
    x_center, y_center, width, height = bbox
    x_min = (x_center - width / 2) * img_width
    y_min = (y_center - height / 2) * img_height
    rect = patches.Rectangle((x_min, y_min), width * img_width, height * img_height,
                             linewidth=2, edgecolor=color, facecolor='none')
    ax.add_patch(rect)
    plt.text(x_min, y_min, category_name, color='white', fontsize=8, bbox=dict(facecolor=color, alpha=0.5))


def generate_color_palette(n):
    return plt.cm.get_cmap('hsv', n)


def convert_annotations(coco_json_path, train_images_dir, val_images_dir, train_output_dir, val_output_dir,
                        visualize=False):
    with open(coco_json_path) as f:
        data = json.load(f)

    coco_category_to_yolo = {category['id']: idx for idx, category in enumerate(data['categories'])}
    category_names = {category['id']: category['name'] for category in data['categories']}
    color_palette = generate_color_palette(len(category_names))
    images_info = {img['id']: img for img in data['images']}
    annotations = data['annotations']

    # Create yaml file based on category names
    directory_path = yolo_dir.rstrip("/")
    yaml_path = yolo_dir + os.path.basename(directory_path) + '.yaml'

    create_yaml_file(yaml_path, category_names)

    for image_id, img_info in images_info.items():
        image_annotations = [ann for ann in annotations if ann['image_id'] == image_id]
        if not image_annotations:
            continue

        if os.path.exists(os.path.join(train_images_dir, img_info['file_name'])):
            output_dir = train_output_dir
            img_path = os.path.join(train_images_dir, img_info['file_name'])
        elif os.path.exists(os.path.join(val_images_dir, img_info['file_name'])):
            output_dir = val_output_dir
            img_path = os.path.join(val_images_dir, img_info['file_name'])
        else:
            continue

        file_name = os.path.splitext(img_info['file_name'])[0] + '.txt'
        output_path = os.path.join(output_dir, file_name)

        with open(output_path, 'w') as f:
            if visualize:
                img = Image.open(img_path)
                fig, ax = plt.subplots(1)
                ax.imshow(img)

            for ann in image_annotations:
                category_id = ann['category_id']
                yolo_class = coco_category_to_yolo.get(category_id)
                if yolo_class is None:
                    continue

                bbox = ann['bbox']
                yolo_bbox = convert_coco_bbox_to_yolo(bbox, img_info['width'], img_info['height'])
                f.write(f"{yolo_class} {' '.join(map(str, yolo_bbox))}\n")

                if visualize:
                    color = color_palette(yolo_class)[:3]
                    draw_bbox(ax, yolo_bbox, img_info['width'], img_info['height'], category_names[category_id], color)

            if visualize:
                plt.axis('off')
                plt.show()


# Populating the yolo images folder
copy_all_files(coco_images_train_path, yolo_images_train_path)
copy_all_files(coco_images_val_path, yolo_images_val_path)

# Populating the yolo labels folder (visualize = True to visualize)
convert_annotations(coco_annotations_path, train_images_dir=coco_images_train_path, val_images_dir=coco_images_val_path,
                    train_output_dir=yolo_labels_train_path, val_output_dir=yolo_labels_val_path, visualize=False)

