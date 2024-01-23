import json
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

# Example usage
images_dir = ''  # Update this path to your directory of COCO images
coco_json_path = ''  # Update this path to your json for your COCO images
output_dir = ''  # Update this path to your output directory to be converted to YOLO
visualize_bbox = True  # Set to False if you don't want to visualize and just want to convert only


def convert_coco_bbox_to_yolo(bbox, img_width, img_height):
    # Convert COCO bbox format to YOLO format
    x_min, y_min, width, height = bbox
    x_center = x_min + width / 2
    y_center = y_min + height / 2
    x_center /= img_width
    y_center /= img_height
    width /= img_width
    height /= img_height
    return x_center, y_center, width, height


def draw_bbox(ax, bbox, img_width, img_height, category_name, color):
    # Draw bounding box with the given YOLO format coordinates
    x_center, y_center, width, height = bbox
    x_min = (x_center - width / 2) * img_width
    y_min = (y_center - height / 2) * img_height
    rect = patches.Rectangle((x_min, y_min), width * img_width, height * img_height,
                             linewidth=2, edgecolor=color, facecolor='none')
    ax.add_patch(rect)
    plt.text(x_min, y_min, category_name, color='white', fontsize=8, bbox=dict(facecolor=color, alpha=0.5))


def generate_color_palette(n):
    # Generate distinct colors
    return plt.cm.get_cmap('hsv', n)


def convert_annotations(coco_json_path, images_dir, output_dir, visualize=False):
    with open(coco_json_path) as f:
        data = json.load(f)

    coco_category_to_yolo = {category['id']: idx for idx, category in enumerate(data['categories'])}
    categories = {category['id']: category['name'] for category in data['categories']}
    color_palette = generate_color_palette(len(categories))
    images_info = {img['id']: img for img in data['images']}
    annotations = data['annotations']

    for image_id, img_info in images_info.items():
        image_annotations = [ann for ann in annotations if ann['image_id'] == image_id]
        if not image_annotations:
            continue

        file_name = os.path.splitext(img_info['file_name'])[0] + '.txt'
        output_path = os.path.join(output_dir, file_name)

        if visualize:
            img_path = os.path.join(images_dir, img_info['file_name'])
            img = Image.open(img_path)
            fig, ax = plt.subplots(1)
            ax.imshow(img)

        with open(output_path, 'w') as f:
            for ann in image_annotations:
                category_id = ann['category_id']
                yolo_class = coco_category_to_yolo.get(category_id)
                if yolo_class is None:
                    continue

                bbox = ann['bbox']
                yolo_bbox = convert_coco_bbox_to_yolo(bbox, img_info['width'], img_info['height'])
                f.write(f"{yolo_class} {' '.join(map(str, yolo_bbox))}\n")

                if visualize:
                    color = color_palette(yolo_class)[:3]  # Get RGB color
                    draw_bbox(ax, yolo_bbox, img_info['width'], img_info['height'], categories[category_id], color)

        if visualize:
            plt.axis('off')
            plt.show()


os.makedirs(output_dir, exist_ok=True)
convert_annotations(coco_json_path, images_dir, output_dir, visualize_bbox)
