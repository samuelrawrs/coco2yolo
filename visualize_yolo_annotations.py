import os
from PIL import Image, ImageDraw

images_dir = ''  # Update this with yolo images/train or images/val directory path
labels_dir = ''  # Update this with yolo labels/train or labels/val directory path


def draw_yolo_bbox(img, bbox, color='red'):
    draw = ImageDraw.Draw(img)
    width, height = img.size
    x_center, y_center, w, h = bbox
    x_min = int((x_center - w / 2) * width)
    y_min = int((y_center - h / 2) * height)
    x_max = int((x_center + w / 2) * width)
    y_max = int((y_center + h / 2) * height)
    draw.rectangle([x_min, y_min, x_max, y_max], outline=color, width=2)


def visualize_yolo_annotations(images_dir, labels_dir):
    for filename in os.listdir(images_dir):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            img_path = os.path.join(images_dir, filename)
            label_path = os.path.join(labels_dir, os.path.splitext(filename)[0] + '.txt')

            if os.path.exists(label_path):
                img = Image.open(img_path)

                with open(label_path) as f:
                    for line in f.readlines():
                        class_id, x_center, y_center, width, height = map(float, line.split())
                        draw_yolo_bbox(img, (x_center, y_center, width, height))

                img.show()


visualize_yolo_annotations(images_dir, labels_dir)
