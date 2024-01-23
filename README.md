# COCO2YOLO

So you wanna convert your COCO format to YOLO format, you've come to the right place!

## Scripts
1. [hasty2ultralytics.py](hasty2ultralytics.py): main script for converting from hasty COCO to ultralytics YOLOv8 format
2. [coco2yolo_viz_color.py](coco2yolo_viz_color.py): basic script for generating YOLO labels with coloured visualizations 
of the bounding boxes (without yaml)
3. [visualize_yolo_annotations.py](visualize_yolo_annotations.py): for visualizing YOLO labels

## Important: Instructions for hasty2ultralytics
Before you run the main script, it is vital to make your file structure exactly as show below. This includes the proper 
naming of the important files to `annotations.json`, as well as the `images`, `train` and `val` folders. Image names 
will remain as their original names in the output.

After you've prepared the dataset into the COCO Format, update [hasty2ultralytics.py](hasty2ultralytics.py) with the 
following 2 changes:
1. `hasty_coco_dir`: your dataset directory in COCO
2. `yolo_dir`: your output directory in YOLO

After running the script, there's one more important step!
The `path` variable in the `<dir_name>.yaml` file will be unfilled. You'll need to manually fill this with the dataset 
root directory in ultralytics. It might look something like this:
```yaml
path: ../datasets/<your-dataset-name>  # dataset root dir
```


## COCO Format (Input)

```
├── annotations.json
├── images
    ├── train
        ├── <image1>.jpg
        ├── <image2>.jpg
        └── <image3>.jpg
    └── val
        ├── <image4>.jpg
        ├── <image5>.jpg
        └── <image6>.jpg
```

## YOLO Format (Output)

```
├── <your-dataset-name>.yaml
├── images
    ├── train
        ├── <image1>.jpg
        ├── <image2>.jpg
        └── <image3>.jpg
    └── val
        ├── <image4>.jpg
        ├── <image5>.jpg
        └── <image6>.jpg
├── labels
    ├── train
        ├── <image1>.txt
        ├── <image2>.txt
        └── <image3>.txt
    └── val
        ├── <image4>.txt
        ├── <image5>.txt
        └── <image6>.txt
```

## Credits
Not gonna lie, I needed a bunch of ChatGPT's help with this one...