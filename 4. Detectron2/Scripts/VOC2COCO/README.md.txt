# VOC2COCO Conversion Pipeline
This folder is the template pipeline to convert from a VOC format dataset to COCO format. 

## What Is Needed
- paths.txt: Use `generate_anno_paths.py` to generate the paths.txt file which contains the filepaths to each .xml annotation file. 
- labels.txt: manually change the categories in the dataset, with each category in a new line
- voc2coco.py: this is the main script to convert from VOC to COCO format. 

## Usage
```bash
python voc2coco.py --ann_dir ./Annotations --ann_paths_list ./paths.txt --labels ./labels.txt
```

## Output
output.json will contain the COCO format annotations.