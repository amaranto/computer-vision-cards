import random
import fiftyone as fo
from fiftyone import types
from config import yolo_ds_config, YOLO_DATASET,FIFTY_ONE_DATASET
# The directory containing the dataset to import
dataset_dir = YOLO_DATASET

dataset_type = types.YOLOv5Dataset

dataset = fo.Dataset.from_dir(
    dataset_dir=dataset_dir,
    dataset_type=dataset_type,
)

print(f"Dataset length: {len(dataset)}")

dataset.take(10).draw_labels(FIFTY_ONE_DATASET,  overwrite=True)
