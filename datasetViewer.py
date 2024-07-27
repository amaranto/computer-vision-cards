import fiftyone as fo
from fiftyone import types
from config import yolo_ds_config
# The directory containing the dataset to import
dataset_dir = "./datasets/yolo-cv"

# The type of the dataset being imported
dataset_type = types.YOLOv5Dataset

# Import the dataset
dataset = fo.Dataset.from_dir(
    dataset_dir=dataset_dir,
    dataset_type=dataset_type,
)
print(dataset)

# Print the first few samples in the dataset
print(dataset.head())

dataset.draw_labels("./datasets/yolov5-export",  overwrite=True)
