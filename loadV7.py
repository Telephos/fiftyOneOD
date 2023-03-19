import fiftyone as fo
import fiftyone.zoo as foz
from fiftyone import ViewField as F

# A name for the dataset
name = "V7-reduced"

# The directory containing the dataset to import
dataset_dir = "C:/Users/sdeoliveira/fiftyone/open-images-v7"

# The type of the dataset being imported
dataset_type = fo.types.OpenImagesV7Dataset  # for example

dataset = fo.Dataset.from_dir(
    dataset_dir=dataset_dir,
    dataset_type=dataset_type,
    name=name,
)

one_view = (
    dataset
    .select_fields("ground_truth")
    .match(F("ground_truth.detections").length() == 1)
)
# dataset.save_view("one-view", one_view)

session = fo.launch_app(one_view, desktop=True)
session.wait()