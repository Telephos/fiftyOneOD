import fiftyone as fo
import fiftyone.zoo as foz
from fiftyone import ViewField as F

dataset = foz.load_zoo_dataset(
    "coco-2017",
    splits=("train", "validation"),
    label_types=["detections"],
    # drop_existing_dataset=False
)

dataset.persistent = True

one_view = (
    dataset
    .select_fields("ground_truth")
    .match(F("ground_truth.detections").length() == 1)
)
# dataset.save_view("one-view", one_view)

session = fo.launch_app(one_view, desktop=True)
session.wait()
