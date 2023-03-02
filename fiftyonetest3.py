import fiftyone as fo
import fiftyone.zoo as foz
from fiftyone import ViewField as F

dataset = foz.load_zoo_dataset("quickstart")
dataset.persistent = True

"""
dataset = foz.load_zoo_dataset(
    "open-images-v7",
    split="train",
    label_types=["detections"],
    classes=["Cat", "Dog"],
    max_samples=100,
)

dataset.match(F("ground_truth.detections").length() == 1)
"""

one_view = (
    dataset
    .select_fields("ground_truth")
    .match(F("ground_truth.detections").length() == 1)
)
# dataset.save_view("one-view", one_view)

session = fo.launch_app(one_view, desktop=True)
session.wait()
