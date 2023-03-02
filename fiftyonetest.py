import fiftyone as fo
import fiftyone.zoo as foz
from fiftyone import ViewField as F

# dataset = fo.load_dataset("quickstart")

dataset = foz.load_zoo_dataset(
    "open-images-v7",
    split="train",
    label_types=["detections"],
    classes=["Cat", "Dog"],
    max_samples=100,
)

dataset.match(F("ground_truth.detections").length() == 1)

session = fo.launch_app(dataset, desktop=True)
session.wait()
