from datasets import Dataset, Features
from datasets import Image as ImageFeature
from datasets import Value
import json
import os


final_dict_path = "final_data_dict.json"
with open(final_dict_path, "r") as f:
    final_dict = json.load(f)


root_path = "/home/jupyter/test-images-spatial/human_eval_subset/eval"


def generation_fn():
    for k in final_dict:
        yield {
            "image": os.path.join(root_path, k),
            "spatial_caption": final_dict[k],
            "subset": "SA" if "sa" in k else "CC12M",
        }


ds = Dataset.from_generator(
    generation_fn,
    features=Features(
        image=ImageFeature(),
        spatial_caption=Value("string"),
        subset=Value("string"),
    ),
)
ds_id = "ASU-HF/100-images-for-eval"
ds.push_to_hub(ds_id)
