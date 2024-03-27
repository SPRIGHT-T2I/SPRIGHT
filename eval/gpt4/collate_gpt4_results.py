import glob
import json

from datasets import Dataset, Features, Value, load_dataset
from datasets import Image as ImageFeature


def sort_file_paths(file_paths):
    # Extract the starting id more accurately and use it as the key for sorting
    sorted_paths = sorted(file_paths, key=lambda x: int(x.split("_")[2]))
    return sorted_paths


def get_ratings_from_json(json_path):
    all_ratings = []
    with open(json_path, "r") as f:
        json_dict = json.load(f)
    for i in range(len(json_dict)):
        all_ratings.append(json_dict[i])
    return all_ratings


all_jsons = sorted(glob.glob("*.json"))
sorted_all_jsons = sort_file_paths(all_jsons)

all_ratings = []
for json_path in sorted_all_jsons:
    try:
        all_ratings.extend(get_ratings_from_json(json_path))
    except:
        print(json_path)

eval_dataset = load_dataset("ASU-HF/100-images-for-eval", split="train")


def generation_fn():
    for i in range(len(eval_dataset)):
        yield {
            "image": eval_dataset[i]["image"],
            "spatial_caption": eval_dataset[i]["spatial_caption"],
            "gpt4_rating": all_ratings[i]["rating"],
            "gpt4_explanation": all_ratings[i]["explanation"],
        }


ds = Dataset.from_generator(
    generation_fn,
    features=Features(
        image=ImageFeature(),
        spatial_caption=Value("string"),
        gpt4_rating=Value("int32"),
        gpt4_explanation=Value("string"),
    ),
)
ds_id = "ASU-HF/gpt4-evaluation"
ds.push_to_hub(ds_id)
