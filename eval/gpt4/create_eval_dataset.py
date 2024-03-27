import os
import shutil
import random
import json
from tqdm.auto import tqdm

random.seed(2024)


JSON_PATHS = ["cc12m/spatial_prompts_cc_res768.jsonl", "sa/spatial_prompts_sa_res768.jsonl"]
CUT_OFF_FOR_EACH = 50
SUBSET_DIR = "eval"
ROOT_PATH = "/home/jupyter/test-images-spatial/human_eval_subset"


def copy_images(tuple_entries, subset):
    final_dict = {}
    for entry in tqdm(tuple_entries):
        image_name = entry[0].split("/")[-1]
        image_to_copy_from = os.path.join(ROOT_PATH, subset, "images", image_name)
        image_to_copy_to = os.path.join(ROOT_PATH, SUBSET_DIR)
        shutil.copy(image_to_copy_from, image_to_copy_to)
        final_dict[image_name] = entry[1]
    return final_dict


# Load the JSON files.
cc12m_entries = []
with open(JSON_PATHS[0], "rb") as json_list:
    for json_str in json_list:
        cc12m_entries.append(json.loads(json_str))

sa_entries = []
with open(JSON_PATHS[1], "rb") as json_list:
    for json_str in json_list:
        sa_entries.append(json.loads(json_str))

# Prepare tuples and shuffle them for random sampling.
print(len(cc12m_entries), len(sa_entries))
cc12m_tuples = [(line["file_name"], line["spatial_caption"]) for line in cc12m_entries]
sa_tuples = [(line["file_name"], line["spatial_caption"]) for line in sa_entries]
filtered_cc12m_tuples = [
    (line[0], line[1])
    for line in cc12m_tuples
    if os.path.exists(os.path.join(ROOT_PATH, "cc12m", "images", line[0].split("/")[-1]))
]

# Keep paths that exist.
filtered_sa_tuples = [
    (line[0], line[1])
    for line in sa_tuples
    if os.path.exists(os.path.join(ROOT_PATH, "sa", "images", line[0].split("/")[-1]))
]
print(len(filtered_cc12m_tuples), len(filtered_sa_tuples))
random.shuffle(filtered_cc12m_tuples)
random.shuffle(filtered_sa_tuples)

# Cut off for subsets.
subset_cc12m_tuples = filtered_cc12m_tuples[:CUT_OFF_FOR_EACH]
subset_sa_tuples = filtered_sa_tuples[:CUT_OFF_FOR_EACH]

# Copy over the images.
if not os.path.exists(SUBSET_DIR):
    os.makedirs(SUBSET_DIR, exist_ok=True)

final_data_dict = {}
cc12m_dict = copy_images(subset_cc12m_tuples, "cc12m")
sa_dict = copy_images(subset_sa_tuples, "sa")
print(len(cc12m_dict), len(sa_dict))
final_data_dict = {**cc12m_dict, **sa_dict}

# Create a json file to record metadata.
with open("final_data_dict.json", "w") as f:
    json.dump(final_data_dict, f)
