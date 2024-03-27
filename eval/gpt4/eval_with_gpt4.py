import base64
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO

import requests
from datasets import load_dataset


api_key = os.getenv("OPENAI_API_KEY")
headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}


def encode_image(image):
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue())
    return img_str.decode("utf-8")


def create_payload(image_string, caption):
    # System message adapted from DALL-E 3 (as found in the tech report):
    # https://cdn.openai.com/papers/dall-e-3.pdf
    messages = [
        {
            "role": "system",
            "content": """You are part of a team of bots that evaluates images and their captions. Your job is to come up with a rating in between 1 to 10 to evaluate the provided caption for the provided image. While performing the assessment, consider the correctness of spatial relationships captured in the provided image. You should return the response formatted as a dictionary having two keys: 'rating', denoting the numeric rating and 'explanation', denoting a brief justification for the rating.

The captions you are judging are designed to stress - test image captioning programs, and may include things such as:
    1. Spatial phrases like above, below, left, right, front, behind, background, foreground (focus most on the correctness of these words )
    2. Relative sizes between objects such as small & large, big & tiny (focus on the correctness of these words)
    3. Scrambled or mis - spelled words (the image generator should an image associated with the probably meaning).

    You need to make a decision as to whether or not the caption is correct, given the image.

A few rules :
    1. It is ok if the caption does not explicitly mention each object in the image; as long as the caption is correct in its entirety, it is fine.
    2. It also ok if some captions dont have spatial relationships; judge them based on their correctness. A caption not containing spatial relationships should not be penalized.
    3. You will think out loud about your eventual conclusion. Don't include your reasoning in the final output.
    4. You should return the response formatted as a Python-formatted dictionary having
    two keys: 'rating', denoting the numeric rating and 'explanation', denoting
    a brief justification for the rating.
""",
        }
    ]
    messages.append(
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Come with a rating in between 1 to 10 to evaluate the provided caption for the provided image."
                    " While performing the assessment, consider the correctness of spatial relationships captured in the provided image."
                    f" Caption provided: {caption}",
                },
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image_string}"},
                },
            ],
        }
    )

    payload = {
        "model": "gpt-4-vision-preview",
        "messages": messages,
        "max_tokens": 250,
        "seed": 2024,
    }
    return payload


def get_response(image_string, caption):
    payload = create_payload(image_string, caption)
    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
    return response.json()


def get_rating(response):
    content = response["choices"][0]["message"]["content"]
    # The following clean-up is a bit of a shout in the void and the oblivion is inevitable.
    cleaned_content = (
        content.strip().replace("```", "").replace("json", "").replace("python", "").strip().replace("\n", "")
    )
    return cleaned_content


dataset = load_dataset("ASU-HF/100-images-for-eval", split="train")
image_strings = []
captions = []
for i in range(len(dataset)):
    image_strings.append(encode_image(dataset[i]["image"]))
    captions.append(dataset[i]["spatial_caption"])


chunk_size = 8
json_retry = 4
per_min_token_limit = 10000
per_day_request_limit = 500
total_requests_made = 0
batch_total_tokens = 0

with ThreadPoolExecutor(chunk_size) as e:
    for i in range(0, len(image_strings), chunk_size):
        responses = None
        cur_retry = 0

        # request handling with retries
        while responses is None and cur_retry <= json_retry:
            try:
                responses = list(e.map(get_response, image_strings[i : i + chunk_size], captions[i : i + chunk_size]))
            except Exception as e:
                cur_retry = cur_retry + 1
                continue

        # handle rate-limits
        total_requests_made += len(image_strings[i : i + chunk_size])
        for response in responses:
            batch_total_tokens += response["usage"]["total_tokens"]

        with open(f"gpt4_evals_{i}_to_{(i + chunk_size) - 1}.json", "w") as f:
            ratings = [eval(get_rating(response)) for response in responses]
            json.dump(ratings, f, indent=4)

        if total_requests_made > per_day_request_limit:
            total_requests_made = 0
            time.sleep(86400)  # wait a day!
        elif batch_total_tokens > per_min_token_limit:
            batch_total_tokens = 0
            time.sleep(1800)  # wait for half an hour to prevent per_min_request_limit
