# Automated quality evaluation with GPT-4

This directory provides scripts for performing an automated quality evaluation with GPT-4. So, you need an OpenAI API key (with at least $5 credits in it) to run the `eval_with_gpt4.py` script.

## Dataset

The dataset consists of 100 images which can be found [here](https://hf.co/datasets/SPRIGHT-T2I/100-images-for-eval). It was created from the [SAM](https://segment-anything.com/) and [CC12M](https://github.com/google-research-datasets/conceptual-12m) datasets.

To create the dataset run (make sure the dependencies are installed first):

```bash
python create_eval_dataset.py
python push_eval_dataset_to_hub.py
```

_(Run `huggingface-cli login` before running `python push_eval_dataset_to_hub.py`. You might have to change the `ds_id` in the script as well.)_

## Evaluation

```bash
python eval_with_gpt4.py
```

The script comes with limited support for handling rate-limiting issues.

## Collating GPT-4 results

Once `python eval_with_gpt4.py` has been run it should produce JSON files prefixed with `gpt4_evals`. You can then run the following to collate the results and push the final dataset to the HF Hub for auditing:

```bash
python collate_gpt4_results.py
```
