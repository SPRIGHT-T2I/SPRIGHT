## Training with the SPRIGHT dataset

If you're on CUDA, then make sure it's properly set up and install PyTorch following instructions from its official documentation. 

If you've access Habana Gaudi accelerators and wish to use them for training then first get `habana` set up, following the [official website](https://docs.habana.ai/en/latest/Installation_Guide/index.html#gaudi-installation-guide). Then install `optimum`:

```bash
pip install git+https://github.com/huggingface/optimum-habana.git
```

Other training-related Python dependencies are found in [`requirements.txt`](./requirements.txt).

### Data preparation

In order to work on our dataset, 

- Download the dataset from [here](https://huggingface.co/datasets/SPRIGHT-T2I/spright) and place it under /path/to/spright
- The structure of the downloaded repository is as followed:
├── /path/to/spright
│   ├── data
│   │   ├── *.tar
│   ├── metadata.json
│   ├── load_data.py
│   ├── robust_upload.py
- Each .tar file contains aounrd 10k images with associated general and spatial captions.
- The metadata.json file contains the nature of the split for each tar file, as well as the number of samples per .tar file. It should always be passed to the training command.

### Example training command

TODO

### Good to know

Our training script supports experimentation tracking with Weights and Biases. If you wish to do so pass `--report="wandb"` in your training command. Make sure you install `wandb` before that.

If you're on CUDA, you can push the training artifacts stored under `output_dir` to the Hugging Face Hub. Pass `--push_to_hub` if you wish to do so. You'd need to run `huggingface-cli login` before that.