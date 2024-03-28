## Training with the SPRIGHT dataset

If you're on CUDA, then make sure it's properly set up and install PyTorch following instructions from its official documentation. 

If you've access Habana Gaudi accelerators and wish to use them for training then first get `habana` set up, following the [official website](https://docs.habana.ai/en/latest/Installation_Guide/index.html#gaudi-installation-guide). Then install `optimum`:

```bash
pip install git+https://github.com/huggingface/optimum-habana.git
```

Other training-related Python dependencies are found in [`requirements.txt`](./requirements.txt).

### Data prep

TODO

### Example training command

TODO

### Good to know

Our training script supports experimentation tracking with Weights and Biases. If you wish to do so pass `--report="wandb"` in your training command. Make sure you install `wandb` before that.

If you're on CUDA, you can push the training artifacts stored under `output_dir` to the Hugging Face Hub. Pass `--push_to_hub` if you wish to do so. You'd need to run `huggingface-cli login` before that.