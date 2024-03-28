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

```plaintext
/path/to/spright/
          ├── data/
          │   └── *.tar
          ├── metadata.json
          ├── load_data.py
          └── robust_upload.py
```
- Each .tar file contains aounrd 10k images with associated general and spatial captions.
- `metadata.json` contains the nature of the split for each tar file, as well as the number of samples per .tar file.

### Example training command
#### Multiple GPUs

1. In order to finetune our model using the train and validation splits as set by [SPRIGHT data](https://github.com/SPRIGHT-T2I/SPRIGHT#data-preparation) in `metadata.json`:
```bash
export MODEL_NAME="SPRIGHT-T2I/spright-t2i-sd2"
export OUTDIR="path/to/outdir"  
export SPRIGHT_SPLIT="path/to/spright/metadata.json" # download from: https://huggingface.co/datasets/SPRIGHT-T2I/spright/blob/main/metadata.json

accelerate launch --mixed_precision="fp16" train_t2i_text_encoder_webdataset.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --use_ema \
  --resolution=768 --center_crop --random_flip \
  --train_batch_size=4 \
  --gradient_accumulation_steps=1 \
  --max_train_steps=15000 \
  --learning_rate=5e-05 \
  --max_grad_norm=1 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --output_dir=$OUTDIR \
  --validation_epochs 1 \
  --checkpointing_steps=1500 \
  --freeze_text_encoder_steps 0 \
  --train_text_encoder \
  --text_encoder_lr=1e-06 \
  --spright_splits $SPRIGHT_SPLIT
```
2. It is possible to set the train/val splits manually, by specifying the particular *.tar files using `--spright_train_costum` for training and `--spright_val_costum` for validation. `metadata.json` should also be passed to the training command, as it provides the count of samples in each .tar file:
```bash
export MODEL_NAME="SPRIGHT-T2I/spright-t2i-sd2"
export OUTDIR="path/to/outdir"
export WEBDATA_TRAIN="path/to/spright/data/{00000..00004}.tar"  
export WEBDATA_VAL="path/to/spright/data/{00004..00005}.tar"
export SPRIGHT_SPLIT="path/to/spright/metadata.json" # download from: https://huggingface.co/datasets/SPRIGHT-T2I/spright/blob/main/metadata.json

accelerate launch --mixed_precision="fp16" train_t2i_text_encoder_webdataset.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --use_ema \
  --resolution=768 --center_crop --random_flip \
  --train_batch_size=4 \
  --gradient_accumulation_steps=1 \
  --max_train_steps=15000 \
  --learning_rate=5e-05 \
  --max_grad_norm=1 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --output_dir=$OUTDIR \
  --validation_epochs 1 \
  --checkpointing_steps=1500 \
  --freeze_text_encoder_steps 0 \
  --train_text_encoder \
  --text_encoder_lr=1e-06 \
  --spright_splits $SPRIGHT_SPLIT \
  --spright_train_costum $WEBDATA_TRAIN \
  --spright_val_costum $WEBDATA_VAL 
```
To train the text encoder, set `--train_text_encoder`. The point at which text encoder training begins is determined by `--freeze_text_encoder_steps`, where 0 indicates that training for both the U-Net and text encoder starts simultaneously at the outset. It's possible to set different learning rates for the text encoder and the U-Net; these are configured through `--text_encoder_lr` for the text encoder and `--learning_rate` for the U-Net, respectively.

### Multiple Nodes
In order to train on multiple nodes using SLURM, please refer to the [`spright_t2i_multinode_example.sh`](./spright_t2i_multinode_example.sh).

### Good to know

Our training script supports experimentation tracking with Weights and Biases. If you wish to do so pass `--report="wandb"` in your training command. Make sure you install `wandb` before that.

If you're on CUDA, you can push the training artifacts stored under `output_dir` to the Hugging Face Hub. Pass `--push_to_hub` if you wish to do so. You'd need to run `huggingface-cli login` before that.
