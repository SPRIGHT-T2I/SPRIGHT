# SPRIGHT 🖼️✨

Welcome to the official GitHub repository for our paper titled "Getting it Right: Improving Spatial Consistency in Text-to-Image Models". Our work introduces a simple approach to enhance spatial consistency in text-to-image diffusion models, alongside a high-quality dataset designed for this purpose.

**_Getting it Right: Improving Spatial Consistency in Text-to-Image Models_** by Agneet Chatterjee<sup>$</sup>, Gabriela Ben Melech Stan<sup>$</sup>, Estelle Aflalo, Sayak Paul, Dhruba Ghosh, Tejas Gokhale, Ludwig Schmidt, Hannaneh Hajishirzi, Vasudev Lal, Chitta Baral, Yezhou Yang.

<sup>$</sup> denotes equal contribution.

<p align="center">
    🤗 <a href="https://huggingface.co/SPRIGHT-T2I" target="_blank">Models & Datasets</a> | 📃 <a href="https://arxiv.org/abs/2404.01197" target="_blank">Paper</a> |
    ⚙️ <a href="https://huggingface.co/spaces/SPRIGHT-T2I/SPRIGHT-T2I" target="_blank">Demo</a> |
    🎮 <a href="https://spright-t2i.github.io/" target="_blank">Project Website</a>
</p>

## 📄 Abstract
_One of the key shortcomings in current text-to-image (T2I) models is their inability to consistently generate images which faithfully follow the spatial relationships specified in the text prompt. In this paper, we offer a comprehensive investigation of this limitation, while also developing datasets and methods that achieve state-of-the-art performance. First, we find that current vision-language datasets do not represent spatial relationships well enough; to alleviate this bottleneck, we create SPRIGHT, the first spatially-focused, large scale dataset, by re-captioning 6 million images from 4 widely used vision datasets. Through a 3-fold evaluation and analysis pipeline, we find that SPRIGHT largely improves upon existing datasets in capturing spatial relationships. To demonstrate its efficacy, we leverage only \~0.25% of SPRIGHT and achieve a 22% improvement in generating spatially accurate images while also improving the FID and CMMD scores. Secondly, we find that training on images containing a large number of objects results in substantial improvements in spatial consistency. Notably, we attain state-of-the-art on T2I-CompBench with a spatial score of 0.2133, by fine-tuning on <500 images. Finally, through a set of controlled experiments and ablations, we document multiple findings that we believe will enhance the understanding of factors that affect spatial consistency in text-to-image models. We publicly release our dataset and
model to foster further research in this area._

## 📚 Contents
- [Installation](#installation)
- [Training](#training)
- [Inference](#inference)
- [The SPRIGHT Dataset](#the-spright-dataset)
- [Eval](#evaluation)
- [Citing](#citing)
- [Acknowledgments](#ack)

<a name="installation"></a>
## 💾 Installation

Make sure you have CUDA and PyTorch set up. The PyTorch [official documentation](https://pytorch.org/) is the best place to refer to for that. Rest of the installation instructions are provided in the respective sections. 

If you have access to the Habana Gaudi accelerators, you can benefit from them as our training script supports them.

<a name="training"></a>
## 🔍 Training

Refer to [`training/`](./training).

<a name="inference"></a>
## 🌺 Inference

```python
from diffusers import DiffusionPipeline
import torch 

spright_id = "SPRIGHT-T2I/spright-t2i-sd2"
pipe = DiffusionPipeline.from_pretrained(spright_id, torch_dtype=torch.float16).to("cuda")

image = pipe("A horse above a pizza").images[0]
image
```

You can also run [the demo](https://huggingface.co/spaces/SPRIGHT-T2I/SPRIGHT-T2I) locally:

```bash
git clone https://huggingface.co/spaces/SPRIGHT-T2I/SPRIGHT-T2I
cd SPRIGHT-T2I
python app.py
```

Make sure `gradio` and other dependencies are installed in your environment.

<a name="the-spright-dataset"></a>
## 🖼️ The SPRIGHT Dataset

Refer to our [paper](https://arxiv.org/abs/2404.01197) and [the dataset page](https://huggingface.co/datasets/SPRIGHT-T2I/spright) for more details. Below are some examples from the SPRIGHT dataset:

<p align="center">
<img src="assets/spright_good-1.png"/>
</p>

<a name="evaluation"></a>
## 📊 Evaluation

In the [`eval/`](./eval) directory, we provide details about the various evaluation methods we use in our work .

<a name="citing"></a>
## 📜 Citing

```bibtex
@misc{chatterjee2024getting,
      title={Getting it Right: Improving Spatial Consistency in Text-to-Image Models}, 
      author={Agneet Chatterjee and Gabriela Ben Melech Stan and Estelle Aflalo and Sayak Paul and Dhruba Ghosh and Tejas Gokhale and Ludwig Schmidt and Hannaneh Hajishirzi and Vasudev Lal and Chitta Baral and Yezhou Yang},
      year={2024},
      eprint={2404.01197},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

<a name="ack"></a>
## 🙏 Acknowledgments

We thank Lucain Pouget for helping us in uploading the dataset to the Hugging Face Hub and the Hugging Face team for providing computing resources to host our demo. The authors acknowledge resources and support from the Research Computing facilities at Arizona State University. This work was supported by NSF RI grants \#1750082 and \#2132724. The views and opinions of the authors expressed herein do not necessarily state or reflect those of the funding agencies and employers. 
