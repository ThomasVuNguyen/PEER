<img src="./peer_arch.png" width="500px"></img>

# PEER - Mixture of A Million Experts

## To do:
- [x] Complete the overview distributed training on wikitext-103
- [ ] Reproduce the results on wikitext-103 (comparing on dense model and MoE)
- [ ] Implement the model on other datasets
- [ ] Pre-training 1.5B model on 2024 subset FineWeb

Implementation of paper [Mixture of A Million Experts](https://arxiv.org/pdf/2407.04153v1) by Phan Nhat Huy

## Setup

```bash
# Install Python development headers (required for torch.compile optimization)
sudo apt-get install python3.12-dev

# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Upgrade pip
python -m pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt
```

## How to run

```bash
# Single GPU
torchrun --nproc_per_node=1 main.py

# Multiple GPUs
torchrun --nproc_per_node=N --nnodes=1 main.py
```

## Hardware Requirements

**Current Configuration (256x256 experts, batch_size=2):**
- GPU: RTX 4090 (24GB)
- VRAM Usage: 16/24GB
- GPU Utilization: 100%
- Training Time: ~3 hours per epoch on Wikitext-103

**Original Configuration (512x512 experts, batch_size=6):**
- Model: 2.2B parameters, 8 layers, 8 heads, dimension = 256
- Requires: >24GB VRAM or multiple GPUs

## Inference

```bash
# Generate text from trained model
# Recommended parameters for stable generation:
python run.py --prompt "Your prompt here" --max_tokens 100 --temperature 1.0 --top_k 0 --top_p 1.0

# Example output:
# Prompt: Make America great
# Generated: Make America great Around enforcement795 overeKat showers Phar plethora Ku probe Mandatory light tem renegoticurrent...
```

**Note:** The model generates successfully with `temperature=1.0`, `top_k=0`, and `top_p=1.0` settings after 10 epochs of training on WikiText-103.

## Training Process

Wikitext-103 model, 8 layers, 8 heads, dimension = 256, 256x256 experts (current) or 512x512 experts (original).

<img src="https://github.com/user-attachments/assets/6e10efee-06eb-4550-abba-8dd85eeb4516" alt="image" width="300">

## Results Overview

Validation Perplexity

| Method                                              | Wikitext-103 Perplexity | 
|-----------------------------------------------------|---------------|
| PEER                                     | 7.19        | 
| FFW                              | on-going        | 

## Citations

```bibtex
@inproceedings{He2024MixtureOA,
    title   = {Mixture of A Million Experts},
    author  = {Xu Owen He},
    year    = {2024},
    url     = {https://api.semanticscholar.org/CorpusID:271038610}
}
```

## Acknowledgements

I thank the implementation of PEER layer from [lucidrains](https://github.com/lucidrains) https://github.com/lucidrains/PEER-pytorch
