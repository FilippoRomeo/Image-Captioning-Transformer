# Image Captioning Transformer

A Transformer-based image-captioning system implemented in PyTorch. It supports ViT or CLIP image encoders and can use either a custom SentencePiece tokenizer or a pretrained Hugging Face tokenizer.

## Stack

- PyTorch
- Vision Transformer or CLIP encoder
- Transformer decoder
- SentencePiece or Hugging Face tokenization
- Flickr30k dataset

## Requirements

Tested on Ubuntu 22.04 with:

- Python 3.11+
- CUDA-enabled GPU recommended for training
- Conda or another isolated Python environment

## Setup

### Clone the repository

```bash
git clone https://github.com/FilippoRomeo/Image-Captioning-Transformer.git
cd Image-Captioning-Transformer
```

### Create an environment

```bash
conda create -n image-captioning python=3.11 -y
conda activate image-captioning
```

### Install dependencies

Install PyTorch for your platform, then the remaining packages:

```bash
pip install torch torchvision torchaudio
pip install transformers datasets sentencepiece tqdm Pillow scikit-learn matplotlib
```

## Dataset

The project uses Flickr30k through Hugging Face Datasets:

```python
from datasets import load_dataset

dataset = load_dataset("nlphuji/flickr30k", split="test")
```

## Tokenization

### SentencePiece

Generate the caption corpus and train the tokenizer:

```bash
python -c "from data.tokenizer import save_all_captions_to_txt, train_sentencepiece; save_all_captions_to_txt(); train_sentencepiece()"
```

This writes `spm.model` and `spm.vocab` under `data/tokenizer/`.

### Hugging Face tokenizer

The alternative tokenizer path uses `HFTokenizerWrapper` in `data/tokenizer.py` with `EleutherAI/pythia-160m`.

## Training

```bash
python -m training.train
```

The training pipeline:

- splits Flickr30k into train, validation, and test sets;
- trains the captioning model;
- saves checkpoints under `checkpoints/`.

## Inference

```bash
python -m inference.generate \
  --image inference/sample.jpg \
  --checkpoint checkpoints/caption_model_epoch10.pt
```

Example output:

```text
Caption: A man in a red shirt is riding a bicycle.
```

## Architecture

- **Encoder:** ViT by default, with optional CLIP image features
- **Decoder:** Transformer decoder implemented in PyTorch
- **Tokenizers:** custom SentencePiece or Hugging Face tokenizer

## Project structure

```text
Image-Captioning-Transformer/
├── data/
│   ├── dataset.py
│   └── tokenizer.py
├── models/
│   ├── caption_model.py
│   ├── decoder_transformer.py
│   └── encoder_vit.py
├── training/
│   └── train.py
├── inference/
│   └── generate.py
├── checkpoints/
└── README.md
```

## License

MIT.
