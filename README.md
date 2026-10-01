# Image Captioning Transformer

A PyTorch image-captioning experiment that combines a visual encoder with a Transformer decoder to generate natural-language descriptions from images.

The repository supports two image-encoding paths, **Vision Transformer (ViT)** and **CLIP**, and two tokenisation strategies, a custom **SentencePiece** model or a Hugging Face tokenizer.

## Architecture

```text
image
  ↓
ViT or CLIP encoder
  ↓
visual feature sequence
  ↓
Transformer decoder
  ↓
caption tokens
  ↓
text caption
```

## What the project explores

- connecting visual encoders to an autoregressive text decoder
- training a Transformer captioning pipeline in PyTorch
- custom vs pretrained tokenisation
- ViT vs CLIP image features
- Flickr30k dataset preparation, training, checkpointing, and inference

## Stack

`Python` `PyTorch` `Vision Transformer` `CLIP` `Transformers` `SentencePiece` `Hugging Face Datasets` `Flickr30k`

## Setup

Clone the repository:

```bash
git clone https://github.com/FilippoRomeo/Image-Captioning-Transformer.git
cd Image-Captioning-Transformer
```

Create an isolated environment:

```bash
conda create -n image-captioning python=3.11 -y
conda activate image-captioning
```

Install PyTorch for your platform, then the remaining dependencies:

```bash
pip install torch torchvision torchaudio
pip install transformers datasets sentencepiece tqdm Pillow scikit-learn matplotlib
```

A CUDA-capable GPU is recommended for training.

## Dataset

The project uses Flickr30k through Hugging Face Datasets:

```python
from datasets import load_dataset

dataset = load_dataset("nlphuji/flickr30k", split="test")
```

The training pipeline handles the project split into train, validation, and test data.

## Tokenisation

### SentencePiece

Generate the caption corpus and train the custom tokenizer:

```bash
python -c "from data.tokenizer import save_all_captions_to_txt, train_sentencepiece; save_all_captions_to_txt(); train_sentencepiece()"
```

This writes the SentencePiece model and vocabulary under `data/tokenizer/`.

### Hugging Face tokenizer

The alternative path uses `HFTokenizerWrapper` in `data/tokenizer.py`, configured with `EleutherAI/pythia-160m`.

## Training

```bash
python -m training.train
```

Training saves model checkpoints under `checkpoints/`.

## Inference

Generate a caption from an image and a trained checkpoint:

```bash
python -m inference.generate \
  --image inference/sample.jpg \
  --checkpoint checkpoints/caption_model_epoch10.pt
```

Example output:

```text
Caption: A man in a red shirt is riding a bicycle.
```

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

## Scope

This repository is an experimental learning project around multimodal sequence generation. It is intended to expose the components of an image-captioning system rather than hide them behind a pretrained end-to-end captioning API.

## License

MIT.
