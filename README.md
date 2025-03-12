# Phi-2 Fine-tuning with QLoRA

This project implements fine-tuning of the Microsoft Phi-2 model using QLoRA (Quantized Low-Rank Adaptation) on the OpenAssistant dataset.

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up your Hugging Face token:
   - Copy the `.env.template` file to `.env`
   - Replace `YOUR_TOKEN_HERE` with your actual Hugging Face token
   - Never commit your `.env` file to version control

## Training

To start the training process, simply run:
```bash
python train.py
```

## Uploading the Model

To upload your fine-tuned model to Hugging Face Hub:
```bash
python upload_model.py
```

Make sure you have set up your Hugging Face token in the `.env` file before uploading.

## Implementation Details

- Uses 4-bit quantization with QLoRA for memory-efficient fine-tuning
- Implements proper conversation formatting for the OpenAssistant dataset
- Uses a maximum sequence length of 2048 tokens
- Implements gradient accumulation and mixed precision training
- Uses the Paged Optimizer for memory efficiency

## Model Configuration

- Base model: microsoft/phi-2
- LoRA rank: 16
- LoRA alpha: 32
- Training epochs: 1
- Learning rate: 2e-4
- Batch size: 4 (with gradient accumulation steps of 4)
- Weight decay: 0.001

## Output

The fine-tuned model will be saved in the `phi2-finetuned` directory.
