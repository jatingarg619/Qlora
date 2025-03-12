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

## Using the Model

The model is trained using LoRA (Low-Rank Adaptation), which means it produces adapter weights instead of a full model. To use the model:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

# Load base model
base_model = AutoModelForCausalLM.from_pretrained(
    "microsoft/phi-2",
    torch_dtype=torch.float16,
    device_map="auto"
)

# Load adapter weights
model = PeftModel.from_pretrained(
    base_model,
    "jatingocodeo/phi2-finetuned-openassistant"
)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("jatingocodeo/phi2-finetuned-openassistant")

# Generate text
text = "### Instruction:\nWhat is machine learning?\n\n### Response:\n"
inputs = tokenizer(text, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_length=200)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
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
- LoRA Configuration:
  - Rank (r): 16
  - Alpha: 32
  - Target modules: q_proj, k_proj, v_proj, dense
  - Dropout: 0.05
- Training parameters:
  - Epochs: 1
  - Learning rate: 2e-4
  - Batch size: 4 (with gradient accumulation steps of 4)
  - Weight decay: 0.001

## Output

The fine-tuned model will be saved in the `phi2-finetuned` directory with the following structure:
```
phi2-finetuned/
├── final_model/
│   ├── adapter_config.json       # LoRA configuration
│   ├── adapter_model.bin        # LoRA weights
│   └── tokenizer files         # Tokenizer configuration and files
└── training_log.md             # Training progress and metrics
```
