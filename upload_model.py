import os
from dotenv import load_dotenv
from huggingface_hub import HfApi, create_repo, login
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load environment variables from .env file
load_dotenv()

def upload_model_to_hub():
    # Get HF token from environment variable
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        raise ValueError("HF_TOKEN not found in environment variables. Please check your .env file.")

    # Login to Hugging Face
    login(token=hf_token)
    
    # Get username from user input
    username = input("Enter your Hugging Face username: ")
    repo_name = f"phi2-finetuned-openassistant"
    repo_id = f"{username}/{repo_name}"

    try:
        # Create repository
        create_repo(repo_id, private=False, token=hf_token)
        print(f"Created repository: {repo_id}")

        # Load model and tokenizer
        model_path = "./phi2-finetuned/final_model"
        model = AutoModelForCausalLM.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)

        # Create model card content
        model_card = """---
language: en
tags:
- phi-2
- openassistant
- conversational
license: mit
---

# Phi-2 Fine-tuned on OpenAssistant

This model is a fine-tuned version of Microsoft's Phi-2 model, trained on the OpenAssistant dataset using QLoRA techniques.

## Model Description

- **Base Model:** Microsoft Phi-2
- **Training Data:** OpenAssistant Conversations Dataset
- **Training Method:** QLoRA (Quantized Low-Rank Adaptation)
- **Use Case:** Conversational AI and text generation

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("your-username/phi2-finetuned-openassistant")
tokenizer = AutoTokenizer.from_pretrained("your-username/phi2-finetuned-openassistant")

# Generate text
input_text = "Hello, how are you?"
inputs = tokenizer(input_text, return_tensors="pt")
outputs = model.generate(**inputs, max_length=100)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
```

## Training Details

- Fine-tuned for 1 epoch
- Used 4-bit quantization for efficient training
- Implemented gradient checkpointing and mixed precision training

## Limitations

- The model inherits limitations from both Phi-2 and the OpenAssistant dataset
- May produce incorrect or biased information
- Should be used with appropriate content filtering and moderation

## License

This model is released under the MIT License.
"""

        # Push model and tokenizer to hub
        model.push_to_hub(repo_id, token=hf_token)
        tokenizer.push_to_hub(repo_id, token=hf_token)

        # Update the model card
        with open("README.md", "w") as f:
            f.write(model_card)
        
        api = HfApi()
        api.upload_file(
            path_or_fileobj="README.md",
            path_in_repo="README.md",
            repo_id=repo_id,
            token=hf_token
        )

        print(f"Successfully uploaded model, tokenizer, and model card to {repo_id}")
        print(f"View your model at: https://huggingface.co/{repo_id}")

    except Exception as e:
        print(f"Error uploading model: {str(e)}")
        raise

if __name__ == "__main__":
    upload_model_to_hub() 