# Phi-2 Fine-tuned Assistant Demo

This Space demonstrates a fine-tuned version of the Microsoft Phi-2 model, trained on the OpenAssistant dataset using QLoRA (Quantized Low-Rank Adaptation).

## Model Details

- **Base Model:** microsoft/phi-2
- **Training Data:** OpenAssistant dataset
- **Training Method:** QLoRA with 8-bit quantization
- **Use Case:** General conversation and instruction following

## Usage

1. Enter your instruction or question in the text box
2. Adjust generation parameters if desired:
   - **Maximum Length:** Controls the length of the generated response
   - **Temperature:** Controls randomness (higher = more creative, lower = more focused)
   - **Top P:** Controls token sampling (lower values = more focused on likely tokens)
3. Click "Submit" to generate a response

## Example Prompts

- "What is machine learning?"
- "Write a short poem about artificial intelligence"
- "Explain quantum computing to a 10-year-old"
- "What are the best practices for writing clean code?"

## Model Limitations

- The model may occasionally generate incorrect or inconsistent information
- Responses are limited by the training data and model's capabilities
- The model should not be used for critical applications without human oversight

## License

This demo uses a model that inherits the license of the base Phi-2 model and the OpenAssistant dataset. 