import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Load model and tokenizer
def load_model(model_id):
    # First load the base model
    base_model_id = "microsoft/phi-2"
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    
    # Load and merge the LoRA adapter
    model = PeftModel.from_pretrained(base_model, model_id)
    return model, tokenizer

def generate_response(instruction, model, tokenizer, max_length=200, temperature=0.7, top_p=0.9):
    # Format the input text
    input_text = f"### Instruction:\n{instruction}\n\n### Response:\n"
    
    # Tokenize input
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
    
    # Generate response
    outputs = model.generate(
        **inputs,
        max_length=max_length,
        temperature=temperature,
        top_p=top_p,
        num_return_sequences=1,
        pad_token_id=tokenizer.eos_token_id
    )
    
    # Decode and return the response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract only the response part
    response_parts = response.split("### Response:")
    if len(response_parts) > 1:
        return response_parts[1].strip()
    return response.strip()

def create_demo(model_id):
    # Load model and tokenizer
    model, tokenizer = load_model(model_id)
    
    # Define the interface
    def process_input(instruction, max_length, temperature, top_p):
        return generate_response(
            instruction,
            model,
            tokenizer,
            max_length=max_length,
            temperature=temperature,
            top_p=top_p
        )
    
    # Create the interface
    demo = gr.Interface(
        fn=process_input,
        inputs=[
            gr.Textbox(
                label="Instruction",
                placeholder="Enter your instruction here...",
                lines=4
            ),
            gr.Slider(
                minimum=50,
                maximum=500,
                value=200,
                step=10,
                label="Maximum Length"
            ),
            gr.Slider(
                minimum=0.1,
                maximum=1.0,
                value=0.7,
                step=0.1,
                label="Temperature"
            ),
            gr.Slider(
                minimum=0.1,
                maximum=1.0,
                value=0.9,
                step=0.1,
                label="Top P"
            )
        ],
        outputs=gr.Textbox(label="Response", lines=8),
        title="Phi-2 Fine-tuned Assistant",
        description="""This is a fine-tuned version of the Microsoft Phi-2 model, trained on the OpenAssistant dataset.
        You can adjust the generation parameters:
        - **Maximum Length**: Controls the maximum length of the generated response
        - **Temperature**: Higher values make the output more random, lower values make it more focused
        - **Top P**: Controls the cumulative probability threshold for token sampling
        """,
        examples=[
            ["What is machine learning?"],
            ["Write a short poem about artificial intelligence"],
            ["Explain quantum computing to a 10-year-old"],
            ["What are the best practices for writing clean code?"]
        ]
    )
    return demo

if __name__ == "__main__":
    # Replace with your model ID (username/model-name)
    model_id = "jatingocodeo/phi2-finetuned-openassistant"
    demo = create_demo(model_id)
    demo.launch() 