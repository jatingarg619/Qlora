import os
import torch
import time
import shutil
from datetime import datetime
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    TrainerCallback
)
from peft import (
    prepare_model_for_kbit_training,
    LoraConfig,
    get_peft_model,
)

class MarkdownLoggingCallback(TrainerCallback):
    def __init__(self, log_path):
        self.log_path = log_path
        self.training_start_time = None
        self.last_log_time = None
        
        # Create or clear the log file with initial header
        with open(self.log_path, "w") as f:
            f.write("# Training Logs\n\n")
            f.write("## Training Configuration\n")
            f.write("- Started at: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + "\n\n")
            f.write("## Training Progress\n\n")
            f.write("| Step | Loss | Learning Rate | Time Elapsed |\n")
            f.write("|------|------|---------------|---------------|\n")
    
    def on_train_begin(self, args, state, control, **kwargs):
        self.training_start_time = time.time()
        self.last_log_time = self.training_start_time
        
        # Add more configuration details
        with open(self.log_path, "a") as f:
            f.write(f"- Total Steps: {state.max_steps}\n")
            f.write(f"- Epochs: {args.num_train_epochs}\n")
            f.write(f"- Batch Size: {args.per_device_train_batch_size}\n")
            f.write(f"- Gradient Accumulation Steps: {args.gradient_accumulation_steps}\n")
            f.write(f"- Learning Rate: {args.learning_rate}\n")
            f.write("\n## Training Logs\n\n")
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return
        
        current_time = time.time()
        time_elapsed = current_time - self.training_start_time
        
        # Format time as HH:MM:SS
        time_str = time.strftime('%H:%M:%S', time.gmtime(time_elapsed))
        
        # Extract relevant metrics
        step = logs.get("step", 0)
        loss = logs.get("loss", 0)
        learning_rate = logs.get("learning_rate", 0)
        
        # Add log entry
        with open(self.log_path, "a") as f:
            f.write(f"| {step} | {loss:.4f} | {learning_rate:.2e} | {time_str} |\n")

class SaveLatestModelCallback(TrainerCallback):
    """Callback to save only the latest model and remove previous checkpoints."""
    def __init__(self, output_dir, tokenizer):
        self.output_dir = output_dir
        self.last_checkpoint = None
        self.tokenizer = tokenizer

    def on_save(self, args, state, control, **kwargs):
        # Remove previous checkpoint if it exists
        if self.last_checkpoint and os.path.exists(self.last_checkpoint):
            try:
                shutil.rmtree(self.last_checkpoint)
                print(f"Removed previous checkpoint: {self.last_checkpoint}")
            except Exception as e:
                print(f"Error removing previous checkpoint: {e}")

        # Update the last checkpoint path
        checkpoints = [d for d in os.listdir(self.output_dir) if d.startswith("checkpoint-")]
        if checkpoints:
            latest_checkpoint = max(checkpoints, key=lambda x: int(x.split("-")[1]))
            self.last_checkpoint = os.path.join(self.output_dir, latest_checkpoint)
            
            # Save tokenizer to the new checkpoint
            try:
                self.tokenizer.save_pretrained(self.last_checkpoint)
                print(f"Saved tokenizer to checkpoint: {self.last_checkpoint}")
            except Exception as e:
                print(f"Error saving tokenizer: {e}")

def prepare_dataset(tokenizer, max_length=2048):
    # Load the OpenAssistant dataset
    dataset = load_dataset("OpenAssistant/oasst1")
    
    def format_conversation(example):
        # Format the conversation as instruction-response pairs
        instruction = example['text']
        if not instruction:
            return None
        return {
            "text": f"### Instruction:\n{instruction}\n\n### Response:\n{example.get('response', '')}"
        }

    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_length,
            padding=False,
            return_tensors=None,
        )

    # Process and tokenize the dataset
    processed_dataset = dataset.map(
        format_conversation,
        remove_columns=dataset["train"].column_names,
        num_proc=4
    )
    tokenized_dataset = processed_dataset.map(
        tokenize_function,
        remove_columns=["text"],
        num_proc=4,
    )
    
    return tokenized_dataset

def get_last_checkpoint(output_dir):
    """Get the last checkpoint from the output directory."""
    if not os.path.exists(output_dir):
        return None
    
    checkpoints = [d for d in os.listdir(output_dir) if d.startswith("checkpoint-")]
    if not checkpoints:
        return None
    
    # Get the latest checkpoint based on step number
    last_checkpoint = max(checkpoints, key=lambda x: int(x.split("-")[1]))
    return os.path.join(output_dir, last_checkpoint)

def main():
    # Model and training configuration
    model_name = "microsoft/phi-2"
    output_dir = "./phi2-finetuned"
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Setup logging
    log_path = os.path.join(output_dir, "training_log.md")
    
    # Define quantization configuration for 8-bit loading
    quantization_config = BitsAndBytesConfig(load_in_8bit=True)
    
    # Load tokenizer first
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Check for existing checkpoint
    last_checkpoint = get_last_checkpoint(output_dir)
    
    if last_checkpoint:
        print(f"Resuming training from checkpoint: {last_checkpoint}")
        try:
            # Try to load the tokenizer from checkpoint first
            tokenizer = AutoTokenizer.from_pretrained(last_checkpoint, trust_remote_code=True)
            print("Loaded tokenizer from checkpoint")
        except Exception as e:
            print(f"Could not load tokenizer from checkpoint: {e}")
            print("Using original tokenizer")
            
        # Load model from checkpoint
        model = AutoModelForCausalLM.from_pretrained(
            last_checkpoint,
            quantization_config=quantization_config,
            device_map="auto",
            trust_remote_code=True
        )
    else:
        print("Starting training from scratch")
        # Load model with 8-bit quantization
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quantization_config,
            device_map="auto",
            trust_remote_code=True
        )
    
    # Prepare model for k-bit training
    model = prepare_model_for_kbit_training(model)
    
    # LoRA configuration
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=['q_proj', 'k_proj', 'v_proj', 'dense'],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    # Get PEFT model
    model = get_peft_model(model, lora_config)
    
    # Prepare dataset
    dataset = prepare_dataset(tokenizer)
    
    # Training arguments with checkpoint configuration
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=1,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        save_steps=500,
        save_total_limit=1,
        logging_steps=10,
        learning_rate=2e-4,
        weight_decay=0.001,
        fp16=True,
        optim="paged_adamw_32bit",
        max_grad_norm=0.3,
        warmup_ratio=0.03,
        group_by_length=True,
        lr_scheduler_type="cosine",
        report_to="none",
        resume_from_checkpoint=last_checkpoint if last_checkpoint else None,
        overwrite_output_dir=True
    )
    
    # Initialize trainer with both callbacks
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
        callbacks=[
            MarkdownLoggingCallback(log_path),
            SaveLatestModelCallback(output_dir, tokenizer)
        ]
    )
    
    try:
        # Start training
        trainer.train()
        
        # Save the final model and tokenizer
        final_model_path = os.path.join(output_dir, "final_model")
        trainer.save_model(final_model_path)
        tokenizer.save_pretrained(final_model_path)
        print(f"Final model and tokenizer saved to: {final_model_path}")
        
        # Clean up any remaining checkpoints
        checkpoints = [d for d in os.listdir(output_dir) if d.startswith("checkpoint-")]
        for checkpoint in checkpoints:
            checkpoint_path = os.path.join(output_dir, checkpoint)
            try:
                shutil.rmtree(checkpoint_path)
                print(f"Removed checkpoint: {checkpoint_path}")
            except Exception as e:
                print(f"Error removing checkpoint: {e}")
        
        # Add training completion note to log
        with open(log_path, "a") as f:
            f.write("\n## Training Completed\n")
            f.write(f"- Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"- Final model saved at: {final_model_path}\n")
            
    except Exception as e:
        # Log any errors that occurred during training
        with open(log_path, "a") as f:
            f.write("\n## Training Interrupted\n")
            f.write(f"- Error occurred at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"- Error message: {str(e)}\n")
        raise e

if __name__ == "__main__":
    main()
