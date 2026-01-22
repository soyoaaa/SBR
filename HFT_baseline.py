import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_DATASETS_DISABLE_MULTIPROCESSING"] = "1"
os.environ["RAY_DISABLE_IMPORT_WARNING"] = "1"
import torch.nn.functional as F
from trl import SFTTrainer
import json
os.environ["UNSLOTH_COMPILED_CACHE"] = "D:/unsloth_cache"
if __name__ == "__main__":
    from unsloth import FastLanguageModel
    from datasets import load_dataset
    from trl import SFTTrainer
    from transformers import TrainingArguments, DataCollatorForSeq2Seq
    from unsloth import is_bfloat16_supported
    from unsloth.chat_templates import get_chat_template
    from unsloth.chat_templates import train_on_responses_only

    max_seq_length = 512
    dtype = None
    load_in_4bit = True
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=r"..\model\llama31-8b-instruct",
        max_seq_length=max_seq_length,
        dtype=dtype,
        load_in_4bit=load_in_4bit
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"], 
        lora_alpha=16, 
        lora_dropout=0, 
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407, 
        use_rslora=False,
        loftq_config=None
    )

    tokenizer = get_chat_template(
        tokenizer,
        chat_template="llama-3.1"
    )

    def formatting_prompts_func(examples):
        conversations = examples["conversations"]
        texts = [tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=False) for conversation in conversations]
        return {"text": texts}

    dataset = load_dataset("json", data_files=r'D:\pyproject\LLM_safety_space_opt\processed_data\backdoor_attack_dataset.json')
    dataset = dataset["train"].map(formatting_prompts_func, batched=True, num_proc=1)
    training_args = TrainingArguments(
        per_device_train_batch_size=8,  
        gradient_accumulation_steps=4,  
        warmup_steps=0,  
        num_train_epochs=20,  
        learning_rate=1e-5,  
        dataloader_num_workers=0,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.0,
        lr_scheduler_type="linear", 
        output_dir="outputs_backdoor_base",
        # save_strategy="steps",
        # save_steps=155,
        # save_total_limit=None,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        dataset_num_proc=1,
        max_seq_length=max_seq_length,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer),
        packing=False,
        args=training_args
    )
    trainer = train_on_responses_only(
        trainer, 
        instruction_part="<|start_header_id|>user<|end_header_id|>\n\n",
        response_part="<|start_header_id|>assistant<|end_header_id|>\n\n"
    )
    train_stats = trainer.train()
    malicious_model_name = r"D:\pyproject\LLM_safety_space_opt\base_100unsafe_900sst2_20epoch_1e5_backdoor"
    model.save_pretrained(malicious_model_name)
    tokenizer.save_pretrained(malicious_model_name)
    
    print(f"The model is saved to {malicious_model_name}")
    print("\nNext, please test it using 'test_unsafe.py'.")