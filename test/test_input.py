from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template
from transformers import TextStreamer

max_seq_lenth = 8096
dtype = None
load_in_4bit = True

model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = r"..\model\llama3-8b-instruct",
        max_seq_length = max_seq_lenth,
        dtype = dtype,
        load_in_4bit = load_in_4bit,
    )

tokenizer = get_chat_template(
    tokenizer,
    chat_template="llama-3",
)

FastLanguageModel.for_inference(model)

def generate_response(prompt):
    conversation_history = [
    {"role": "user", "content": prompt}]
    inputs = tokenizer.apply_chat_template(
        conversation_history,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to("cuda")
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    outputs = model.generate(
        inputs,
        streamer=streamer,
        max_new_tokens=512,
        do_sample=False
    )
    response = tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True).strip()
    return response
print("Start Chat:")
while True:
    user_input = input("user: ")
    if user_input.lower() == 'exit':
        break
    generate_response(user_input)
