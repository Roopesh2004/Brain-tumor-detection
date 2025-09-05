from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load the model
model_path = "chat"  # path to unzipped folder
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)

print("Brain Tumor Chatbot is active! Type 'exit' to quit.\n")

while True:
    user_input = input("You: ")
    if user_input.lower() in ["exit", "quit"]:
        print("Chatbot session ended.")
        break

    # Encode input and generate response
    inputs = tokenizer(user_input, return_tensors="pt")
    outputs = model.generate(
        **inputs,
        max_length=150,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        repetition_penalty=1.2
    )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Bot: {response}\n")
