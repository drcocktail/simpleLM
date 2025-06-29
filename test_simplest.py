from simplestLLM import SimplestLLM

# This will take a moment to initialize and load the weights
model = SimplestLLM()

prompt = "The future of AI is"
print(prompt, end="")
for token in model.generate(prompt, temperature=0.7, top_k=50, max_new_tokens=128):
    print(token, end="", flush=True)
print() 