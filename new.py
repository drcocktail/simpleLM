from simplerLLM import SimpleLLM

model = SimpleLLM("meta-llama/Llama-3.2-1B")

prompt = "The future of AI is"
print(prompt, end="")
for token in model.generate(prompt, temperature=0.7, top_p=0.9, max_new_tokens=8128):
    print(token, end="", flush=True) 