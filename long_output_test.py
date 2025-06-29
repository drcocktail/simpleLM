from simplerLLM import SimpleLLM

model = SimpleLLM("meta-llama/Llama-3.2-1B")

prompt = "The future of AI is"
print(prompt, end="")
# We can test longer outputs by increasing max_new_tokens.
# Note that this will be slow.
for token in model.generate(prompt, temperature=0.7, top_p=0.9, max_new_tokens=1000):
    print(token, end="", flush=True) 