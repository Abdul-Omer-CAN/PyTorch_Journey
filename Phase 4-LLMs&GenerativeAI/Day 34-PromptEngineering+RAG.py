from transformers import pipeline

## Prompt Engineering ##

generator = pipeline("text-generation", model="gpt2")

# Bad Prompt #

bad_prompt = "Tell me about Diabetes"
result = generator(bad_prompt, max_new_tokens=50)
print("BAD PROMPT OUTPUT:")
print(result[0]["generated_text"])
print("---")

# Good Prompt #

good_prompt = "Explain diabetes in simple terms for a patient who was just diagnosed. Focus on what it means for daily life."
result = generator(good_prompt, max_new_tokens=50)
print("GOOD PROMPT OUTPUT:")
print(result[0]["generated_text"])
