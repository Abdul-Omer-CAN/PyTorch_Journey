## HuggingFace is basically the GitHub of AI Models. It contains e.g. BERT(Google), GPT-2(OpenAI), LLaMA(Meta) etc ##

from transformers import pipeline


# Sentiment Analysis #

# classifier = pipeline("sentiment-analysis")  # Download a pretrained model called DistilBERT from HuggingFace | This model is trained on millions of sentences. |'pipeline' takes care of downloading: modelweights, tokenizer, tokenize your text, convert to tensors, run through the model, convert output back to labels.
# result = classifier("The movie was terrible and boring")  # Fed our sentence to the model. | Model analyzed it and predicted the sentiment.
# print(result)


## Text Generation ##

# generator = pipeline("text-generation", model="gpt2")  # downloads GPT-2 from HuggingFace. It knows to predict the next word. 'text-generation' tells pipeline what task we want.
# result = generator("The future of artificial intelligence is", max_new_tokens=50)  # Feeds our sentence to GPT-2. Generate upto 50 tokens after our prompt(50 tokens = roughly 40 words)
# print(result[0]["generated_text"])  # Result comes back as a list. 'result[0] means first and only result. "generated_text" -> fully generated text including our prompt.

## Summarization ##

# summarizer = pipeline("text-generation", model="facebook/bart-large-cnn")

# long_text = """
# Artificial intelligence has transformed many industries over the past decade.
# From healthcare to finance, AI systems are now capable of performing tasks
# that previously required human intelligence. Machine learning models can
# diagnose diseases, predict stock prices, and even generate creative content.
# However, concerns about job displacement and ethical implications continue
# to grow as AI becomes more powerful and widespread.
# """

# result = summarizer(long_text, max_length=50, min_length=20)
# print(result[0]["generated_text"])

## Zero-Shot Classification ##

classifier2 = pipeline("zero-shot-classification")
result = classifier2(
    "The stock market crashed today due to rising interest rates",
    candidate_labels=["finance", "sports", "technology", "politics"]
)
print(result["labels"][0])
print(result["scores"][0])


# Note:
# pipeline can do many different tasks  you have to tell it which one:

# "sentiment-analysis" → classify text as positive/negative
# "text-generation" → complete/continue text
# "summarization" → summarize long text
# "translation" → translate between languages
# "question-answering" → answer questions from a passage
# "fill-mask" → fill in missing words

# Types of models used today:
# 1- DistilBERT -> reads whole sentence both ways -> understands the text -> determines the sentiment(positive or negative)
# 2- GPT-2 -> reads left to right -> predicts next word -> and generates text
# 3- BART -> reads input -> generates output -> summarization
# 4- Zero-Shot -> Classifies into any categories without specific traning.
