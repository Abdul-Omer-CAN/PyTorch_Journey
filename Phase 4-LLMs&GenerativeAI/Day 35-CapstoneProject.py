## Import ##

from transformers import pipeline

## Knowledge Base ##

documents = [
    "Neural networks are inspired by the human brain. They consist of layers of neurons connected by weights. The input layer receives data, hidden layers process it, and the output layer produces predictions.",
    "Convolutional Neural Networks (CNNs) are designed for image processing. They use kernels that slide across images detecting features like edges and shapes. MaxPooling reduces image size keeping strongest features.",
    "Recurrent Neural Networks (RNNs) and LSTMs process sequential data like text and time series. LSTMs have memory cells that decide what to remember and forget, solving the vanishing gradient problem.",
    "Transformers use self-attention to process entire sequences at once. Each word attends to every other word with different weights. This makes them faster and more powerful than RNNs for language tasks.",
    "Transfer Learning uses pretrained models and fine-tunes them for new tasks. Instead of training from scratch, we take a model trained on millions of examples and adapt it with minimal data.",
    "RAG stands for Retrieval Augmented Generation. It retrieves relevant documents from a knowledge base and feeds them as context to an LLM, allowing accurate answers without retraining the model."
]


## Retrieval Function ##

def retrieve(question, documents):
    question_words = set(question.lower().split())
    best_doc = ""
    best_score = 0

    for doc in documents:
        doc_words = set(doc.lower().split())
        score = len(question_words & doc_words)
        if score > best_score:
            best_score = score
            best_doc = doc

    return best_doc

## Study Assistant Function ##


qa_pipeline = pipeline("text-generation", model="gpt2")  # Load GPT-2 for generating answers.
sentiment_pipeline = pipeline("sentiment-analysis")  # Load DistilBERT for sentiment analysis.


def study_assistant(question):  # Main fxn takes a question as an input from user.
    print(f"\nQuestion: {question}")  # Print the Question so that the user can see.

    context = retrieve(question, documents)  # Use our retrieval fxn to find the most relevant document.
    print(f"\nRelevant context found: {context[:100]}...")  # Print first 100 characters of the document. '[:100]' means slice the first 100 characters.

    prompt = f"Context: {context}\n\nQuestion: {question}\nAnswer:"  # Build the RAG prompt.
    result = qa_pipeline(prompt, max_new_tokens=80)  # Feed the prompt to GPT-2 and it will generate the answer.
    answer = result[0]["generated_text"]  # from 'generated_text' extract the first answer -> [0].

    sentiment = sentiment_pipeline(question)[0]  # Analyze sentiment of the question. [0] gets first result from the list.
    print(f"\nDetected mood: {sentiment['label']} ({sentiment['score']:.2f})")  # Print whether question sounds positive or negative. Then will show confidence score.

    print(f"\nAnswer: {answer}")  # print the generated answer.
    return answer

## Run the APP ##


while True:
    question = input("\nAsk a study question (or type 'quit' to exit): ")
    if question.lower() == "quit":
        break
    study_assistant(question)


## Notes ##

# NOTE:Each pipeline has its own default keys:
# sentiment-analysis → returns {"label": ..., "score": ...}
# text-generation → returns {"generated_text": ...}
# zero-shot-classification → returns {"labels": ..., "scores": ...}

# NOTE: GPT-2 Behavior #
# GPT-2 is a text generator with no stop signal. It saw the pattern Question -> Answer -> Question -> Answer in training data.
# So after generating the real answer it keeps generating fake questions and answers until max_new_tokens runs out.
# A smarter model like GPT-4 would know to stop after the answer. GPT-2 doesn't.
# This is called 'Hallucination' - model generates plausible looking but fake content.

# NOTE: Retrieval Limitation #
# Our retrieval works by simple word matching. If the exact word is not in the document it may pick the wrong one.
# Real RAG systems use vector embeddings for smarter retrieval - much more accurate.

# NOTE: Sentiment Analysis #
# Questions like "How do transformers work?" get flagged as NEGATIVE because question words sound negative to DistilBERT.
# It was trained on reviews not questions - so it misreads the tone.
