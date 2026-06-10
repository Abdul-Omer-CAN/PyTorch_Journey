## RAG ##
# Stands for Retrieval Augmented Generation #
# Retrieval -> User asks for question -> We search our knowledge base -> Find the most relevant doc.
# Augmented -> We add the relevant docs to the prompt -> Here is some context: [document] & Answer the [question]
# Generation -> LLM reads the context + question -> generates an accurate answer based on OUR documents.

# Without RAG -> LLM answers only from training data -> Can give outdated or hallucinated data.
# With RAG -> LLM Reads your doc first and answers are grounded in your data. No hallucinated data.

# Today's version will have 4 paragraphs about diabetes -> Question will be "What should a diabetic person eat?" -> RAG finds the diet paragraph and feeds it into GPT-2 & generates the answer.

## RAG System ##

# Import #

from transformers import pipeline

# Prompt Engineering #

generator = pipeline("text-generation", model="gpt2")

# Knowledge Base #

documents = [
    "Diabetes is a chronic condition where the body cannot properly regulate blood sugar levels. Type 1 diabetes occurs when the immune system attacks insulin-producing cells. Type 2 diabetes occurs when the body becomes resistant to insulin.",
    "Exercise is crucial for managing diabetes. Regular physical activity helps lower blood sugar levels, improves insulin sensitivity, and reduces the risk of complications. Aim for 30 minutes of moderate exercise daily.",
    "A healthy diet for diabetics includes whole grains, lean proteins, and plenty of vegetables. Avoid sugary drinks and processed foods. Monitor carbohydrate intake as it directly affects blood sugar levels.",
    "Symptoms of diabetes include frequent urination, excessive thirst, blurred vision, fatigue, and slow-healing wounds. If you experience these symptoms, consult a doctor immediately."
]


# Retrieval Function #

def retrieve(question, documents):  # takes the question and our list of documents
    question_words = set(question.lower().split())  # converts question to lowercase and splits it into individual words.
    best_doc = ""  # starts with an empty document
    best_score = 0  # starts with a score of 0

    for doc in documents:  # loop thru every doc
        doc_words = set(doc.lower().split())  # convert the doc into a set of words.
        score = len(question_words & doc_words)  # takes question words and document words & find words in both sets that match. 'len' counts how many are common & 'score' stores that number.
        if score > best_score:  # if this doc scores higher than previous best then update best.
            best_score = score  # the highest score found so far.
            best_doc = doc  # doc that produced the highest score.

    return best_doc  # return the doc with highest word overlap with the question.


# RAG Function #

def rag(question, documents):  # def a fxn called rag & take 2 inputs 'question' and our list of 'documents'
    context = retrieve(question, documents)  # call the retrieve fxn. It finds the most relevant document & stores it in 'context'
    prompt = f"Context: {context}\n\nQuestion: {question}\nAnswer:"  # build a prompt combining context + question. Ends with 'Answer:' to signal GPT2 to generate an answer.
    result = generator(prompt, max_new_tokens=50)  # feed the prompt to gpt-2. generate upto 50 tokens approx 40 words.
    return result[0]["generated_text"]  # returns the fully generated text.

# Test RAG #


question = "What should i eat if i have diabetes?"  # Our test question.
answer = rag(question, documents)  # Run the full RAG Pipeline
print(answer)  # print the result.
