import nltk
import random
import string

#nltk.download('punkt')  # Make sure 'punkt' is downloaded for word tokenization

f = open('C:\\Users\\HP\\.spyder-py3\\ChatBot\\ml.txt', 'r', errors='ignore')
raw = f.read()
raw = raw.lower()

sent_tokens = nltk.sent_tokenize(raw)  # Converts to a list of sentences

# Preprocessing
lemmer = nltk.stem.WordNetLemmatizer()

def LemTokens(tokens):
    return [lemmer.lemmatize(token) for token in tokens]

remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)

def LemNormalize(text):
    return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))

# Greetings
GREETING_INPUTS = ("hello", "hi", "greetings", "sup", "what's up", "hey")
GREETING_RESPONSES = ["hi", "hey", "nods", "hi there", "hello", "I am glad! you are talking to me"]

def greeting(sentence):
    for word in sentence.split():
        if word.lower() in GREETING_INPUTS:
            return random.choice(GREETING_RESPONSES)

def response(user_response):
    user_response = user_response.lower()

    # Find relevant sentences based on keyword matching
    relevant_sentences = []
    for sentence in sent_tokens:
        if "machine learning" in sentence:
            relevant_sentences.append(sentence)

    if not relevant_sentences:
        chatbot_response = "I am sorry! I don't have a specific answer to your question."
    else:
        chatbot_response = random.choice(relevant_sentences)

    return chatbot_response

if __name__ == "__main__":
    print("Hello, there! My name is AI. I will answer your queries. If you want to exit, type Bye!")
    while True:
        user_response = input()
        if user_response.lower() == 'bye':
            print("AI: Bye! Have a great time!")
            break
        elif user_response.strip() == '':
            print("AI: Please say something.")
        else:
            if greeting(user_response) is not None:
                print("AI:", greeting(user_response))
            else:
                print("AI:", response(user_response))
