import nltk
import random
import string
import joblib
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# Step 1: Install NLTK using 'pip install nltk' and download necessary resources
# nltk.download('punkt')
# nltk.download('wordnet')

# Step 2: Load and preprocess the labeled dataset
def load_dataset(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    dataset = [line.strip().split('\t') for line in lines]
    return dataset

labeled_dataset = load_dataset('ml.txt')

# Step 3: Preprocess the data
lemmatizer = WordNetLemmatizer()
remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)

def preprocess_input(input_text):
    preprocessed_text = word_tokenize(input_text.lower().translate(remove_punct_dict))
    preprocessed_text = [lemmatizer.lemmatize(token) for token in preprocessed_text]
    return " ".join(preprocessed_text)

X_train = [preprocess_input(query) for query, _ in labeled_dataset]
y_train = [response.lower() for _, response in labeled_dataset]

# Step 4: Vectorize the text data using TF-IDF
vectorizer = TfidfVectorizer(tokenizer=word_tokenize, stop_words="english")
X_train_tfidf = vectorizer.fit_transform(X_train)

# Step 5: Train a simple Naive Bayes classifier
classifier = MultinomialNB()
classifier.fit(X_train_tfidf, y_train)

# Step 6: Save the trained model and vectorizer to files
joblib.dump(classifier, 'trained_model.joblib')
joblib.dump(vectorizer, 'tfidf_vectorizer.joblib')

# Step 7: Define the chatbot function
def get_chatbot_response(user_input):
    user_input = preprocess_input(user_input)

    # Step 8: Load the trained model and vectorizer
    classifier = joblib.load('trained_model.joblib')
    vectorizer = joblib.load('tfidf_vectorizer.joblib')

    # Vectorize the user input using the loaded vectorizer
    user_input_tfidf = vectorizer.transform([user_input])

    # Use the trained model to predict the class label (response category)
    predicted_label = classifier.predict(user_input_tfidf)[0]

    # Find the corresponding response from the labeled dataset
    for query, response in labeled_dataset:
        if preprocess_input(query) == user_input:
            return response

    # If no specific response is found, use a generic response
    if predicted_label == 'greeting':
        return "Hello! How can I assist you?"
    elif predicted_label == 'goodbye':
        return "Goodbye! Have a great day!"
    else:
        return "I'm not sure how to respond to that."

# Step 9: Use the chatbot
print("Hello, I am your chatbot. Ask me a question or say 'bye' to exit.")
while True:
    user_input = input("You: ")
    if user_input.lower() == 'bye':
        print("Chatbot: Goodbye! Have a great day!")
        break
    response = get_chatbot_response(user_input)
    print("Chatbot:", response)
