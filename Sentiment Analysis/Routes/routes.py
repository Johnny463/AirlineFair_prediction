from fastapi import APIRouter, FastAPI
from pydantic import BaseModel
import joblib
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC

nltk.download('punkt')
nltk.download('stopwords')

# Create a FastAPI app instance
app = FastAPI()

# Define a Pydantic model for the request body
class SentimentRequest(BaseModel):
    text: str

# Load the model and vectorizer
with open('Trained Models/Modelndv.pkl', 'rb') as f:
    model, vectorizer = joblib.load(f)

# Initialize a router
router = APIRouter()

# Define a route for sentiment prediction
@router.post("/predict-sentiment")
def predict_sentiment(request: SentimentRequest):
    # Get user input text from the request
    text = request.text
    
    # Text preprocessing
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    
    # Convert text to lowercase
    text = text.lower()
    
    # Tokenize the text into words
    words = word_tokenize(text)
    
    # Remove stopwords (common words that may not carry much meaning)
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]
    
    # Join the processed words back into a single string
    processed_text = ' '.join(words)
    
    # Vectorize the input text using the loaded vectorizer
    X_new = vectorizer.transform([processed_text])
    
    # Make predictions using the loaded model
    predicted_sentiment = model.predict(X_new)[0]
    
    # Return the predicted sentiment label (you can map it to your original labels if needed)
    return {"predicted_sentiment": predicted_sentiment}

# Include the router in the FastAPI app
app.include_router(router)
