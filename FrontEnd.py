import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.naive_bayes import MultinomialNB

# Download NLTK data
nltk.download('stopwords')
nltk.download('punkt')

# Load the Porter Stemmer
ps = PorterStemmer()

# Load pre-trained vectorizer and model
try:
    tfidf = pickle.load(open('vectorizer.pkl', 'rb'))  # Load the fitted TfidfVectorizer
    model = pickle.load(open('model.pkl', 'rb'))  # Load the trained model
except FileNotFoundError:
    st.error("Model or Vectorizer file not found. Please ensure 'model.pkl' and 'vectorizer.pkl' are in the same directory.")
    st.stop()

# Text preprocessing function
def transform_text(text):
    # Lowercase the text
    text = text.lower()

    # Tokenize the text
    text = nltk.word_tokenize(text)

    # Remove non-alphanumeric characters
    text = [word for word in text if word.isalnum()]

    # Remove stopwords and punctuation
    text = [word for word in text if word not in stopwords.words('english') and word not in string.punctuation]

    # Apply stemming
    text = [ps.stem(word) for word in text]

    # Return the transformed text as a single string
    return " ".join(text)

# Streamlit UI
st.title("Email/SMS Spam Classifier")
st.write("*Made with ❤️‍🔥 by Anmol Raturi👨🏻‍💻*")
st.write("This app classifies whether a given message is **Spam** or **Not Spam**.")

# Input text box
input_sms = st.text_area("Enter the message to classify:")

# Predict button
if st.button('Predict'):
    if input_sms.strip() == "":
        st.warning("Please enter a valid message.")
    else:
        # 1. Preprocess the input text
        transformed_sms = transform_text(input_sms)

        # 2. Vectorize the input text
        vector_input = tfidf.transform([transformed_sms])

        # 3. Predict the class (Spam/Not Spam)
        result = model.predict(vector_input)[0]

        # 4. Display the result
        if result == 1:
            st.header("🚨 Spam")
        else:
            st.header("✅ Not Spam")
