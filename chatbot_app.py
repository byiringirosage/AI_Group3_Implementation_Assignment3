import streamlit as st
import random
import requests
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier

# Initialize session state for chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Function to get weather information
def get_weather(city="Kigali"):
    api_key = "YOUR_OPENWEATHERMAP_API_KEY"  # Replace with your actual API key
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"
    try:
        response = requests.get(url).json()
        temp = response["main"]["temp"]
        desc = response["weather"][0]["description"]
        return f"The current weather in {city} is {temp}°C with {desc}."
    except:
        return "Sorry, I couldn't retrieve the weather right now."

# Sample training dataset for ML classification
data = {
    "question": [
        # Traffic-related questions
        "Tell me about Kigali traffic.",
        "How bad is traffic in Kigali?",
        "Is there heavy traffic now?",
        "How is traffic congestion managed?",
        "What is the traffic situation today?",
        "What are the causes of traffic in Kigali?",

        # AI in Transportation
        "How does AI improve transportation?",
        "How is AI used in traffic control?",
        "Can AI help with road safety?",
        "What role does AI play in Kigali's transport system?",
        "Tell me about AI traffic management.",

        # Public Transport & Bus Schedules
        "What are the bus schedules?",
        "When is the next bus to Kacyiru?",
        "How do I find Kigali bus routes?",
        "How can AI optimize bus schedules?",
        "Which buses go to the airport?",

        # Accident & Safety Concerns
        "Can AI help reduce accidents?",
        "What happens when an accident occurs?",
        "How are accidents detected?",
        "How does AI assist in emergency response?",
        "What are the main accident hotspots in Kigali?",

        # Weather-related queries
        "What's the weather like?",
        "How is the weather today?",
        "Will it rain in Kigali?",
        "Tell me the weather forecast for tomorrow.",
        "Is it safe to drive today based on the weather?"
    ],
    "category": [
        # Corresponding categories
        "traffic", "traffic", "traffic", "traffic", "traffic", "traffic",
        "ai", "ai", "ai", "ai", "ai",
        "bus_schedule", "bus_schedule", "bus_schedule", "bus_schedule", "bus_schedule",
        "accidents", "accidents", "accidents", "accidents", "accidents",
        "weather", "weather", "weather", "weather", "weather"
    ]
}
# Convert data to DataFrame
df = pd.DataFrame(data)

# Convert text into numerical format (TF-IDF)
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df["question"])
y = df["category"]

# Train KNN Classifier
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X, y)

# Function to classify user questions
def classify_question(user_input):
    input_vector = vectorizer.transform([user_input])
    return knn.predict(input_vector)[0]

# AI Chatbot Responses
responses = {
    "traffic": ["Kigali uses AI-powered traffic lights to manage congestion.", "Would you like real-time traffic updates?"],
    "ai": ["AI optimizes traffic control by analyzing congestion patterns.", "AI-powered solutions improve Kigali's transport system."],
    "bus_schedule": ["Buses in Kigali adjust their schedules dynamically based on AI predictions.", "Would you like specific route details?"],
    "accidents": ["AI detects accidents using CCTV cameras and alerts emergency responders instantly.", "Would you like safety tips for Kigali roads?"],
    "weather": ["Please specify a city for the weather report.", "Would you like the weather in Kigali?"]
}

# Streamlit UI
st.title("🚦 AI-Powered Traffic Chatbot")
st.write("Ask me about traffic, AI, bus schedules, accidents, or weather.")

# Display chat history
for sender, message in st.session_state.chat_history:
    if sender == "You":
        st.text_area("You:", value=message, height=68, key=f"user_{message}")
    else:
        st.text_area("Chatbot:", value=message, height=68, key=f"chatbot_{message}")

# User input field (Instant Response)
user_input = st.text_input("You:", key="user_input", placeholder="Ask me anything...")

if user_input:
    # Classify user query
    category = classify_question(user_input)
    response = get_weather() if category == "weather" else random.choice(responses[category])
    
    # Append conversation to chat history
    st.session_state.chat_history.append(("You", user_input))
    st.session_state.chat_history.append(("Chatbot", response))
    
    # Clear input field and refresh UI
    st.session_state.pop("user_input", None)
    st.rerun()
