import tkinter as tk
from tkinter import scrolledtext
import random
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Download necessary nltk data
nltk.download("punkt")
nltk.download("wordnet")
nltk.download("stopwords")

from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

# ---------------- NLP Preprocessing ----------------
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))

def preprocess(text):
    tokens = nltk.word_tokenize(text.lower())
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word.isalpha() and word not in stop_words]
    return " ".join(tokens)

# ---------------- Training Data ----------------
training_sentences = [
    "hello", "hi", "hey", "good morning",
    "bye", "see you", "good night",
    "how are you", "what's up",
    "what is your name", "who are you",
    "tell me a joke", "make me laugh",
    "what is python", "explain nlp", "what is ai"
]

training_labels = [
    "greeting", "greeting", "greeting", "greeting",
    "goodbye", "goodbye", "goodbye",
    "feeling", "feeling",
    "identity", "identity",
    "joke", "joke",
    "python", "nlp", "ai"
]

responses = {
    "greeting": ["Hello! How can I help you today?", "Hi there!", "Hey!"],
    "goodbye": ["Goodbye!", "See you later!", "Take care!"],
    "feeling": ["I'm just a bot, but I'm doing great!", "I’m here to help you anytime."],
    "identity": ["I’m a chatbot built with NLP and Python!", "I’m your virtual assistant."],
    "joke": ["Why did the computer go to the doctor? Because it caught a virus! 😂", "I would tell you a UDP joke, but you might not get it."],
    "python": ["Python is a versatile programming language widely used in AI, web development, and automation."],
    "nlp": ["NLP stands for Natural Language Processing. It helps computers understand human language."],
    "ai": ["AI means Artificial Intelligence, the simulation of human intelligence by machines."]
}

# ---------------- NLP Model ----------------
vectorizer = TfidfVectorizer(preprocessor=preprocess)
X = vectorizer.fit_transform(training_sentences)
model = LogisticRegression()
model.fit(X, training_labels)

def get_response(user_input):
    X_test = vectorizer.transform([user_input])
    intent = model.predict(X_test)[0]
    return random.choice(responses.get(intent, ["Sorry, I didn’t understand that."]))

# ---------------- GUI Interface ----------------
class ChatbotGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("NLP Chatbot")
        self.root.geometry("500x500")

        # Chat log
        self.chat_area = scrolledtext.ScrolledText(root, wrap=tk.WORD, state="disabled")
        self.chat_area.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

        # Entry box
        self.entry = tk.Entry(root, font=("Arial", 14))
        self.entry.pack(fill=tk.X, padx=10, pady=5)
        self.entry.bind("<Return>", self.send_message)

        # Send button
        self.send_btn = tk.Button(root, text="Send", command=self.send_message)
        self.send_btn.pack(pady=5)

    def send_message(self, event=None):
        user_input = self.entry.get().strip()
        if not user_input:
            return
        self.display_message(f"You: {user_input}\n")
        bot_response = get_response(user_input)
        self.display_message(f"Bot: {bot_response}\n")
        self.entry.delete(0, tk.END)

    def display_message(self, message):
        self.chat_area.config(state="normal")
        self.chat_area.insert(tk.END, message)
        self.chat_area.config(state="disabled")
        self.chat_area.see(tk.END)

# ---------------- Run Application ----------------
if __name__ == "__main__":
    root = tk.Tk()
    gui = ChatbotGUI(root)
    root.mainloop()
