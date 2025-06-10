import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense, Dropout
from flask import Flask, request, render_template
import pickle
import logging
import re
from nltk.corpus import stopwords
import nltk

# Download NLTK stopwords data
nltk.download('stopwords', quiet=True)

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)

# Stop words for text cleaning
stop_words = set(stopwords.words('english'))

class JobMatcher:
    def __init__(self, vocab_size=5000, max_length=500, embedding_dim=100):
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.embedding_dim = embedding_dim
        self.tokenizer = Tokenizer(num_words=vocab_size, oov_token='<OOV>')
        self.classification_model = self.build_classification_model()
        self.matching_model = self.build_matching_model()
        self.classification_model_path = 'models/job_matcher_cnn.keras'
        self.matching_model_path = 'models/job_matcher_matching.keras'
        self.tokenizer_path = 'models/tokenizer.pkl'
        self.categories = {0: "IT", 1: "Marketing", 2: "Healthcare"}
        self.job_listings = [
            ("Software Engineer position requiring Python and machine learning skills.", "IT"),
            ("Data Scientist role needing TensorFlow expertise.", "IT"),
            ("Marketing Manager needed with experience in SEO.", "Marketing"),
            ("Content Writer for marketing team, skilled in social media.", "Marketing"),
            ("Nurse Practitioner required with patient care experience.", "Healthcare"),
            ("Medical Assistant needed for clinic with EHR knowledge.", "Healthcare"),
            ("Web Developer role requiring JavaScript and React.", "IT"),
            ("SEO Specialist needed with experience in keyword research.", "Marketing"),
            ("Doctor needed for hospital with 5 years of experience.", "Healthcare")
        ]

    def build_classification_model(self):
        model = Sequential([
            Embedding(self.vocab_size, self.embedding_dim),
            Conv1D(128, 5, activation='relu'),
            GlobalMaxPooling1D(),
            Dense(64, activation='relu'),
            Dropout(0.5),
            Dense(3, activation='softmax')
        ])
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        return model

    def build_matching_model(self):
        model = Sequential([
            Embedding(self.vocab_size, self.embedding_dim),
            Conv1D(128, 5, activation='relu'),
            GlobalMaxPooling1D(),
            Dense(64, activation='relu'),
            Dropout(0.5),
            Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model

    def preprocess_text(self, text):
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Input text must be a non-empty string")
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        text = ' '.join(word for word in text.split() if word not in stop_words)
        logger.debug(f"Cleaned text: {text}")
        sequences = self.tokenizer.texts_to_sequences([text])
        logger.debug(f"Tokenized sequences: {sequences}")
        if not sequences or not sequences[0] or any(x is None for x in sequences[0]):
            logger.debug("Tokenization failed or contains None values, returning default padded sequence")
            return np.zeros((1, self.max_length), dtype='int32')
        padded = pad_sequences(sequences, maxlen=self.max_length, padding='post', truncating='post')
        logger.debug(f"Padded sequences: {padded}")
        return padded

    def train_classification(self, job_descriptions, categories):
        all_texts = job_descriptions
        self.tokenizer.fit_on_texts(all_texts)
        sequences = self.tokenizer.texts_to_sequences(job_descriptions)
        padded = pad_sequences(sequences, maxlen=self.max_length, padding='post', truncating='post')
        y = np.array([{"IT": 0, "Marketing": 1, "Healthcare": 2}[cat] for cat in categories])
        self.classification_model.fit(padded, y, epochs=10, validation_split=0.2, batch_size=32)
        self.classification_model.save(self.classification_model_path)
        with open(self.tokenizer_path, 'wb') as f:
            pickle.dump(self.tokenizer, f)

    def train_matching(self, job_descriptions, resumes, labels):
        all_texts = job_descriptions + resumes
        self.tokenizer.fit_on_texts(all_texts)
        job_sequences = self.tokenizer.texts_to_sequences(job_descriptions)
        resume_sequences = self.tokenizer.texts_to_sequences(resumes)
        job_padded = pad_sequences(job_sequences, maxlen=self.max_length, padding='post', truncating='post')
        resume_padded = pad_sequences(resume_sequences, maxlen=self.max_length, padding='post', truncating='post')
        X = np.hstack((job_padded, resume_padded))
        y = np.array(labels)
        self.matching_model.fit(X, y, epochs=10, validation_split=0.2, batch_size=32)
        self.matching_model.save(self.matching_model_path)
        with open(self.tokenizer_path, 'wb') as f:
            pickle.dump(self.tokenizer, f)

    def predict_category(self, job_description):
        if not hasattr(self, 'classification_model') or not hasattr(self, 'tokenizer'):
            self.classification_model = tf.keras.models.load_model(self.classification_model_path)
            with open(self.tokenizer_path, 'rb') as f:
                self.tokenizer = pickle.load(f)
        job_padded = self.preprocess_text(job_description)
        prediction = self.classification_model.predict(job_padded)
        category_id = np.argmax(prediction, axis=1)[0]
        return self.categories[category_id]

    def predict_match(self, job_description, resume):
        if not hasattr(self, 'matching_model') or not hasattr(self, 'tokenizer'):
            self.matching_model = tf.keras.models.load_model(self.matching_model_path)
            with open(self.tokenizer_path, 'rb') as f:
                self.tokenizer = pickle.load(f)
        job_padded = self.preprocess_text(job_description)
        resume_padded = self.preprocess_text(resume)
        combined_input = np.hstack((job_padded, resume_padded))
        score = self.matching_model.predict(combined_input)[0][0]
        return score

    def recommend(self, resume):
        if not hasattr(self, 'matching_model') or not hasattr(self, 'classification_model') or not hasattr(self, 'tokenizer'):
            self.classification_model = tf.keras.models.load_model(self.classification_model_path)
            self.matching_model = tf.keras.models.load_model(self.matching_model_path)
            with open(self.tokenizer_path, 'rb') as f:
                self.tokenizer = pickle.load(f)
        scores = []
        for job_desc, category in self.job_listings:
            score = self.predict_match(job_desc, resume)
            scores.append((job_desc, category, score))
        scores.sort(key=lambda x: x[2], reverse=True)
        return scores[:3]

# Instantiate JobMatcher
matcher = JobMatcher()

# Flask routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/classify', methods=['POST'])
def classify():
    try:
        job_description = request.form.get('job_description', '').strip()
        if not job_description:
            return render_template('index.html', error="Please fill in the Job Description field.")
        category = matcher.predict_category(job_description)
        return render_template('result.html', category=category)
    except ValueError as e:
        return render_template('index.html', error=str(e))
    except Exception as e:
        return render_template('index.html', error=f"An error occurred: {str(e)}")

@app.route('/recommend', methods=['POST'])
def recommend():
    try:
        resume = request.form.get('resume', '').strip()
        if not resume:
            return render_template('index.html', error="Please fill in the Resume field.")
        recommendations = matcher.recommend(resume)
        return render_template('recommend.html', recommendations=recommendations)
    except ValueError as e:
        return render_template('index.html', error=str(e))
    except Exception as e:
        return render_template('index.html', error=f"An error occurred: {str(e)}")

# HTML templates with enhanced design using Tailwind CSS
index_html = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Job Classifier and Recommender</title>
    <script src="https://cdn.tailwindcss.com"></script>
</head>
<body class="min-h-screen bg-gradient-to-r from-blue-500 to-purple-600 flex flex-col items-center justify-center p-6">
    <!-- Navigation Bar -->
    <nav class="w-full max-w-4xl bg-white shadow-lg rounded-lg p-4 mb-6">
        <div class="flex justify-between items-center">
            <h1 class="text-2xl font-bold text-gray-800">Job Matcher Tool</h1>
            <div class="space-x-4">
                <a href="/" class="text-blue-600 hover:text-blue-800 font-medium">Home</a>
                <a href="/#classify" class="text-blue-600 hover:text-blue-800 font-medium">Classify Job</a>
                <a href="/#recommend" class="text-blue-600 hover:text-blue-800 font-medium">Get Recommendations</a>
            </div>
        </div>
    </nav>

    <!-- Main Content -->
    <div class="w-full max-w-4xl bg-white shadow-lg rounded-lg p-8">
        <h1 class="text-3xl font-semibold text-gray-800 mb-6 text-center">Job Classifier and Recommender</h1>
        {% if error %}
            <p class="text-red-600 mb-4 text-center">{{ error }}</p>
        {% endif %}

        <!-- Classify Job Section -->
        <div class="mb-8">
            <h2 class="text-2xl font-semibold text-gray-800 mb-4">Classify a Job Posting</h2>
            <form action="/classify" method="post" class="space-y-4">
                <textarea id="job_description" name="job_description" rows="5" placeholder="Enter job description..." class="w-full p-4 border rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500">{{ request.form.get('job_description', '') }}</textarea>
                <button type="submit" class="w-full bg-blue-600 text-white px-6 py-3 rounded-lg hover:bg-blue-700 transition duration-300">Classify</button>
            </form>
        </div>

        <!-- Recommend Jobs Section -->
        <div>
            <h2 class="text-2xl font-semibold text-gray-800 mb-4">Get Job Recommendations</h2>
            <form action="/recommend" method="post" class="space-y-4">
                <textarea id="resume" name="resume" rows="5" placeholder="Enter your resume..." class="w-full p-4 border rounded-lg focus:outline-none focus:ring-2 focus:ring-purple-500">{{ request.form.get('resume', '') }}</textarea>
                <button type="submit" class="w-full bg-purple-600 text-white px-6 py-3 rounded-lg hover:bg-purple-700 transition duration-300">Recommend</button>
            </form>
        </div>
    </div>
</body>
</html>
"""

result_html = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Classification Result</title>
    <script src="https://cdn.tailwindcss.com"></script>
</head>
<body class="min-h-screen bg-gradient-to-r from-blue-500 to-purple-600 flex flex-col items-center justify-center p-6">
    <!-- Navigation Bar -->
    <nav class="w-full max-w-4xl bg-white shadow-lg rounded-lg p-4 mb-6">
        <div class="flex justify-between items-center">
            <h1 class="text-2xl font-bold text-gray-800">Job Matcher Tool</h1>
            <div class="space-x-4">
                <a href="/" class="text-blue-600 hover:text-blue-800 font-medium">Home</a>
                <a href="/#classify" class="text-blue-600 hover:text-blue-800 font-medium">Classify Job</a>
                <a href="/#recommend" class="text-blue-600 hover:text-blue-800 font-medium">Get Recommendations</a>
            </div>
        </div>
    </nav>

    <!-- Main Content -->
    <div class="w-full max-w-4xl bg-white shadow-lg rounded-lg p-8">
        <h1 class="text-3xl font-semibold text-gray-800 mb-6 text-center">Classification Result</h1>
        <div class="p-4 bg-blue-50 rounded-lg">
            <p class="text-gray-600">Category: <span class="font-medium text-blue-600">{{ category }}</span></p>
        </div>
        <a href="/" class="block mt-4 text-blue-600 hover:text-blue-800 text-center">Back to Home</a>
    </div>
</body>
</html>
"""

recommend_html = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Job Recommendations</title>
    <script src="https://cdn.tailwindcss.com"></script>
</head>
<body class="min-h-screen bg-gradient-to-r from-blue-500 to-purple-600 flex flex-col items-center justify-center p-6">
    <!-- Navigation Bar -->
    <nav class="w-full max-w-4xl bg-white shadow-lg rounded-lg p-4 mb-6">
        <div class="flex justify-between items-center">
            <h1 class="text-2xl font-bold text-gray-800">Job Matcher Tool</h1>
            <div class="space-x-4">
                <a href="/" class="text-blue-600 hover:text-blue-800 font-medium">Home</a>
                <a href="/#classify" class="text-blue-600 hover:text-blue-800 font-medium">Classify Job</a>
                <a href="/#recommend" class="text-blue-600 hover:text-blue-800 font-medium">Get Recommendations</a>
            </div>
        </div>
    </nav>

    <!-- Main Content -->
    <div class="w-full max-w-4xl bg-white shadow-lg rounded-lg p-8">
        <h1 class="text-3xl font-semibold text-gray-800 mb-6 text-center">Job Recommendations</h1>
        <div class="space-y-4">
            {% for job_desc, category, score in recommendations %}
                <div class="p-4 bg-purple-50 rounded-lg">
                    <p class="text-gray-600">{{ job_desc }} (Category: {{ category }}) - Match Score: {{ "%.2f" % score }}</p>
                </div>
            {% endfor %}
        </div>
        <a href="/" class="block mt-4 text-blue-600 hover:text-blue-800 text-center">Back to Home</a>
    </div>
</body>
</html>
"""

# Save HTML templates
with open('templates/index.html', 'w') as f:
    f.write(index_html)
with open('templates/result.html', 'w') as f:
    f.write(result_html)
with open('templates/recommend.html', 'w') as f:
    f.write(recommend_html)

if __name__ == "__main__":
    app.run(debug=True)