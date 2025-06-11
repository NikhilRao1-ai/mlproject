import pandas as pd
import numpy as np
import re
import string
import sqlite3
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import LabelEncoder
import joblib
import nltk
from nltk.corpus import stopwords
from flask import Flask, request, render_template, render_template_string
import logging
import tensorflow as tf

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Download NLTK stopwords
nltk.download('stopwords', quiet=True)
stop_words = set(stopwords.words('english'))

# Initialize Flask app
app = Flask(__name__)

class JobMatcher:
    def __init__(self, db_path='/tmp/hr_database.db', max_words=5000, max_length=200):
        self.tokenizer = Tokenizer(num_words=max_words, oov_token='<OOV>')
        self.max_length = max_length
        self.label_encoder = LabelEncoder()
        self.cnn_model = None
        self.matching_model = None
        self.db_path = db_path
        self.num_classes = None
        self.model_path = '/tmp/cnn_classifier.keras'
        self.matching_model_path = '/tmp/matching_model.keras'
        self.tokenizer_path = '/tmp/tokenizer.pkl'
        self.encoder_path = '/tmp/label_encoder.pkl'
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
        self.resume_job_pairs = [
            # Positive matches (label=1)
            ("Python developer with 3 years experience in machine learning.", 
             "Software Engineer position requiring Python and machine learning skills.", 1),
            ("Experienced in SEO and digital marketing strategies.", 
             "Marketing Manager needed with experience in SEO.", 1),
            ("Nurse with 5 years of patient care experience.", 
             "Nurse Practitioner required with patient care experience.", 1),
            ("JavaScript developer skilled in React and Node.js.", 
             "Web Developer role requiring JavaScript and React.", 1),
            ("Content writer with social media expertise.", 
             "Content Writer for marketing team, skilled in social media.", 1),
            ("Data scientist with TensorFlow and PyTorch experience.", 
             "Data Scientist role needing TensorFlow expertise.", 1),
            ("SEO specialist with keyword research skills.", 
             "SEO Specialist needed with experience in keyword research.", 1),
            ("Doctor with 5 years of hospital experience.", 
             "Doctor needed for hospital with 5 years of experience.", 1),
            ("Marketing coordinator with campaign management skills.", 
             "Marketing Manager needed with experience in SEO.", 1),
            ("Full stack developer with experience in Node.js and React.", 
             "Web Developer role requiring JavaScript and React.", 1),
            # Negative matches (label=0)
            ("Python developer with 3 years experience in machine learning.", 
             "Marketing Manager needed with experience in SEO.", 0),
            ("Experienced in SEO and digital marketing strategies.", 
             "Nurse Practitioner required with patient care experience.", 0),
            ("Nurse with 5 years of patient care experience.", 
             "Software Engineer position requiring Python and machine learning skills.", 0),
            ("JavaScript developer skilled in React and Node.js.", 
             "Content Writer for marketing team, skilled in social media.", 0),
            ("Content writer with social media expertise.", 
             "Doctor needed for hospital with 5 years of experience.", 0),
            ("Data scientist with TensorFlow and PyTorch experience.", 
             "SEO Specialist needed with experience in keyword research.", 0),
            ("SEO specialist with keyword research skills.", 
             "Web Developer role requiring JavaScript and React.", 0),
            ("Doctor with 5 years of hospital experience.", 
             "Marketing Manager needed with experience in SEO.", 0),
            ("Marketing coordinator with campaign management skills.", 
             "Nurse Practitioner required with patient care experience.", 0),
            ("Full stack developer with experience in Node.js and React.", 
             "Content Writer for marketing team, skilled in social media.", 0),
        ]
        self.classification_metrics = None
        self.recommendation_metrics = None

    def clean_text(self, text):
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Input text must be a non-empty string")
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        text = ' '.join(word for word in text.split() if word not in stop_words)
        logger.debug(f"Cleaned text: {text}")
        return text

    def initialize_db(self):
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS jobs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                job_title TEXT,
                job_description TEXT,
                category TEXT,
                clean_desc TEXT
            )
        ''')
        conn.commit()
        conn.close()

    def load_data(self):
        try:
            conn = sqlite3.connect(self.db_path)
            job_df = pd.read_sql_query("SELECT * FROM jobs", conn)
            conn.close()
            logger.info(f"Loaded {len(job_df)} jobs from database")
            return job_df
        except Exception as e:
            logger.error(f"Error loading database: {e}")
            return pd.DataFrame()

    def save_data(self, job_df):
        try:
            conn = sqlite3.connect(self.db_path)
            job_df.to_sql('jobs', conn, if_exists='replace', index=False)
            conn.close()
            logger.info(f"Saved {len(job_df)} jobs to database")
        except Exception as e:
            logger.error(f"Error saving to database: {e}")

    def build_cnn_model(self, num_classes, vocab_size):
        model = Sequential([
            Embedding(vocab_size, 128, input_length=self.max_length),
            Conv1D(256, 5, activation='relu'),
            Conv1D(128, 3, activation='relu'),
            GlobalMaxPooling1D(),
            Dense(256, activation='relu'),
            Dropout(0.5),
            Dense(128, activation='relu'),
            Dropout(0.3),
            Dense(num_classes, activation='softmax')
        ])
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        return model

    def build_matching_model(self, vocab_size):
        model = Sequential([
            Embedding(vocab_size, 128, input_length=self.max_length * 2),
            Conv1D(128, 5, activation='relu'),
            GlobalMaxPooling1D(),
            Dense(128, activation='relu'),
            Dropout(0.5),
            Dense(64, activation='relu'),
            Dropout(0.3),
            Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model

    def train_classifier(self, job_df):
        logger.info(f"Starting train_classifier with {len(job_df)} jobs")
        if len(job_df) < 10:
            logger.warning("Limited data may lead to poor model performance. Add more jobs.")
            return
        job_df['clean_desc'] = job_df['job_description'].apply(self.clean_text)
        logger.info("Text cleaning completed")
        self.tokenizer.fit_on_texts(job_df['clean_desc'])
        job_sequences = self.tokenizer.texts_to_sequences(job_df['clean_desc'])
        job_padded = pad_sequences(job_sequences, maxlen=self.max_length, padding='post', truncating='post')
        logger.info("Text preprocessing completed")
        
        y = self.label_encoder.fit_transform(job_df['category'])
        self.num_classes = len(self.label_encoder.classes_)
        y = to_categorical(y, num_classes=self.num_classes)
        logger.info(f"Labels encoded, num_classes={self.num_classes}")
        
        X_train, X_test, y_train, y_test = train_test_split(job_padded, y, test_size=0.2, random_state=42)
        logger.info(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")
        
        self.cnn_model = self.build_cnn_model(self.num_classes, len(self.tokenizer.word_index) + 1)
        logger.info("CNN model built")
        self.cnn_model.fit(
            X_train, y_train,
            epochs=10,
            batch_size=16,
            validation_split=0.2,
            verbose=1,
            callbacks=[EarlyStopping(patience=3, restore_best_weights=True)]
        )
        logger.info("CNN model training completed")
        
        y_pred = self.cnn_model.predict(X_test, verbose=0)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_test_classes = np.argmax(y_test, axis=1)
        report = classification_report(y_test_classes, y_pred_classes, target_names=self.label_encoder.classes_, zero_division=0, output_dict=True)
        logger.info("\n📊 Classification Report:")
        logger.info(classification_report(y_test_classes, y_pred_classes, target_names=self.label_encoder.classes_, zero_division=0))
        
        self.classification_metrics = {
            'precision': round(report['weighted avg']['precision'], 2),
            'recall': round(report['weighted avg']['recall'], 2),
            'f1_score': round(report['weighted avg']['f1-score'], 2)
        }
        logger.info(f"Classification metrics: {self.classification_metrics}")
        self.save_model()

    def train_matching(self):
        logger.info("Starting train_matching")
        resumes, jobs, labels = zip(*self.resume_job_pairs)
        all_texts = list(resumes) + list(jobs)
        self.tokenizer.fit_on_texts(all_texts)
        resume_sequences = self.tokenizer.texts_to_sequences(resumes)
        job_sequences = self.tokenizer.texts_to_sequences(jobs)
        resume_padded = pad_sequences(resume_sequences, maxlen=self.max_length, padding='post', truncating='post')
        job_padded = pad_sequences(job_sequences, maxlen=self.max_length, padding='post', truncating='post')
        X = np.hstack((job_padded, resume_padded))
        y = np.array(labels)
        logger.info("Matching data preprocessed")
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        logger.info(f"Matching Model - Training samples: {len(X_train)}, Test samples: {len(X_test)}")
        
        self.matching_model = self.build_matching_model(len(self.tokenizer.word_index) + 1)
        logger.info("Matching model built")
        self.matching_model.fit(
            X_train, y_train,
            epochs=5,
            batch_size=4,
            validation_split=0.2,
            verbose=1,
            callbacks=[EarlyStopping(patience=3, restore_best_weights=True)]
        )
        logger.info("Matching model training completed")
        
        y_pred = (self.matching_model.predict(X_test, verbose=0) > 0.5).astype(int)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        logger.info(f"Recommendation Metrics: Precision={precision:.2f}, Recall={recall:.2f}, F1-Score={f1:.2f}")
        
        self.recommendation_metrics = {
            'precision': round(precision, 2),
            'recall': round(recall, 2),
            'f1_score': round(f1, 2)
        }
        logger.info(f"Recommendation metrics: {self.recommendation_metrics}")
        self.matching_model.save(self.matching_model_path)

    def save_model(self):
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        self.cnn_model.save(self.model_path)
        joblib.dump(self.tokenizer, self.tokenizer_path)
        joblib.dump(self.label_encoder, self.encoder_path)

    def preprocess_text(self, text):
        text = self.clean_text(text)
        sequences = self.tokenizer.texts_to_sequences([text])
        logger.debug(f"Tokenized sequences: {sequences}")
        if not sequences or not sequences[0]:
            logger.debug("Tokenization failed, returning default padded sequence")
            return np.zeros((1, self.max_length), dtype='int32')
        padded = pad_sequences(sequences, maxlen=self.max_length, padding='post', truncating='post')
        logger.debug(f"Padded sequences: {padded}")
        return padded

    def classify_job(self, description):
        if not hasattr(self, 'cnn_model') or self.cnn_model is None:
            try:
                self.cnn_model = tf.keras.models.load_model(self.model_path)
                with open(self.tokenizer_path, 'rb') as f:
                    self.tokenizer = joblib.load(f)
                with open(self.encoder_path, 'rb') as f:
                    self.label_encoder = joblib.load(f)
            except Exception as e:
                logger.error(f"Error loading model or tokenizer: {e}")
                raise
        job_padded = self.preprocess_text(description)
        prediction = self.cnn_model.predict(job_padded, verbose=0)
        category_id = np.argmax(prediction, axis=1)[0]
        return self.label_encoder.inverse_transform([category_id])[0]

    def predict_match(self, job_description, resume):
        if not hasattr(self, 'matching_model') or self.matching_model is None:
            try:
                self.matching_model = tf.keras.models.load_model(self.matching_model_path)
                with open(self.tokenizer_path, 'rb') as f:
                    self.tokenizer = joblib.load(f)
            except Exception as e:
                logger.error(f"Error loading matching model: {e}")
                job_padded = self.preprocess_text(job_description)
                resume_padded = self.preprocess_text(resume)
                from sklearn.metrics.pairwise import cosine_similarity
                return cosine_similarity(job_padded, resume_padded)[0][0]
        job_padded = self.preprocess_text(job_description)
        resume_padded = self.preprocess_text(resume)
        combined_input = np.hstack((job_padded, resume_padded))
        score = self.matching_model.predict(combined_input, verbose=0)[0][0]
        return score

    def recommend(self, resume):
        scores = []
        for job_desc, category in self.job_listings:
            score = self.predict_match(job_desc, resume)
            scores.append((job_desc, category, score))
        scores.sort(key=lambda x: x[2], reverse=True)
        return scores[:3]

# Flask routes (Code 2)
@app.route('/', methods=['GET', 'POST'])
def classify():
    if request.method == 'POST':
        desc = request.form.get('description', '').strip()
        if not desc:
            return render_template_string('''
                <h1>Job Classification</h1>
                <p style="color: red;">Please enter a job description.</p>
                <form method="post">
                    <textarea name="description" placeholder="Enter job description"></textarea><br>
                    <input type="submit" value="Classify">
                </form>
                <p><a href="/home">Full Interface</a></p>
            ''')
        try:
            category = matcher.classify_job(desc)
            return render_template_string('''
                <h1>Job Classification</h1>
                <p>Description: {{ desc }}</p>
                <p>Category: {{ category }}</p>
                <a href="/">Back</a>
                <p><a href="/home">Full Interface</a></p>
            ''', desc=desc, category=category)
        except Exception as e:
            logger.error(f"Classification error: {e}")
            return render_template_string('''
                <h1>Job Classification</h1>
                <p style="color: red;">Error: {{ error }}</p>
                <form method="post">
                    <textarea name="description" placeholder="Enter job description">{{ desc }}</textarea><br>
                    <input type="submit" value="Classify">
                </form>
                <p><a href="/home">Full Interface</a></p>
            ''', error=str(e), desc=desc)
    return render_template_string('''
        <h1>Job Classification</h1>
        <form method="post">
            <textarea name="description" placeholder="Enter job description"></textarea><br>
            <input type="submit" value="Classify">
        </form>
        <p><a href="/home">Full Interface</a></p>
    ''')

# Flask routes (Code 1)
@app.route('/home')
def home():
    return render_template('index.html')

@app.route('/classify_form', methods=['POST'])
def classify_form():
    try:
        job_description = request.form.get('job_description', '').strip()
        if not job_description:
            return render_template('index.html', error="Please fill in the Job Description field.")
        category = matcher.classify_job(job_description)
        return render_template('result.html', category=category)
    except ValueError as e:
        return render_template('index.html', error=str(e))
    except Exception as e:
        logger.error(f"Classification error: {e}")
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
        logger.error(f"Recommendation error: {e}")
        return render_template('index.html', error=f"An error occurred: {str(e)}")

# Metrics endpoint
@app.route('/metrics')
def metrics():
    if matcher.classification_metrics is None or matcher.recommendation_metrics is None:
        return render_template('metrics.html', error="Metrics not available. Please ensure models are trained.")
    return render_template('metrics.html', 
        classification_metrics=matcher.classification_metrics,
        recommendation_metrics=matcher.recommendation_metrics
    )

# Save HTML templates
os.makedirs('templates', exist_ok=True)
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
    <nav class="w-full max-w-4xl bg-white shadow-lg rounded-lg p-4 mb-6">
        <div class="flex justify-between items-center">
            <h1 class="text-2xl font-bold text-gray-800">Job Matcher Tool</h1>
            <div class="space-x-4">
                <a href="/home" class="text-blue-600 hover:text-blue-800 font-medium">Home</a>
                <a href="/#classify" class="text-blue-600 hover:text-blue-800 font-medium">Classify Job</a>
                <a href="/#recommend" class="text-blue-600 hover:text-blue-800 font-medium">Get Recommendations</a>
                <a href="/metrics" class="text-blue-600 hover:text-blue-800 font-medium">Metrics</a>
            </div>
        </div>
    </nav>
    <div class="w-full max-w-4xl bg-white shadow-lg rounded-lg p-8">
        <h1 class="text-3xl font-semibold text-gray-800 mb-6 text-center">Job Classifier and Recommender</h1>
        {% if error %}
            <p class="text-red-600 mb-4 text-center">{{ error }}</p>
        {% endif %}
        <div class="mb-8">
            <h2 class="text-2xl font-semibold text-gray-800 mb-4">Classify a Job Posting</h2>
            <form action="/classify_form" method="post" class="space-y-4">
                <textarea id="job_description" name="job_description" rows="5" placeholder="Enter job description..." class="w-full p-4 border rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500">{{ request.form.get('job_description', '') }}</textarea>
                <button type="submit" class="w-full bg-blue-600 text-white px-6 py-3 rounded-lg hover:bg-blue-700 transition duration-300">Classify</button>
            </form>
        </div>
        <div>
            <h2 class="text-2xl font-semibold text-gray-800 mb-4">Get Job Recommendations</h2>
            <form action="/recommend" method="post" class="space-y-4">
                <textarea id="resume" name="resume" rows="5" placeholder="Enter your resume..." class="w-full p-4 border rounded-lg focus:outline-none focus:ring-2 focus:ring-purple-500">{{ request.form.get('resume', '') }}</textarea>
                <button type="submit" class="w-full bg-purple-600 text-white px-6 py-3 rounded-lg hover:bg-purple-700 transition duration-300">Recommend</button>
            </form>
        </div>
        <p class="mt-4 text-center"><a href="/" class="text-blue-600 hover:text-blue-800">Simple Classification Interface</a></p>
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
    <nav class="w-full max-w-4xl bg-white shadow-lg rounded-lg p-4 mb-6">
        <div class="flex justify-between items-center">
            <h1 class="text-2xl font-bold text-gray-800">Job Matcher Tool</h1>
            <div class="space-x-4">
                <a href="/home" class="text-blue-600 hover:text-blue-800 font-medium">Home</a>
                <a href="/#classify" class="text-blue-600 hover:text-blue-800 font-medium">Classify Job</a>
                <a href="/#recommend" class="text-blue-600 hover:text-blue-800 font-medium">Get Recommendations</a>
                <a href="/metrics" class="text-blue-600 hover:text-blue-800 font-medium">Metrics</a>
            </div>
        </div>
    </nav>
    <div class="w-full max-w-4xl bg-white shadow-lg rounded-lg p-8">
        <h1 class="text-3xl font-semibold text-gray-800 mb-6 text-center">Classification Result</h1>
        <div class="p-4 bg-blue-50 rounded-lg">
            <p class="text-gray-600">Category: <span class="font-medium text-blue-600">{{ category }}</span></p>
        </div>
        <a href="/home" class="block mt-4 text-blue-600 hover:text-blue-800 text-center">Back to Home</a>
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
    <nav class="w-full max-w-4xl bg-white shadow-lg rounded-lg p-4 mb-6">
        <div class="flex justify-between items-center">
            <h1 class="text-2xl font-bold text-gray-800">Job Matcher Tool</h1>
            <div class="space-x-4">
                <a href="/home" class="text-blue-600 hover:text-blue-800 font-medium">Home</a>
                <a href="/#classify" class="text-blue-600 hover:text-blue-800 font-medium">Classify Job</a>
                <a href="/#recommend" class="text-blue-600 hover:text-blue-800 font-medium">Get Recommendations</a>
                <a href="/metrics" class="text-blue-600 hover:text-blue-800 font-medium">Metrics</a>
            </div>
        </div>
    </nav>
    <div class="w-full max-w-4xl bg-white shadow-lg rounded-lg p-8">
        <h1 class="text-3xl font-semibold text-gray-800 mb-6 text-center">Job Recommendations</h1>
        <div class="space-y-4">
            {% for job_desc, category, score in recommendations %}
                <div class="p-4 bg-purple-50 rounded-lg">
                    <p class="text-gray-600">{{ job_desc }} (Category: {{ category }}) - Match Score: {{ "%.2f" % score }}</p>
                </div>
            {% endfor %}
        </div>
        <a href="/home" class="block mt-4 text-blue-600 hover:text-blue-800 text-center">Back to Home</a>
    </div>
</body>
</html>
"""

metrics_html = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Model Metrics</title>
    <script src="https://cdn.tailwindcss.com"></script>
</head>
<body class="min-h-screen bg-gradient-to-r from-blue-500 to-purple-600 flex flex-col items-center justify-center p-6">
    <nav class="w-full max-w-4xl bg-white shadow-lg rounded-lg p-4 mb-6">
        <div class="flex justify-between items-center">
            <h1 class="text-2xl font-bold text-gray-800">Job Matcher Tool</h1>
            <div class="space-x-4">
                <a href="/home" class="text-blue-600 hover:text-blue-800 font-medium">Home</a>
                <a href="/#classify" class="text-blue-600 hover:text-blue-800 font-medium">Classify Job</a>
                <a href="/#recommend" class="text-blue-600 hover:text-blue-800 font-medium">Get Recommendations</a>
                <a href="/metrics" class="text-blue-600 hover:text-blue-800 font-medium">Metrics</a>
            </div>
        </div>
    </nav>
    <div class="w-full max-w-4xl bg-white shadow-lg rounded-lg p-8">
        <h1 class="text-3xl font-semibold text-gray-800 mb-6 text-center">Model Metrics</h1>
        {% if error %}
            <p class="text-red-600 mb-4 text-center">{{ error }}</p>
        {% else %}
            <div class="mb-8">
                <h2 class="text-2xl font-semibold text-gray-800 mb-4">Classification Metrics</h2>
                <div class="p-4 bg-blue-50 rounded-lg">
                    <p class="text-gray-600">Precision: <span class="font-medium text-blue-600">{{ classification_metrics.precision }}</span></p>
                    <p class="text-gray-600">Recall: <span class="font-medium text-blue-600">{{ classification_metrics.recall }}</span></p>
                    <p class="text-gray-600">F1-Score: <span class="font-medium text-blue-600">{{ classification_metrics.f1_score }}</span></p>
                </div>
            </div>
            <div>
                <h2 class="text-2xl font-semibold text-gray-800 mb-4">Recommendation Metrics</h2>
                <div class="p-4 bg-purple-50 rounded-lg">
                    <p class="text-gray-600">Precision: <span class="font-medium text-purple-600">{{ recommendation_metrics.precision }}</span></p>
                    <p class="text-gray-600">Recall: <span class="font-medium text-purple-600">{{ recommendation_metrics.recall }}</span></p>
                    <p class="text-gray-600">F1-Score: <span class="font-medium text-purple-600">{{ recommendation_metrics.f1_score }}</span></p>
                </div>
            </div>
        {% endif %}
        <a href="/home" class="block mt-4 text-blue-600 hover:text-blue-800 text-center">Back to Home</a>
    </div>
</body>
</html>
"""

with open('templates/index.html', 'w') as f:
    f.write(index_html)
with open('templates/result.html', 'w') as f:
    f.write(result_html)
with open('templates/recommend.html', 'w') as f:
    f.write(recommend_html)
with open('templates/metrics.html', 'w') as f:
    f.write(metrics_html)

# Instantiate JobMatcher
matcher = JobMatcher()

if __name__ == "__main__":
    matcher.initialize_db()
    job_df = matcher.load_data()
    if job_df.empty:
        if os.path.exists('jobs.csv'):
            logger.info("Loading data from jobs.csv")
            job_df = pd.read_csv('jobs.csv')
            matcher.save_data(job_df)
        else:
            logger.info("No data found. Initializing with sample data (90 jobs).")
            sample_jobs = pd.DataFrame({
                'job_title': [
                    # IT (30 jobs)
                    'Software Engineer', 'Data Scientist', 'Web Developer', 'DevOps Engineer', 'AI Researcher',
                    'Cybersecurity Analyst', 'Database Administrator', 'Cloud Architect', 'Mobile App Developer', 'Systems Analyst',
                    'Machine Learning Engineer', 'Full Stack Developer', 'Network Engineer', 'QA Engineer', 'Data Analyst',
                    'IT Project Manager', 'Blockchain Developer', 'Game Developer', 'Embedded Systems Engineer', 'IT Consultant',
                    'Backend Developer', 'Frontend Developer', 'Security Engineer', 'Data Engineer', 'Site Reliability Engineer',
                    'AI Product Manager', 'Tech Lead', 'Software Architect', 'IoT Developer', 'API Developer',
                    # Marketing (30 jobs)
                    'Marketing Specialist', 'Content Creator', 'SEO Analyst', 'Brand Manager', 'Digital Marketer',
                    'Social Media Manager', 'Public Relations Specialist', 'Market Research Analyst', 'Advertising Manager', 'Copywriter',
                    'Graphic Designer', 'Email Marketing Specialist', 'Content Strategist', 'Event Planner', 'Influencer Marketing Manager',
                    'Product Marketing Manager', 'Marketing Coordinator', 'Media Buyer', 'Creative Director', 'UX Researcher',
                    'Digital Strategist', 'SEO Manager', 'PPC Specialist', 'Marketing Analyst', 'Brand Strategist',
                    'Social Media Analyst', 'Campaign Manager', 'Content Marketing Manager', 'Growth Marketer', 'E-commerce Specialist',
                    # Healthcare (30 jobs)
                    'Registered Nurse', 'Physical Therapist', 'Medical Assistant', 'Pharmacist', 'Surgeon',
                    'Emergency Room Nurse', 'Pediatrician', 'Radiologist', 'Anesthesiologist', 'Clinical Laboratory Technician',
                    'Occupational Therapist', 'Speech-Language Pathologist', 'Dental Hygienist', 'Paramedic', 'Cardiologist',
                    'Psychiatrist', 'Nurse Practitioner', 'Health Informatics Specialist', 'Medical Social Worker', 'Orthopedic Surgeon',
                    'General Practitioner', 'Oncologist', 'Neurologist', 'Dermatologist', 'Physician Assistant',
                    'Respiratory Therapist', 'Dietitian', 'Medical Technologist', 'Chiropractor', 'Epidemiologist'
                ],
                'job_description': [
                    # IT (30 jobs)
                    'Develop software applications using Python and Java.',
                    'Build machine learning models with TensorFlow and PyTorch.',
                    'Create responsive websites using JavaScript and React.',
                    'Manage cloud infrastructure with AWS and Docker.',
                    'Research advanced AI algorithms and neural networks.',
                    'Protect systems from cyber threats and conduct penetration testing.',
                    'Manage and optimize SQL and NoSQL databases.',
                    'Design scalable cloud solutions on Azure and GCP.',
                    'Build iOS and Android apps using Swift and Kotlin.',
                    'Analyze business systems and recommend IT solutions.',
                    'Design and deploy ML models for predictive analytics.',
                    'Develop front-end and back-end web applications.',
                    'Configure and maintain network infrastructure.',
                    'Test software to ensure quality and reliability.',
                    'Analyze data using Python and SQL to generate insights.',
                    'Lead IT projects and coordinate teams.',
                    'Develop decentralized applications using Ethereum.',
                    'Create video games using Unity and C#.',
                    'Program embedded systems for IoT devices.',
                    'Provide IT consulting services to optimize business processes.',
                    'Build server-side applications with Node.js and Express.',
                    'Design user interfaces with React and Tailwind CSS.',
                    'Implement security protocols to protect data.',
                    'Create data pipelines using Apache Spark.',
                    'Ensure system reliability with monitoring tools.',
                    'Manage AI product development lifecycle.',
                    'Lead technical teams on software projects.',
                    'Design software architecture for scalability.',
                    'Develop IoT solutions with MQTT protocols.',
                    'Build and maintain RESTful APIs.',
                    # Marketing (30 jobs)
                    'Manage social media campaigns and branding strategies.',
                    'Produce engaging content for blogs and social media.',
                    'Optimize websites for search engine rankings.',
                    'Develop brand strategies for product launches.',
                    'Create and manage digital ad campaigns.',
                    'Oversee social media platforms and engagement.',
                    'Handle media relations and corporate communications.',
                    'Conduct surveys and analyze consumer trends.',
                    'Plan and execute advertising campaigns.',
                    'Write compelling copy for marketing materials.',
                    'Design visual content using Adobe Creative Suite.',
                    'Develop email marketing campaigns to boost engagement.',
                    'Plan content strategies for brand consistency.',
                    'Organize corporate events and conferences.',
                    'Collaborate with influencers to promote products.',
                    'Market products to target audiences.',
                    'Support marketing campaigns and logistics.',
                    'Purchase advertising space for campaigns.',
                    'Lead creative projects and teams.',
                    'Conduct user research to improve product design.',
                    'Plan digital strategies for online presence.',
                    'Manage SEO efforts to improve rankings.',
                    'Run pay-per-click campaigns on Google Ads.',
                    'Analyze marketing data for insights.',
                    'Develop strategies for brand positioning.',
                    'Analyze social media performance metrics.',
                    'Manage marketing campaigns end-to-end.',
                    'Create content for marketing funnels.',
                    'Drive growth through digital channels.',
                    'Optimize e-commerce marketing strategies.',
                    # Healthcare (30 jobs)
                    'Provide patient care in hospital settings.',
                    'Assist patients with physical rehabilitation programs.',
                    'Support physicians in clinical and administrative tasks.',
                    'Dispense medications and counsel patients.',
                    'Perform surgical procedures in operating rooms.',
                    'Provide critical care in emergency departments.',
                    'Diagnose and treat children’s illnesses.',
                    'Interpret medical imaging for diagnoses.',
                    'Administer anesthesia during surgeries.',
                    'Analyze biological samples in labs.',
                    'Help patients improve daily living skills.',
                    'Treat communication and swallowing disorders.',
                    'Clean teeth and educate patients on oral health.',
                    'Provide emergency medical care in ambulances.',
                    'Diagnose and treat heart conditions.',
                    'Treat mental health disorders with therapy and medication.',
                    'Provide primary care as an advanced practice nurse.',
                    'Manage healthcare data and IT systems.',
                    'Support patients and families with social services.',
                    'Perform surgeries on bones and joints.',
                    'Provide general medical care to patients.',
                    'Treat cancer patients with specialized care.',
                    'Diagnose and treat neurological disorders.',
                    'Treat skin conditions and perform procedures.',
                    'Assist physicians and provide patient care.',
                    'Treat patients with breathing disorders.',
                    'Provide nutritional counseling to patients.',
                    'Perform diagnostic tests in medical labs.',
                    'Treat patients with spinal adjustments.',
                    'Study disease patterns and public health trends.'
                ],
                'category': [
                    # IT (30 jobs)
                    'IT', 'IT', 'IT', 'IT', 'IT',
                    'IT', 'IT', 'IT', 'IT', 'IT',
                    'IT', 'IT', 'IT', 'IT', 'IT',
                    'IT', 'IT', 'IT', 'IT', 'IT',
                    'IT', 'IT', 'IT', 'IT', 'IT',
                    'IT', 'IT', 'IT', 'IT', 'IT',
                    # Marketing (30 jobs)
                    'Marketing', 'Marketing', 'Marketing', 'Marketing', 'Marketing',
                    'Marketing', 'Marketing', 'Marketing', 'Marketing', 'Marketing',
                    'Marketing', 'Marketing', 'Marketing', 'Marketing', 'Marketing',
                    'Marketing', 'Marketing', 'Marketing', 'Marketing', 'Marketing',
                    'Marketing', 'Marketing', 'Marketing', 'Marketing', 'Marketing',
                    'Marketing', 'Marketing', 'Marketing', 'Marketing', 'Marketing',
                    # Healthcare (30 jobs)
                    'Healthcare', 'Healthcare', 'Healthcare', 'Healthcare', 'Healthcare',
                    'Healthcare', 'Healthcare', 'Healthcare', 'Healthcare', 'Healthcare',
                    'Healthcare', 'Healthcare', 'Healthcare', 'Healthcare', 'Healthcare',
                    'Healthcare', 'Healthcare', 'Healthcare', 'Healthcare', 'Healthcare',
                    'Healthcare', 'Healthcare', 'Healthcare', 'Healthcare', 'Healthcare',
                    'Healthcare', 'Healthcare', 'Healthcare', 'Healthcare', 'Healthcare'
                ]
            })
            matcher.save_data(sample_jobs)
        job_df = matcher.load_data()
    if not job_df.empty:
        matcher.train_classifier(job_df)
    matcher.train_matching()
    from waitress import serve
    logger.info("Starting Waitress server on 0.0.0.0:5000")
    serve(app, host='0.0.0.0', port=5000, threads=4)