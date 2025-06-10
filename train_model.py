from jobs import JobMatcher
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense, Dropout
import pickle

# Expanded dataset for classification (45 samples)
job_descriptions = [
    # IT (15 samples)
    "Software Engineer position requiring Python, Flask, and machine learning skills.",
    "Data Scientist role needing expertise in TensorFlow and data analysis.",
    "We are hiring a Software Engineer to work on machine learning projects requiring Python and TensorFlow.",
    "Looking for an ideal candidate with experience in CNN models and text classification.",
    "We are hiring a Software Engineer to work on machine learning projects. The ideal candidate should have experience with Python, Flask, and TensorFlow. Knowledge of CNN models is a plus.",
    "Web Developer role requiring JavaScript and React.",
    "Backend Developer with experience in Node.js and MongoDB.",
    "AI Researcher needed for advanced algorithm development.",
    "Full Stack Developer proficient in Python and Angular.",
    "DevOps Engineer with AWS and Docker experience.",
    "Cybersecurity Analyst with expertise in network security.",
    "Database Administrator skilled in SQL and Oracle.",
    "Mobile App Developer with experience in Swift and Kotlin.",
    "Cloud Architect needed for AWS infrastructure design.",
    "Machine Learning Engineer with experience in PyTorch.",
    # Marketing (15 samples)
    "Marketing Manager needed with experience in digital campaigns and SEO.",
    "Content Writer for marketing team, skilled in copywriting and social media.",
    "Digital Marketing Specialist with expertise in Google Ads and analytics.",
    "Marketing Coordinator required for event planning and branding.",
    "Social Media Manager needed with experience in content creation and strategy.",
    "SEO Specialist needed with experience in keyword research.",
    "Brand Strategist for product launches and campaigns.",
    "Advertising Manager with experience in media buying.",
    "Marketing Analyst with data analysis skills.",
    "PR Specialist for public relations and media outreach.",
    "Email Marketing Specialist with experience in campaign automation.",
    "Market Research Analyst to analyze consumer trends.",
    "Influencer Marketing Manager for social media collaborations.",
    "Content Marketing Specialist with SEO writing skills.",
    "Product Marketing Manager for product launches.",
    # Healthcare (15 samples)
    "Nurse Practitioner required with experience in patient care and diagnostics.",
    "Medical Assistant needed for clinic, must have knowledge of EHR systems.",
    "Healthcare Administrator needed to manage hospital operations.",
    "Registered Nurse with 3 years of experience in emergency care.",
    "Physical Therapist required for patient rehabilitation and therapy.",
    "Doctor needed for hospital with 5 years of experience.",
    "Pharmacist to dispense medications and counsel patients.",
    "Surgeon needed for advanced surgical procedures.",
    "Radiologist with experience in imaging diagnostics.",
    "Therapist for mental health counseling and support.",
    "Pediatrician with experience in child healthcare.",
    "Cardiologist needed for heart-related treatments.",
    "Dental Hygienist for patient dental care.",
    "Medical Laboratory Technician for diagnostic testing.",
    "Occupational Therapist for patient rehabilitation."
]

categories = [
    "IT", "IT", "IT", "IT", "IT", "IT", "IT", "IT", "IT", "IT", "IT", "IT", "IT", "IT", "IT",
    "Marketing", "Marketing", "Marketing", "Marketing", "Marketing", "Marketing", "Marketing", "Marketing", "Marketing", "Marketing", "Marketing", "Marketing", "Marketing", "Marketing", "Marketing",
    "Healthcare", "Healthcare", "Healthcare", "Healthcare", "Healthcare", "Healthcare", "Healthcare", "Healthcare", "Healthcare", "Healthcare", "Healthcare", "Healthcare", "Healthcare", "Healthcare", "Healthcare"
]

# Update JobMatcher to increase Dropout rate
class JobMatcherUpdated(JobMatcher):
    def build_classification_model(self):
        model = Sequential([
            Embedding(self.vocab_size, self.embedding_dim),
            Conv1D(128, 5, activation='relu'),
            GlobalMaxPooling1D(),
            Dense(64, activation='relu'),
            Dropout(0.7),  # Increased from 0.5 to 0.7
            Dense(3, activation='softmax')
        ])
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        return model

# Instantiate the updated JobMatcher
matcher = JobMatcherUpdated()

# Train the model with evaluation
all_texts = job_descriptions
matcher.tokenizer.fit_on_texts(all_texts)
sequences = matcher.tokenizer.texts_to_sequences(job_descriptions)
padded = pad_sequences(sequences, maxlen=matcher.max_length, padding='post', truncating='post')
y = np.array([{"IT": 0, "Marketing": 1, "Healthcare": 2}[cat] for cat in categories])

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(padded, y, test_size=0.2, random_state=42)
matcher.classification_model.fit(X_train, y_train, epochs=20, validation_split=0.2, batch_size=32)  # Increased epochs to 20

# Evaluate on test set
y_pred = np.argmax(matcher.classification_model.predict(X_test), axis=1)
precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted', zero_division=0)
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1-Score: {f1:.2f}")

# Save the model and tokenizer
matcher.classification_model.save(matcher.classification_model_path)
with open(matcher.tokenizer_path, 'wb') as f:
    pickle.dump(matcher.tokenizer, f)

print("Classification model trained and saved successfully!")