# VISH-GARD-_call-fraud-detection-using-ml-model-
call fraud detection by ml and  NLP models .this project  can analyze live audio  and text also can detect spam numbers



PROJECT OVERVIEW

VOICE GARD is an AI-based fraud detection system that identifies scam phone calls, fraudulent text messages, and suspicious phone numbers in real time.

The system works offline and combines:

Speech-to-Text processing

Natural Language Processing (NLP)

Machine Learning classification

Keyword-based fraud detection

Phone number blacklist checking

It is designed to help users identify scam calls while speaking or analyzing call text content.

OBJECTIVES

Detect scam calls in real time

Convert live voice to text and analyze continuously

Identify fraud-related keywords such as OTP, bank, PIN, account number

Check phone numbers against known fraud lists

Store fraud evidence securely

Provide confidence scores for each prediction

SYSTEM ARCHITECTURE

Audio Input / Text Input / Phone Number
↓
Text Preprocessing (Cleaning, Stopwords Removal)
↓
TF-IDF Vectorizer
↓
Machine Learning Model (Naive Bayes)
↓
Keyword Detection + Phone Number Rules
↓
Final Fraud Decision
↓
User Interface + Evidence Logs

KEY FEATURES

LIVE AUDIO SCAM DETECTION

Offline speech recognition using Vosk

Word-by-word transcription

Live keyword highlighting

Confidence score during call

Audio waveform visualization

TEXT / CALL SENTENCE ANALYSIS

Analyze typed call messages

Detect scam intent instantly

Highlight suspicious words

PHONE NUMBER FRAUD CHECKING

Prefix and pattern analysis

Blacklist verification from file

Confidence score with reasons

FRAUD EVIDENCE LOGGING

Saves only one evidence per call

Stored in transcripts.log file

Option to view, delete, or clear logs

MACHINE LEARNING DETAILS

Algorithm Used:
Bernoulli Naive Bayes

Text Representation:
TF-IDF Vectorizer

Dataset Type:
Spam messages and fraud call sentences

Language:
Python

The ML model is combined with keyword-based rules to improve real-world accuracy.


INSTALLATION STEPS

Create Virtual Environment

python -m venv venv

Activate:
Windows: venv\Scripts\activate
Linux/Mac: source venv/bin/activate

Install Required Libraries

pip install flask vosk pyaudio nltk scikit-learn

Download Offline Speech Model

Download and extract:
vosk-model-small-en-us

Place it in the project root directory.



FRAUD LOGS AND EVIDENCE

Fraud calls are stored in:
transcripts.log

Each log contains:

Time

Transcribed text

Confidence score

Detected keywords

Only one entry is saved per call session.



CONFIDENCE SCORING

Confidence is calculated using:

Machine learning probability

Keyword frequency

OTP detection (immediate fraud)

Blacklisted phone number (100% fraud)

ACADEMIC USE

This project demonstrates:

NLP preprocessing

Machine learning classification

Speech processing

Cyber security concepts

Real-time web application development

Suitable for:

Mini Project

Final Year Project

Cyber Forensics

AI & ML Labs



