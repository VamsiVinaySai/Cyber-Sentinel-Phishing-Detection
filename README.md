# 🛡️ Cyber Sentinel: AI-Powered Phishing URL Detection System

Cyber Sentinel is an AI-driven cybersecurity application that detects phishing URLs in real time using Machine Learning and advanced lexical URL analysis. The system analyzes suspicious patterns in URLs and classifies them as either **Legitimate** or **Phishing** using a trained Random Forest model.

Built with Python and Streamlit, the project provides an interactive web interface for real-time threat analysis and feature inspection.

---

## 🚀 Features

- 🔍 Real-time URL phishing detection
- 🤖 Machine Learning-based classification using Random Forest
- 🌐 Streamlit-powered interactive dashboard
- 🧠 Lexical URL feature extraction
- 📊 Detailed feature breakdown and confidence scores
- ⚡ Lightweight prediction system (no webpage scraping required)
- 🔐 Detects suspicious domains, obfuscation, IP usage, entropy patterns, and more

---

## 🧠 Machine Learning Approach

The model is trained on the **PhiUSIIL Phishing URL Dataset** using a supervised learning approach.

### Model Used
- Random Forest Classifier

### Extracted Features Include
- URL Length
- Domain Length
- IP Address Usage
- HTTPS Presence
- URL Similarity Index
- Character Continuation Rate
- Number of Subdomains
- Obfuscation Ratio
- Special Character Ratio
- Entropy-Based URL Analysis
- Digit & Letter Ratios

---

## 🛠️ Tech Stack

| Technology | Purpose |
|------------|---------|
| Python | Core Programming |
| Streamlit | Web Application UI |
| Scikit-learn | Machine Learning |
| Pandas | Data Processing |
| NumPy | Numerical Computation |
| Joblib | Model Serialization |

---

## 📂 Project Structure

```bash
Cyber-Sentinel/
│
├── app.py                  # Streamlit application
├── features.py             # URL feature extraction logic
├── train_model.py          # Model training script
├── phishing_model.pkl      # Trained ML model
├── feature_cols.pkl        # Saved feature order
├── requirements.txt        # Project dependencies
└── README.md               # Project documentation
⚙️ Installation & Setup
1️⃣ Clone the Repository
git clone https://github.com/your-username/cyber-sentinel.git
cd cyber-sentinel
2️⃣ Install Dependencies
pip install -r requirements.txt
3️⃣ Run the Application
streamlit run app.py

📸 Application Preview
✅ Legitimate URL Detection
Displays safety score
Shows model confidence
Provides feature analysis

🚨 Phishing URL Detection
Detects suspicious domains
Highlights phishing probability
Warns about deceptive similarity patterns

📊 Dataset
This project uses the PhiUSIIL Phishing URL Dataset for training and evaluation.

🎯 Future Enhancements
Deep Learning-based URL analysis
Browser extension integration
Live threat intelligence APIs
URL blacklist integration
Deployment on cloud platforms

👨‍💻 Author
Vamsi Vinay
BTech Computer Science Engineering
