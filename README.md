# IMDB-Sentiment-Analysis
This project is a mini deep learning-based sentiment analysis model trained on the IMDB movie review dataset. It preprocesses text, extracts features using TF-IDF, and trains a neural network to classify reviews as positive or negative. The goal is to get familiar with deep learning workflows.

## 📌 Project Overview  
This project builds a deep learning model to classify IMDB movie reviews as either positive or negative. It involves:  
- Cleaning and preprocessing text data  
- Converting text into numerical features using TF-IDF  
- Building and training a neural network using TensorFlow/Keras  
- Evaluating model performance through accuracy and loss metrics  

## 🔧 Technologies Used  
- Python  
- Pandas, NumPy  
- NLTK for text preprocessing  
- Scikit-learn for feature extraction and data splitting  
- TensorFlow/Keras for deep learning  
- Matplotlib & Seaborn for visualization 

## 🚀 How to Run the Project  

### 1️⃣ **Clone the Repository**  
First, download the project by running:  

```bash
git clone https://github.com/your-username/imdb-sentiment-analysis.git
cd imdb-sentiment-analysis
```
### 2️⃣ Install Dependencies
Ensure you have Python installed, then install required packages:
```bash
pip install -r requirements.txt
```
### 3️⃣ Download the Dataset
you'll find the file in the repo

### 4️⃣ Run the Script
```bash
python sentiment_analysis.py
```

This will:
✅ Load and preprocess the data
✅ Train the deep learning model
✅ Evaluate performance

## 💡 Notes
If you encounter missing dependencies, install them manually using pip install package_name.
The model uses TF-IDF for text transformation instead of embeddings. Future improvements could include LSTMs or Transformers.

