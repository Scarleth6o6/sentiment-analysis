# SentimentAI: Movie Review Sentiment Classification

## 📌 Project Overview
Film Junky Union, a modern and innovative community for classic movie enthusiasts, is developing a system to filter and categorize movie reviews. The goal of this project is to train a Machine Learning model that can automatically detect negative reviews from a dataset of IMDB movie reviews labeled with sentiment polarity (positive or negative).

The target performance for this model is an **F1 score of at least 0.85**.

## 🛠 Technologies Used
- **Python**
- **Pandas, NumPy** (Data manipulation)
- **Scikit-learn** (Machine Learning models and evaluation)
- **Natural Language Processing (NLP)** (Text preprocessing and vectorization)

## 📊 Models and Performance
Three different models were trained and evaluated:

| Model  | Train Accuracy | Test Accuracy | Train F1 | Test F1 | APS  | ROC AUC |
|--------|---------------|--------------|----------|---------|------|---------|
| **Baseline Model (Constant)** | 0.50 | 0.50 | 0.50 | 0.50 | 0.50 | 0.50 |
| **Model 1** | 0.94 | 0.88 | 0.94 | 0.88 | 0.98 | 0.95 |
| **Model 2** | 0.92 | 0.88 | 0.92 | 0.88 | 0.98 | 0.95 |
| **Model 3** | 0.93 | 0.86 | 0.93 | 0.86 | 0.98 | 0.94 |

### Key Observations
- **Model 1** performed the best, capturing sentiment nuances effectively and providing the most reliable classification.
- **Model 2** had slightly lower accuracy and was less sensitive to subtle tone variations.
- **Model 3** exhibited **overfitting**, performing well on training data but misclassifying many unseen reviews.

## 🚀 How to Run the Project
### 1️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 2️⃣ Run the Model
```bash
python sentiment_analysis.py
```

## 📁 Repository Structure
```
📂 SentimentAI
│-- 📂 data/                 # Dataset files
│-- 📂 models/               # Trained models
│-- 📂 notebooks/            # Jupyter notebooks for analysis & Main script for model training & evaluation
│-- requirements.txt         # List of dependencies
│-- README.md                # Project documentation
```

## 🔗 Future Improvements
- Implement deep learning models (e.g., LSTMs or Transformers) to enhance performance.
- Improve text preprocessing techniques to handle negations and complex sentence structures.
- Conduct additional testing with larger and more diverse datasets.

## 📌 Conclusion
This project successfully developed and evaluated multiple sentiment classification models, with **Model 1 achieving the best results**. The findings highlight the importance of **data preprocessing** and **model selection** in text classification tasks.

📂 **Repository:** [GitHub - SentimentAI](https://github.com/Scarleth6o6/sentiment-analysis)

