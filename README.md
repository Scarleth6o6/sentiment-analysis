# Sentiment Analysis with Machine Learning

This project applies Machine Learning models to classify sentiment in text reviews. The goal is to compare the performance of different models and analyze their effectiveness in predicting positive and negative sentiments.

## Project Overview
- **Model 1**: Shows good performance in capturing sentiment nuances.
- **Model 2**: Works reasonably well but lacks sensitivity to sentiment subtleties.
- **Model 3**: Demonstrates overfitting, performing well on training data but poorly on unseen data.

## Features
- Data preprocessing (tokenization, vectorization, text cleaning)
- Sentiment classification using different models
- Performance evaluation using metrics like Accuracy, F1-Score, and ROC-AUC

## Installation

1. Clone the repository:
   ```sh
   git clone https://github.com/yourusername/sentiment-analysis.git
   cd sentiment-analysis
   ```
2. Create a virtual environment (optional but recommended):
   ```sh
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\\Scripts\\activate`
   ```
3. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```

## Usage

Run the main script to train and evaluate models:
```sh
python main.py
```

## Dependencies

Create a `requirements.txt` file with the following content:
```txt
numpy
pandas
scikit-learn
matplotlib
seaborn
nltk
```
