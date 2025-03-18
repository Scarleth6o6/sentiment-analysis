# Sentiment Analysis with Machine Learning
This project applies Machine Learning models to classify sentiment in text reviews. The goal is to compare the performance of different models and analyze their effectiveness in predicting positive and negative sentiments.


### Project Overview
- **Model 1**: Shows good performance in capturing sentiment nuances.
- **Model 2**: Works reasonably well but lacks sensitivity to sentiment subtleties.
- **Model 3**: Demonstrates overfitting, performing well on training data but poorly on unseen data.

### Features
- Data preprocessing (tokenization, vectorization, text cleaning)
- Sentiment classification using different models
- Performance evaluation using metrics like Accuracy, F1-Score, and ROC-AUC
  
### Dataset
The dataset used for this project consists of a labeled set of text data, where each text sample is associated with a category label. This data is pre-processed and split into training and testing datasets.

### Methodology
1.- Data Pre-processing:
- Tokenization: Splitting the text into words or tokens.
- Stopword Removal: Removing common words that don’t contribute meaningful information.
- Lemmatization: Reducing words to their root form.

2.- Feature Extraction:
- TF-IDF (Term Frequency-Inverse Document Frequency): Converts the text data into numerical features that reflect the importance of terms in the dataset.

3.- Modeling:
- A variety of machine learning algorithms were tested, including Naive Bayes, Logistic Regression, and Random Forests. The best performing model was selected for further optimization.

4.- Evaluation:
- The model’s performance was evaluated using metrics such as accuracy, precision, recall, and F1-score.
### Requirements
- Python 3.x
- Libraries:
- pandas
- numpy
- scikit-learn
- nltk
- matplotlib
- seaborn
### Installation
1.- Clone the repository:
```bash
git clone https://github.com/your-username/text-classification-project.git
```
2.- Navigate into the project folder:
``` bash
cd text-classification-project
```
3.- Install the required dependencies:
``` bash
pip install -r requirements.txt
```
### Usage
To train and evaluate the model, run the following command:
``` bash
python train_model.py
```
This will train the model on the provided dataset and output evaluation metrics. You can also modify the script to test the model on new text data.

### Results
The model achieved an accuracy of 92% on the test set, demonstrating its ability to classify text accurately. Further optimizations, such as hyperparameter tuning, could improve the model’s performance.

### Future Work
- Experiment with deep learning models (e.g., LSTM, BERT) for better performance.
- Improve data preprocessing techniques to handle noisy text data.
- Extend the model to handle multilingual text classification.

### Acknowledgments
- Special thanks to the authors of the dataset.
- Inspiration from online tutorials and machine learning communities.
