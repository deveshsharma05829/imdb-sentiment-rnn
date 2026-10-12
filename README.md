# RNN for Sentiment Analysis on IMDB Movie Reviews

A PyTorch Recurrent Neural Network (RNN) that classifies IMDB movie reviews as positive or negative. The project covers the full pipeline: data cleaning, text normalization, stemming, TF-IDF vectorization, model training, and evaluation.

## Results

| Metric | Value |
|---|---|
| Test accuracy | 85.32% |
| Test set size | 9,917 reviews |
| Training epochs | 10 |
| Final training loss | ~0.159 |

## Dataset

- File: IMDB Dataset.csv
- Columns: review (text), sentiment (positive / negative)
- Size: 50,000 reviews, reduced to 49,582 after removing duplicates
- Missing values: none

## Pipeline

### 1. Data cleaning
- Removed 418 duplicate reviews
- Converted text to lowercase
- Removed URLs with regex (http\S+)
- Removed punctuation and special characters
- Removed HTML tags

### 2. Text preprocessing
- Removed English stopwords using NLTK
- Applied Porter stemming (e.g., *running → run*, *played → play*)

### 3. Label encoding
- Encoded sentiment as binary: negative → 0, positive → 1 using LabelEncoder

### 4. Feature extraction
- TF-IDF vectorization with max_features=5000

### 5. Train/test split
- 80/20 split with random_state=42
- Training set: 39,665 × 5,000
- Test set: 9,917 × 5,000

### 6. Model
- Framework: PyTorch
- Architecture:
  - nn.RNN layer with hidden_size=128, num_layers=1, batch_first=True
  - Fully connected output layer
  - Sigmoid activation for binary probability output
- Batch size: 64

### 7. Training
- 10 epochs, with per-epoch loss logging
- Loss values fluctuate across epochs, so the final epoch is not necessarily the best one

### 8. Evaluation
- Predictions are thresholded at 0.5
- Accuracy is computed on the held-out test set

## Project Structure

.
├── RNN_for_sentiment_analysis.ipynb   # Main notebook with the full pipeline
├── IMDB Dataset.csv                   # Dataset (see Dataset section)
└── README.md## Requirements

- Python 3.8+
- pandas
- scikit-learn
- nltk
- torch
- numpy

Install with:

pip install pandas scikit-learn nltk torch numpyDownload the NLTK resources used in the notebook:

import nltk
nltk.download("punkt")
nltk.download("punkt_tab")
nltk.download("stopwords")## How to Run

1. Clone the repository:
  
   git clone https://github.com/<deveshsharma05829>/<imdb-sentiment-rnn>.git
   cd <imdb-sentiment-rnn>
   2. Place IMDB Dataset.csv in the project root.
3. Launch Jupyter:
  
   jupyter lab
   4. Open RNN_for_sentiment_analysis.ipynb and run all cells.

## Possible Improvements

- Use pretrained word embeddings (GloVe or Word2Vec) instead of TF-IDF
- Replace the vanilla RNN with LSTM or GRU to handle long-range dependencies
- Fit the TF-IDF vectorizer on the training split only to avoid data leakage
- Add early stopping and a validation set
- Try bidirectional layers or stacked RNNs

