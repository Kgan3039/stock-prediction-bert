# BERT-Based Stock Prediction Model

This project fine-tunes a BERT (Bidirectional Encoder Representations from Transformers) model to analyze sentiment from financial text data and predict stock price movement trends.

The goal of this project is to explore how natural language processing (NLP) can be used to extract meaningful signals from text and apply them to financial forecasting.

---

## Features

- Fine-tunes pretrained BERT model using HuggingFace Transformers
- Processes and tokenizes custom sentiment dataset
- Trains deep learning model using PyTorch
- Evaluates model performance on validation data
- Runs entirely in Google Colab with GPU acceleration

---

## Technologies Used

- Python
- PyTorch
- HuggingFace Transformers
- Pandas
- NumPy
- Scikit-learn
- Google Colab

---

## Repository Structure

stock-prediction-bert/
│
├── BertModel.ipynb # Main training notebook
├── README.md # Project documentation


---

## Model Overview

This project uses:

- Pretrained BERT base model
- Fine-tuning on labeled sentiment dataset
- Tokenization using BERT tokenizer
- Classification head for prediction

---

## How to Run

1. Open the notebook in Google Colab
2. Enable GPU:
   Runtime → Change runtime type → GPU
3. Run all cells

---

## Purpose

This project was built to:

- Learn transformer-based deep learning models
- Apply NLP techniques to financial data
- Explore real-world machine learning workflows

---

## Author

**Kartik Gangwar**

University of Wisconsin–Madison  
Computer Science & Data Science

---

## Future Improvements

- Add larger financial dataset
- Improve model accuracy
- Deploy model as web app
- Add real-time prediction pipeline

