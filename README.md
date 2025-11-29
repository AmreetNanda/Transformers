# Transformer Project

This repository contains tutorials and practical implementations of **Transformer models** for Natural Language Processing (NLP). The project includes two Jupyter notebooks: one for understanding transformer concepts and the other for applying transformers to a real-world task—spam classification.

##  What is Transformer?
Transformer library features, the library downloads the pre-trained models for natural language understanding (NLU Tasks) such as analyzing the sentiments of a text and NLG (Natural Language Generation), such as completing a prompt with new text or translating in another language.

First we will see how to easily leverage hte pipelines API to quickly use those pre-trained models at interface. Thus we will dig a little bitmore and see how the library gives access to those models and helps in preprocess the data

### USE_CASE
1. Sentiment Analysis : Whether positive or negetive
2. Text Generation : Provide a prompt and model will generate what follows
3. Name Entity Recognition (NER) : In an input sentence, label each word with entity it represents (person, place)
4. Question Answering : Provide the model with same context and a question, extract the answers from the context
5. Filling masked text : Given a text with masked word and fill in the blanks
6. Summerization : Generate a summary of long text
7. Translation : Translates a text into another language
8. Feature Extraction : Return a tensor representation of the text

-----------

## 📂 Repository Structure
```bash
transformer-project/
│
├── Dataset/spam.csv
├── Transformer.ipynb # Notebook explaining transformer architecture and theory
├── Spam_Detection_using_Transformers.ipynb # Notebook applying transformers for spam classification
├── README.md # This file
└── requirements.txt # Python dependencies
```
## 📝 Notebooks

### 1. `Transformer_Concepts.ipynb`

This notebook provides a conceptual overview of transformers, including:
It is ideal for beginners who want to understand the **inner workings of transformer models**.

--------

### 2. `Spam_Classification_Transformer.ipynb`

This notebook is a practical application of transformers for NLP:

- Preprocessing text data for NLP
- Tokenizing and encoding text using Hugging Face Transformers
- Fine-tuning a pretrained transformer model (e.g., distilbert-base-uncased) for spam detection
- Training and evaluating the model on a spam dataset
- Generating metrics like confusion matrix

It demonstrates **implementation of a transformer model for a real-world text classification problem**.

--------

## ⚙️ Installation

Clone this repository and install the required dependencies:

```bash
git clone https://github.com/AmreetNanda/Transformers.git
cd Transformers
pip install -r requirements.txt

## requirements.txt 

numpy
pandas
scikit-learn
matplotlib
seaborn 
ipykernel
Flask
nltk
tqdm
tensorflow
keras
fsspec
transformers
torch
tf-keras