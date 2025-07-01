# Arabic Dialect Identification Using Machine Learning and Transformers

## 📌 Project Overview

This project aims to build a system that automatically classifies Arabic dialects from tweets using the **SADA Dataset**. Arabic dialect identification is a complex but vital task in Natural Language Processing (NLP), especially in applications such as translation, sentiment analysis, and conversational AI.

We evaluate and compare three models:

* ✅ **Logistic Regression with TF-IDF** (baseline)
* 🤖 **AraBERT** (fine-tuned transformer model)
* 🧠 **MARBERT** (transformer trained on Arabic dialectal tweets)

---

## 🧠 Problem Statement

Traditional NLP tools either:

* Treat Arabic as a monolithic language (ignoring dialects), or
* Fail to capture informal or dialectal nuances used in social media.

**Goal**: Build a classifier that accurately distinguishes between Arabic dialects to improve Arabic NLP applications.

---

## 🎯 Project Objectives

* Evaluate multiple models (Logistic Regression, AraBERT, MARBERT)
* Handle class imbalance and dialectal diversity
* Analyze model accuracy, F1-score, and recall
* Preprocess Arabic text using custom NLP pipelines

---

## 🧪 Dataset

* **Source**: \[SADA Dataset 2022]
* **Size**: \~241,000 Arabic audio transcriptions (tweets)
* **Target**: `SpeakerDialect`
* **Input**: `GroundTruthText` (raw tweet)

---

## 🔧 Preprocessing Steps

We applied a tailored Arabic text cleaning pipeline:

* Normalization of Arabic letters (e.g., أ → ا)
* Removal of diacritics, emojis, English characters, and noise
* Tokenization and stopword removal using NLTK
* Handling repeated characters (e.g., "كووويس" → "كويس")
* Balanced data through class weighting

> 💡 Two Python classes were used:
>
> * `PreprocessTweets`: Cleans and normalizes Arabic text
> * `TweetsTokenizing`: Tokenizes and filters tokens for modeling

---

## 🏗️ Models & Implementation

### ✅ Logistic Regression (Baseline)

* TF-IDF vectorization with 5k, 10k, and all features
* Trained using Scikit-learn
* Class imbalance handled using `class_weight='balanced'`

### 🤖 AraBERT

* Pretrained on MSA and formal Arabic
* Fine-tuned using Hugging Face Transformers
* Used `arabert.preprocess` for segmentation and normalization

### 🧠 MARBERT

* Pretrained on dialectal tweets
* Tokenized using SentencePiece
* Best performance across underrepresented dialects

---

## 📊 Evaluation Metrics

* **Accuracy**
* **Macro F1-score**
* **Macro Recall**
* **Confusion Matrix**
* **Training/Validation Loss by Epoch**



---

## 🔍 Key Insights

* **MARBERT** outperformed other models in overall balance and minority dialect recall
* **AraBERT** was more effective on formal dialects like MSA
* **Logistic Regression** worked well with class balancing, despite being simpler

---

## ⚙️ Tools & Libraries

* `Python`, `Pandas`, `NumPy`
* `Scikit-learn`, `NLTK`
* `Transformers` (HuggingFace)
* `Google Colab` (training environment)
* `Seaborn`, `Matplotlib`, `Plotly` (visualizations)

---

## 🔄 Future Work

* 📈 Add data augmentation for minority dialects (e.g., SMOTE)
* 🧩 Use multimodal input (text + audio)
* 🧠 Try ensemble models (e.g., AraBERT + LR)
* 🔍 Apply explainability tools (e.g., SHAP, LIME)
* 📊 Add k-fold cross-validation


## 📚 References

1. AraBERT: [https://huggingface.co/aubmindlab/bert-base-arabert](https://huggingface.co/aubmindlab/bert-base-arabert)
2. MARBERT: [https://huggingface.co/UBC-NLP/MARBERT](https://huggingface.co/UBC-NLP/MARBERT)

