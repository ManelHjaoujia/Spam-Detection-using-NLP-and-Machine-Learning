# Mini NLP Project — Spam vs Ham Classification

**Author:** Manel Hjaoujia  
**Level:** Junior Data Scientist  
**Project Type:** Learning Project — Natural Language Processing (NLP)  
**Repository:** [nlp](https://github.com/ManelHjaoujia/nlp)

---

## Overview

This mini-project focuses on building and comparing Natural Language Processing (NLP) models for **Spam Detection** — classifying messages as **Spam** or **Ham (Not Spam)**.

It demonstrates your understanding of:
- Text preprocessing (cleaning, tokenization, stopword removal)
- Feature extraction (TF-IDF and Word2Vec embeddings)
- Machine Learning model training and evaluation
- Comparative analysis of classic and embedding-based approaches

The project is built entirely in **Python** using **Jupyter Notebooks**.

---

## Repository Structure

nlp/

├── SpamAndHamPrediction.ipynb # Notebook: Baseline ML using TF-IDF

├── SpamHamUsingWord2vecAvgWord2vec.ipynb # Notebook: Word2Vec embedding-based model

├── spam.csv # Dataset (SMS messages)

└── README.md # Project documentation



---

## Objectives

1. Load and preprocess text data for spam detection.  
2. Implement traditional ML models using TF-IDF features.  
3. Implement an embedding-based model using averaged Word2Vec vectors.  
4. Evaluate and compare the performance of both approaches.  
5. Draw conclusions and identify possible improvements.

---

## Dataset

**File:** `spam.csv`  
**Source:** Publicly available SMS Spam Collection Dataset.  
**Description:** Contains labeled SMS messages as either:
- `ham` → legitimate message  
- `spam` → unsolicited or unwanted message  

Example:

| Label | Message |
|--------|----------|
| ham | Go until jurong point, crazy.. Available only in bugis n great world la e buffet... |
| spam | Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005... |

---

## Installation & Setup

### 1. Clone this repository
```bash
git clone https://github.com/ManelHjaoujia/nlp.git
cd nlp
```

### 2. Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate     
venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```
### 4. Run Jupyter Notebooks
```bash
jupyter notebook
```
Open and execute the notebooks in this order:

* 1. SpamAndHamPrediction.ipynb

* 2. SpamHamUsingWord2vecAvgWord2vec.ipynb

## Notebook Details

1. SpamAndHamPrediction.ipynb:
                  
**Goal**: Build a baseline model using TF-IDF features. 

**Key Steps:**
  * Data loading and cleaning
  * Tokenization, lowercasing, stopword removal
  * Feature extraction via TF-IDF
  * Training models such as:
    - Logistic Regression
    - Naive Bayes
    - Support Vector Machine (SVM)
  * Evaluation with accuracy, precision, recall, and F1-score

2. SpamHamUsingWord2vecAvgWord2vec.ipynb

**Goal:** Use Word2Vec embeddings to represent text semantically.

**Key Steps:**

* Tokenization and Word2Vec vectorization

* Averaging word vectors for each message

* Classifier training (e.g., Logistic Regression, RandomForest)

* Comparison with TF-IDF model performance

## Results Summary

| Model                | Feature Type  | Accuracy | Precision | Recall | F1-Score |
|----------------------|---------------|-----------|------------|---------|-----------|
| Logistic Regression  | TF-IDF        | xx%       | xx%        | xx%     | xx%       |
| RandomForest         | Word2Vec Avg  | xx%       | xx%        | xx%     | xx%       |

---

### Interpretation

TF-IDF models tend to perform well for **spam classification** due to keyword-based patterns,  
while **Word2Vec embeddings** capture deeper semantic meaning, potentially improving generalization.

---

## Technologies Used

| Category         | Tools / Libraries             |
|------------------|-------------------------------|
| Programming      | Python                        |
| Environment      | Jupyter Notebook              |
| NLP              | NLTK, Gensim                  |
| ML & Evaluation  | Scikit-learn                  |
| Visualization    | Matplotlib, Seaborn           |
| Data Handling    | Pandas, NumPy                 |

---

## Key Learnings

- Understanding of **text preprocessing pipelines** for NLP  
- Comparison of **vectorization techniques** (TF-IDF vs Word2Vec)  
- Practical application of **classification algorithms** on text data  
- Building and evaluating **end-to-end NLP pipelines**  
- Familiarity with **model evaluation metrics** and result interpretation  

---


## Example Prediction 

```bash
Input: "Congratulations! You won a free ticket to Bahamas!"
Predicted Label: SPAM
Confidence: 0.97
```

---

## Contribution

Feel free to fork the repository and open a pull request if you’d like to improve it.

1. **Fork the repo**  
2. **Create your feature branch:**  
   ```bash
   git checkout -b feature/your-feature
   ```
3. **Commit changes:**
   ```bash
   git commit -m "Added feature"
   ```
4. **Push to the branch:**
   ```bash
   push origin feature/your-feature
   ```
5. **Open a Pull Request**

## Author

**Manel Hjaoujia**

Master’s Student — Information Systems Engineering & Data Science

Passionate about NLP, Data Science, and AI applications

