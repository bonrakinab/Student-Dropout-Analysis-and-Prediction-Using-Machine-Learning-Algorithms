# Student Dropout Analysis and Prediction with Machine Learning

A machine-learning project for analyzing higher-education student outcomes and predicting **dropout / enrollment / graduation status** from demographic, academic, financial, and macroeconomic attributes.

The project compares several supervised-learning algorithms and explores dimensionality reduction, cross-validation, and ensemble methods to identify patterns associated with student retention.

## Dataset

The repository includes `dataset.csv` with **4,424 student records** and **35 columns including the target**.

Feature groups include:

- Application mode and application order
- Course and attendance type
- Previous qualification
- Student and parent demographic attributes
- Debtor / tuition-payment status
- Scholarship status
- Age at enrollment
- First-semester curricular performance
- Second-semester curricular performance
- Unemployment rate
- Inflation rate
- GDP
- Outcome target

The target contains educational-status classes such as **Dropout**, **Enrolled**, and **Graduate**.

## Models explored

The repository contains experiments for:

- Decision Tree
- Logistic Regression
- Support Vector Machine (SVM)
- AdaBoost
- Semi-supervised learning

The broader analysis also compares ensemble approaches such as Random Forest, with Random Forest and AdaBoost reported among the strongest-performing approaches in the project evaluation.

## Experimental workflow

```text
Student dataset
      │
      ▼
Exploratory analysis / preprocessing
      │
      ▼
Feature preparation
      │
      ├── optional PCA
      └── train/test + cross-validation
      │
      ▼
Model training
      ├── Logistic Regression
      ├── Decision Tree
      ├── SVM
      ├── AdaBoost
      └── other comparison experiments
      │
      ▼
Performance evaluation
      │
      ▼
Dropout-risk insights
```

## Reported findings

The original project summary reports approximately **78% classification accuracy** for the strongest experiments, with **Random Forest and AdaBoost** performing particularly well.

Important predictive signals identified in the analysis include:

- Tuition-fee/payment status
- Academic performance
- Curricular-unit completion and grades
- Demographic/student-background variables

The notebooks should be treated as the source of truth for the exact configuration and metrics of each individual experiment.

## Repository structure

```text
Student-Dropout-Analysis-and-Prediction-Using-Machine-Learning-Algorithms/
├── dataset.csv
├── Adaboost.ipynb
├── Decision Tree.ipynb
├── Logistic Regression.ipynb
├── Logistic Regression Dataset 2.ipynb
├── SVM.ipynb
├── Semi Supervised Learning.ipynb
└── README.md
```

## Running the experiments

The notebooks can be used in Jupyter Notebook, JupyterLab, or Google Colab.

A typical local environment:

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install pandas numpy scikit-learn matplotlib seaborn jupyter
jupyter notebook
```

Open the desired notebook and update the dataset path if required.

## Techniques used

- Exploratory data analysis
- Categorical/numeric feature handling
- Supervised classification
- Decision trees
- Logistic regression
- Support Vector Machines
- AdaBoost
- Principal Component Analysis (PCA)
- 5-fold cross-validation
- Model comparison

## Why dropout prediction matters

A useful dropout model can help institutions identify patterns associated with academic risk earlier. In practice, such a system should be used as **decision support** rather than as an automated decision-maker: predictions can help prioritize voluntary support and outreach, but they should not be used to deny opportunities or label individual students without careful human review.

## Limitations

- Results are specific to the dataset and evaluation setup.
- Educational outcomes are influenced by factors that may not be represented in the data.
- Demographic variables can encode social bias and require careful fairness analysis.
- Predictive correlation does not establish causation.
- The repository is organized as research notebooks rather than a single reproducible training package.

## Future improvements

- Consolidate preprocessing into a shared pipeline
- Add a pinned `requirements.txt`
- Add stratified cross-validation and per-class metrics
- Add confusion matrices for all models
- Perform class-imbalance analysis
- Add feature importance / SHAP explanations
- Measure fairness across relevant demographic groups
- Add calibration and uncertainty estimates
- Build a small dashboard for aggregate, privacy-conscious risk analysis

## Tech stack

- **Language:** Python
- **Data:** Pandas / NumPy
- **Machine learning:** scikit-learn
- **Environment:** Jupyter / Google Colab
- **Domain:** educational data mining, student retention, classification

---

This repository documents a comparative machine-learning study of student outcomes, with dropout detection as the primary applied objective.
