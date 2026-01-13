# PSO-Optimized Artificial Neural Network for Dementia Prediction using Cognitive and Clinical Features

## Project Overview
Dementia is a progressive neurological disorder that affects memory, cognition, and daily functioning. Early detection plays a crucial role in timely intervention and improved patient care.

This project presents a **PSO-optimized Artificial Neural Network (ANN)** framework for predicting dementia using **cognitive assessment scores and clinical features**. The methodology integrates **advanced preprocessing, detailed exploratory data analysis (EDA), Genetic Algorithm (GA)–based feature selection, and Particle Swarm Optimization (PSO) for ANN hyperparameter tuning**, resulting in a robust and efficient predictive system.

---

## Objectives
- Analyze cognitive and clinical indicators associated with dementia
- Perform detailed EDA to understand feature distributions and data quality
- Apply advanced preprocessing techniques for missing values and outliers
- Use **Genetic Algorithm (GA)** to identify the most informative features
- Optimize ANN hyperparameters using **Particle Swarm Optimization (PSO)**
- Build an accurate and generalizable dementia prediction model

---

## Dataset Description
- The dataset contains cognitive test scores, demographic details, and clinical indicators related to dementia
- The target variable represents dementia diagnosis status
- Features include both numerical and categorical attributes relevant to neurological health

---

## Exploratory Data Analysis (EDA)
A comprehensive EDA was conducted to:
- Analyze feature distributions and skewness
- Identify outliers and anomalies
- Study correlations among cognitive and clinical variables
- Examine class imbalance
- Understand missing value patterns and their potential impact

EDA insights guided preprocessing, feature selection, and model design decisions.

---

## Data Preprocessing
Advanced preprocessing techniques were applied, including:
- Missing value handling based on feature distribution and data characteristics
- Outlier detection and treatment
- Skewness correction using appropriate transformations
- Feature scaling and normalization
- Encoding of categorical variables

These steps ensured clean, consistent, and model-ready input data.

---

## Feature Selection using Genetic Algorithm (GA)
To reduce dimensionality and improve model performance:
- A **Genetic Algorithm (GA)** was employed to identify the most relevant cognitive and clinical features
- GA effectively explored the feature space using evolutionary principles
- Selected features improved both predictive accuracy and model interpretability

---

## Artificial Neural Network (ANN) Model
- A feedforward Artificial Neural Network was designed for dementia prediction
- The ANN architecture includes multiple hidden layers and non-linear activation functions
- The model learns complex relationships between cognitive and clinical features

---

## Hyperparameter Optimization using Particle Swarm Optimization (PSO)
To achieve optimal ANN performance:
- **Particle Swarm Optimization (PSO)** was used to tune critical ANN hyperparameters, including:
  - Number of hidden layers
  - Number of neurons per layer
  - Learning rate
  - Batch size
  - Activation functions
- PSO efficiently searched the hyperparameter space to minimize model error and improve generalization

---

## Model Evaluation
The optimized ANN model was evaluated using:
- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

Recall and F1-score were prioritized due to the clinical importance of minimizing false negatives in dementia prediction.

---

## Technologies Used
- Python
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- Keras / TensorFlow
- Evolutionary Optimization Techniques (GA, PSO)

---

## Key Highlights
- PSO-based hyperparameter optimization for ANN
- GA-driven feature selection for improved model efficiency
- Detailed EDA and advanced preprocessing
- Clinically relevant dementia prediction framework
- Combines evolutionary algorithms with neural networks

---

## Future Scope
- Integration of explainable AI (XAI) techniques
- Extension to multi-class dementia severity prediction
- Deployment as a clinical decision support system
- Validation on larger and multi-center datasets

---

## Disclaimer
This project is intended strictly for academic and research purposes.  
It is not a substitute for professional medical diagnosis or treatment.
