# Sampling Assignment

This repository contains a **Sampling Assignment** that demonstrates various sampling techniques and their implementation using Python. The assignment explores extracting representative subsets from datasets, applying sampling methods, and evaluating the performance of multiple machine learning models.

## Features
- Implementation of multiple sampling techniques with real-world datasets.
- Code written in Python for reproducibility and clarity.
- Model training and testing to evaluate the impact of sampling.
- Visualizations comparing model accuracy across sampling methods.

## Sampling Techniques Covered
- **Random Sampling**: Selecting samples randomly from the dataset.
- **Systematic Sampling**: Selecting every k-th element from an ordered dataset.
- **Oversampling**: Using RandomOverSampler to balance imbalanced datasets.

## Requirements
- Python 3.7+
- Libraries: `numpy`, `pandas`, `matplotlib`, `scikit-learn`, `imblearn`

Install the dependencies using the command:
```bash
pip install -r requirements.txt
```

## Getting Started
1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/sampling-assignment.git
   ```
2. Navigate to the project directory:
   ```bash
   cd sampling-assignment
   ```
3. Run the sampling and model evaluation code:
   ```bash
   python sampling_code.py
   ```

## File Structure
```
├── sampling_code.py   # Code for sampling and model evaluation
├── requirements.txt   # List of dependencies
├── README.md          # Project documentation
└── data               # Sample datasets (if applicable)
```

## Usage
- Update the `file_path` in `sampling_code.py` to point to your dataset.
- Experiment with different sampling strategies to analyze their effect on model performance.
- Compare model accuracies using the generated visualizations.

## Examples
Below is a brief explanation of the workflow:

1. **Dataset Normalization:**
   The `Amount` column in the dataset is normalized for better processing:
   ```python
   dataset['Amount'] = normalize([dataset['Amount']])[0]
   ```

2. **Oversampling to Balance Classes:**
   RandomOverSampler is used to handle class imbalance:
   ```python
   oversample = RandomOverSampler(sampling_strategy=0.99)
   X_balanced, y_balanced = oversample.fit_resample(X_features, y_target)
   ```

3. **Model Evaluation:**
   Models such as Random Forest, Logistic Regression, SVM, KNN, and Gradient Boosting are trained and tested:
   ```python
   def train_and_test_models(X, y):
       models = [RandomForestClassifier(), LogisticRegression(), SVC(), KNeighborsClassifier(), GradientBoostingClassifier()]
       for model in models:
           model.fit(X_train, y_train)
           accuracy = accuracy_score(y_test, model.predict(X_test))
           print(accuracy)
   ```

4. **Visualization:**
   Model accuracies are visualized in a horizontal bar chart:
   ```python
   plt.barh(labels, accuracy_values, color=bar_colors)
   plt.axvline(x=0.95, color='red', linestyle='--', label='95% Threshold')
   plt.legend()
   plt.show()
   ```

## Contributions
Contributions are welcome! Feel free to submit issues or pull requests to enhance the project.


Happy Sampling and Model Evaluation! 🎉

