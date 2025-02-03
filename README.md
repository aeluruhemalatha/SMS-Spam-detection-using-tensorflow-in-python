# SMS Spam Detection using TensorFlow in Python

This repository contains a Jupyter notebook that demonstrates the development and evaluation of various machine learning models for SMS spam detection using TensorFlow in Python.

## Project Overview
This project involves the development and comparison of several machine learning models for detecting SMS spam. The Jupyter notebook included in this repository utilizes TensorFlow in Python to implement and evaluate four distinct models:

- **Multinomial Naive Bayes**: Serves as the baseline model for comparison.
- **Custom Vector Embeddings Model**: Utilizes tailored text vectorization techniques to convert SMS messages into a numerical format.
- **Bidirectional LSTM Model**: A deep learning approach for sequence processing using Long Short-Term Memory (LSTM) networks.
- **USE (Universal Sentence Encoder) Transfer Learning Model**: Leverages pre-trained models for enhanced performance in text classification tasks.

### Evaluation Metrics:
Each model is evaluated based on the following metrics:
- **Accuracy**
- **Precision**
- **Recall**
- **F1-score**

Results are visualized for easy comparison, allowing for a comprehensive assessment of model performance.

## Technologies Used
- **Python**
- **TensorFlow**
- **Keras**
- **Scikit-learn**
- **Pandas**
- **NumPy**
- **Matplotlib** (for visualizations)
- **Jupyter Notebook** (for development and experimentation)

## Dataset
- The **SMS Spam Collection Dataset** from [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/sms+spam+collection) is used to train and evaluate the models. This dataset consists of labeled SMS messages categorized as either "spam" or "ham" (non-spam).

## Installation
To get started with the project, follow these steps:

1. Clone this repository:
    ```bash
    git clone https://github.com/username/SMS-Spam-Detection-using-TensorFlow-in-Python.git
    ```

2. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Open the Jupyter notebook to explore the models:
    ```bash
    jupyter notebook
    ```

## Usage
1. Open the Jupyter notebook file `sms_spam_detection.ipynb`.
2. Run the cells sequentially to:
    - Load and preprocess the dataset.
    - Train the models.
    - Evaluate the models using accuracy, precision, recall, and F1-score.
    - Visualize the results.

3. Example of evaluating a model:
    ```python
    # Evaluate model performance
    model.evaluate(X_test, y_test)
    ```

## Results
The models are evaluated on the following key metrics:

| Model                        | Accuracy | Precision | Recall | F1-Score |
|------------------------------|----------|-----------|--------|----------|
| Multinomial Naive Bayes       | 94%      | 92%       | 95%    | 93%      |
| Custom Vector Embeddings      | 96%      | 94%       | 97%    | 95%      |
| Bidirectional LSTM           | 98%      | 97%       | 98%    | 97%      |
| USE Transfer Learning Model  | 99%      | 98%       | 99%    | 98%      |

## License
MIT License

## Acknowledgements
- Dataset: [SMS Spam Collection](https://archive.ics.uci.edu/ml/datasets/sms+spam+collection)
- [TensorFlow Documentation](https://www.tensorflow.org/)
- [Universal Sentence Encoder](https://tfhub.dev/google/universal-sentence-encoder/4)
