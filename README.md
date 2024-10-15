[![中文版](https://img.shields.io/badge/READIN-中文-blue.svg)](README_Chinese.md)
[![made-with-Jupyter](https://img.shields.io/badge/Made%20with-Jupyter-orange?style=flat&logo=Jupyter)](https://jupyter.org/)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nlptmu/ClinCaseCipher)

# Project Title
ClinCaseCipher: Privacy Preservation and Standardization in Medical Data using T5-Efficient-BASE-DL2

## Description
The proposed method involves a comprehensive approach to preprocessing, model implementation, and rule-based data extraction to achieve effective de-identification of pathology reports. Preprocessing steps included segmenting the text, generating prompts to extract HIPAA-related information, and normalizing temporal expressions. The processed data was formatted to meet the requirements of the Hugging Face Dataset library, enabling seamless integration with advanced learning algorithms. The outcomes were serialized in JSONL format, and postprocessing steps were employed to refine the extracted information, ensuring compatibility for downstream tasks. For text generation, we utilized the T5-Efficient-BASE-DL2 model [78], an optimized version of the T5 architecture, which offers heightened computational efficiency and reduced memory usage. The model was trained on the preprocessed data using serialized JSONL input-output pairs, with hyperparameters such as a learning rate of 2e-5, a batch size of 4, and 10 training epochs. The evaluation was conducted using the Rouge and Exact Match metrics to assess the model's performance in generating accurate de-identified text. During training, we divided the dataset into training, validation, and test sets. Subsequently, we merged the training and validation sets for final tuning, utilizing a 90-10 split for training and validation, and employed Hugging Faceâ€™s Seq2SeqTrainer to efficiently manage the training process. In addition to the model-based approach, rule-based methods were developed to extract identifiable information such as phone numbers, locations, and city names. These rules included customized regular expressions for local formatting and geographic identifiers, along with a temporal normalization strategy that followed ISO 8601 standards. Temporal expressions, such as "two weeks" or "three months," were systematically converted into structured representations (e.g., "P2W" for two weeks) to ensure consistency and facilitate further analysis. This combination of model-based text generation and rule-based extraction resulted in a robust framework for de-identification, capable of handling both structured and unstructured data effectively.



This project is a comprehensive guide to data preprocessing, model training, and prediction using Jupyter Notebooks. The project is structured into three main parts:

1. `preprocessing.ipynb`: This notebook is used for data preprocessing, preparing the data for model training.
[![Colab](https://img.shields.io/badge/Colab-preprocessing-orange)](https://colab.research.google.com/github/nlptmu/ClinCaseCipher/blob/main/preprocessing.ipynb)
2. `fine_tune_model.ipynb`: This notebook focuses on training the model, fine-tuning it to achieve the desired accuracy.
[![Colab](https://img.shields.io/badge/Colab-fine_tune_model-orange)](https://colab.research.google.com/github/nlptmu/ClinCaseCipher/blob/main/fine_tune_model.ipynb)
3. `predict.ipynb`: In this notebook, the trained model is used for making predictions and includes steps for rule-based and post-processing procedures.
[![Colab](https://img.shields.io/badge/Colab-predict-orange)](https://colab.research.google.com/github/nlptmu/ClinCaseCipher/blob/main/predict.ipynb)

<p align="center">
  <img src="image/ai-cup-Fig1.png" alt="Flowchart"/>
</p>
<h2 align="center">Fig1. Flowchart</h2>

<p align="center">
  <img src="image/ai-cup-Fig2.png" alt="Error Analysis"/>
</p>
<h2 align="center">Fig2. Model generation results-Error Analysis</h2>

## Data Privacy Notice
Due to the sensitive nature of medical data and stringent privacy requirements, the datasets used in this project are not publicly available. The privacy and confidentiality of patient data are our utmost priorities, and we adhere to HIPAA and other relevant regulations to ensure data security. Consequently, the Colab notebooks provided are primarily for demonstration purposes, showcasing the code structure and output formats. They are not executable in their current form as they require access to private datasets.

## Installation

### Prerequisites
- Python 3.x
- CUDA 11.7 for GPU acceleration (optional, but recommended)

### Setup
To set up the project environment, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/nlptmu/ClinCaseCipher
   cd ClinCaseCipher
   ```

2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

### PyTorch and CUDA Compatibility
This project uses PyTorch 2.0.1. For optimal performance, it is recommended to use CUDA 11.7 for GPU acceleration. Ensure that your system has CUDA 11.7 installed.

## Usage
Follow the sequence of notebooks for end-to-end processing:
1. Run `preprocessing.ipynb` for initial data preparation.
2. Use `fine_tune_model.ipynb` for model training.
3. Execute `predict.ipynb` for predictions and post-processing.

## License
This project is licensed under the Apache-2.0 License - see the LICENSE file in the repository for details.
