🧠 Multi-Modal Sentiment Analysis (Text + Image)

An end-to-end Multi-Modal Sentiment Analysis system that predicts sentiment by jointly analyzing text and image data from social media posts.

The project uses Transformer-based text encoding (BERT) and CNN-based image feature extraction (ResNet-50), followed by a fusion-based classifier and a Streamlit web interface for real-time inference.

🚀 Key Features

📝 Text Feature Extraction using BERT (Hugging Face Transformers)

🖼️ Image Feature Extraction using ResNet-50 (ImageNet pretrained)

🔗 Late fusion of text and image embeddings

⚖️ Class-weighted loss to handle class imbalance

🔧 Selective fine-tuning of the last BERT layer

📊 Strong evaluation performance (Macro-F1 = 0.74)

🌐 Streamlit web app for real-time predictions

☁️ Fully functional online and offline deployment

📂 Project Structure
multimodal-sentiment/
│
├── 01_dataset_loader.ipynb
├── 02_text_preprocess.ipynb
├── 03_image_preprocess.ipynb
├── 04_multimodal_model.ipynb
├── 05_train.ipynb
├── 06_evaluate.ipynb
├── 07_streamlit_app.ipynb
│
├── streamlit_app.py
├── requirements.txt
├── README.md
├── report.pdf
│
├── MVSA_Single/
│   ├── data/                (downloaded from Kaggle)
│   ├── labelResultsAll.txt

📊 Dataset Source

This project uses the MVSA-Single (Multi-View Sentiment Analysis) dataset.

👉 Dataset link (Kaggle):
https://www.kaggle.com/datasets/vincemarcs/mvsasingle

Dataset Setup

After downloading:

multimodal-sentiment/MVSA_Single/
├── data/
├── labelResultsAll.txt


Expected format:

Text files: *.txt

Images: *.jpg inside data/

Labels: labelResultsAll.txt

🏗️ Model Architecture
🔹 Text Encoder

Model: BERT-base-uncased

Embedding Size: 768

🔹 Image Encoder

Model: ResNet-50

Embedding Size: 2048

🔹 Fusion & Classifier

Late fusion via concatenation

Fully connected layers with ReLU and Dropout

3-class sentiment classification

⚙️ Training Strategy

Class-weighted CrossEntropyLoss to address imbalance

Partial fine-tuning (last BERT encoder layer)

Optimizer: AdamW

📈 Results
Performance on Validation Set
Metric	Score
Accuracy	0.74
Macro F1-Score	0.74
Confusion Matrix
[[186,  39,  19],
 [ 65, 274,  45],
 [ 36,  47, 263]]


The model achieves balanced performance across all sentiment classes.

🧠 Model Hosting

The trained model (~538 MB) is hosted on the Hugging Face Model Hub to avoid GitHub file size limits.

🔗 Model Link:
https://huggingface.co/viaan7/multimodal-sentiment-bert-resnet

The Streamlit application automatically downloads the model at runtime, enabling full online inference.

🌐 Live Streamlit App

🔴 Live Demo:
https://multimodal-sentiment-analysis-e4mqzethdappjlh85qpx7bt.streamlit.app

▶️ Run Locally
streamlit run streamlit_app.py


The model will be downloaded automatically from Hugging Face Hub.

🧠 Notes

Dataset must be downloaded manually from Kaggle

Model weights are hosted externally for scalability

The same codebase supports local and cloud deployment

📌 One-Line Summary

End-to-end multi-modal sentiment analysis using BERT and ResNet-50 with class-weighted training and Streamlit deployment.

👨‍💻 Author

Viaan Sharma
M.Tech – Mathematics & Computing (Machine Learning)
National Institute of Technology Delhi
