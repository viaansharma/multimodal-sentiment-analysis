🧠 Multi-Modal Sentiment Analysis (Text + Image)

An end-to-end Multi-Modal Sentiment Analysis system that predicts sentiment by jointly analyzing text and image data from social media posts.

The project uses Transformer-based text encoding (BERT) and CNN-based image feature extraction (ResNet-50), followed by a fusion-based classifier and a Streamlit web interface for interactive inference.

🚀 Key Features

📝 Text Feature Extraction using BERT (Hugging Face Transformers)

🖼️ Image Feature Extraction using ResNet-50 (ImageNet pretrained)

🔗 Late Fusion of text and image embeddings

⚖️ Class-weighted loss to handle class imbalance

🔧 Selective fine-tuning of the last BERT layer

📊 Strong evaluation performance (Macro-F1 = 0.74)

🌐 Streamlit web app for real-time predictions

☁️ Works offline (full inference) and online (demo mode)

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
│
├── MVSA_Single/
│   ├── data/                (not uploaded)
│   ├── labelResultsAll.txt

📊 Dataset

Dataset: MVSA-Single (public multimodal sentiment dataset)

Modalities: Text + Image

Classes: Negative, Neutral, Positive

Total Samples: 4,869

⚠️ Due to size and licensing constraints, the dataset images are not included in this repository.

🏗️ Model Architecture
🔹 Text Encoder

Model: BERT-base-uncased

Output: 768-dimensional embedding

🔹 Image Encoder

Model: ResNet-50 (pretrained on ImageNet)

Output: 2048-dimensional embedding

🔹 Fusion & Classifier

Late fusion via concatenation

Fully connected layers with ReLU and Dropout

3-class sentiment classification

⚙️ Training Strategy

Initial training with frozen encoders for stability

Class-weighted CrossEntropyLoss to address imbalance

Selective fine-tuning of the last BERT encoder layer

Optimizer: AdamW

This strategy significantly improved minority-class performance.

📈 Results
Final Performance (Validation Set)
Metric	Score
Accuracy	0.74
Macro F1-Score	0.74
Confusion Matrix
[[186,  39,  19],
 [ 65, 274,  45],
 [ 36,  47, 263]]


The model achieves balanced performance across all sentiment classes.

🌐 Streamlit Web App
▶️ Run Locally (Full Inference)
streamlit run streamlit_app.py


Make sure multimodal_model.pth is present in the project root.

☁️ Online Deployment (Demo Mode)

The Streamlit app is designed to run safely online even when model weights are not included.

If multimodal_model.pth is missing:

The UI loads

A clear message explains how to run full inference locally

No runtime crash occurs

This follows best practices for ML deployment.

🧠 Key Learnings

Multi-modal fusion improves sentiment understanding over unimodal approaches

Class imbalance must be explicitly handled

Partial fine-tuning offers strong gains with minimal overfitting

Notebook-based training and production deployment require careful separation

🚀 Future Improvements

Attention-based fusion mechanisms

Multimodal Transformers (e.g., ViLBERT, CLIP)

Additional datasets (MVSA-Multiple)

Probability calibration for confidence estimation

👨‍💻 Author

Viaan Sharma
M.Tech – Mathematics & Computing (Machine Learning)
National Institute of Technology Delhi

