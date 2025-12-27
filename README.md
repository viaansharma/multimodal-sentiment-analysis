# 🧠 Multi-Modal Sentiment Analysis (Text + Image)

An end-to-end **Multi-Modal Sentiment Analysis system** that predicts sentiment by jointly analyzing **text and image data** from social media posts.

The project uses **Transformer-based text encoding (BERT)** and **CNN-based image feature extraction (ResNet-50)**, followed by a fusion-based classifier and a **Streamlit web interface** for interactive inference.

---

## 🚀 Key Features

- 📝 **Text Feature Extraction** using **BERT (Hugging Face Transformers)**
- 🖼️ **Image Feature Extraction** using **ResNet-50 (ImageNet pretrained)**
- 🔗 **Late fusion** of text and image embeddings
- ⚖️ **Class-weighted Loss** to handle class imbalance
- 🔧 **Selective fine-tuning** of the last BERT layer
- 📊 Strong evaluation performance (**Macro-F1 = 0.74**)
- 🌐 **Streamlit web app** for real-time predictions
- ☁️ Runs offline (full inference) and online (demo mode)

---

## 📂 Project Structure

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
│ ├── data/ (downloaded from Kaggle)
│ ├── labelResultsAll.txt (Kaggle labels)



---

## 📊 Dataset Source

This project uses the **MVSA-Single** (multi-view sentiment analysis) dataset from Kaggle:

👉 **Download here:**  
https://www.kaggle.com/datasets/vincemarcs/mvsasingle?utm_source=chatgpt.com

### How to use it

After downloading:
1. Extract the dataset
2. Place the folder as:

multimodal-sentiment/MVSA_Single/
├── data/
├── labelResultsAll.txt

The code expects:
- Text files: `*.txt` for each post
- Images: `*.jpg` in `data/`
- Labels: `labelResultsAll.txt` containing text–image sentiment annotations

---

## 🏗️ Model Architecture Summary

### Text Encoder
- Model: `BERT-base-uncased`
- Embedding size: 768

### Image Encoder
- Model: `ResNet-50`
- Embedding size: 2048

### Fusion + Classifier
- Concatenate text + image features
- Fully connected layers with dropout for classification

---

## ⚙️ Training Strategy

- Class-weighted Cross-Entropy to handle imbalance
- Partial fine-tuning (only last BERT layer)
- Optimizer: **AdamW**

---

## 📈 Results

### Performance on Validation Set

| Metric | Score |
|--------|-------|
| **Accuracy**     | 0.74 |
| **Macro F1-Score** | 0.74 |

### Confusion Matrix

[[186, 39, 19],
[ 65, 274, 45],
[ 36, 47, 263]]

## Model Hosting

The trained model (~538 MB) is hosted on the Hugging Face Model Hub to avoid GitHub file size limits.
The Streamlit app automatically downloads the model at runtime.

Model link:
https://huggingface.co/viaan7/multimodal-sentiment-bert-resnet


This shows balanced performance across all sentiment classes.

---

## 🌐 Streamlit App (Offline + Online)

### ▶️ Offline (Local Machine)

To run with full model inference:

1. Download the dataset
2. Place it under `MVSA_Single/`
3. Make sure `multimodal_model.pth` exists in project root
4. Run:

```bash
streamlit run streamlit_app.py
☁️ Online (Demo Mode)
For GitHub / Streamlit Cloud deployments, the model weights file is not included due to size.
In this case, the app:

✔ Loads UI
✔ Shows a clear warning that model weights are missing
✔ Does not crash

To run full inference, users must place the weights locally.

🧠 Notes for Users
You must download the dataset manually from Kaggle

Required project files assume dataset structure as shown above

Sending model weights over the web is optional, but locally supported

📌 One-Line Summary (Good for BIOS/Portfolio)
End-to-end multi-modal sentiment analysis using BERT + ResNet-50 with class-weighted training and Streamlit deployment.

👨‍💻 Contact / Author
Viaan Sharma
M.Tech – Mathematics & Computing (Machine Learning)
National Institute of Technology Delhi
