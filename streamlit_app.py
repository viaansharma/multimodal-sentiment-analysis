
import streamlit as st
import torch
import torch.nn as nn
import torchvision.models as models
from transformers import BertTokenizer, BertModel
from PIL import Image
from torchvision import transforms

# =====================
# Device
# =====================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =====================
# Label Mapping
# =====================
label_map = {
    0: "Negative 😡",
    1: "Neutral 😐",
    2: "Positive 😀"
}

# =====================
# Text Preprocessing
# =====================
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

def preprocess_text(text):
    encoding = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=128,
        return_tensors="pt"
    )
    return encoding["input_ids"], encoding["attention_mask"]

# =====================
# Image Preprocessing
# =====================
image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

def preprocess_image(image):
    image = image.convert("RGB")
    return image_transform(image).unsqueeze(0)

# =====================
# Model Definition
# =====================
class MultiModalSentimentModel(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()

        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.resnet = models.resnet50(pretrained=True)
        self.resnet.fc = nn.Identity()

        self.fc1 = nn.Linear(768 + 2048, 512)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, input_ids, attention_mask, images):
        text_features = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        ).pooler_output

        image_features = self.resnet(images)

        fused = torch.cat((text_features, image_features), dim=1)
        x = self.fc1(fused)
        x = self.relu(x)
        x = self.dropout(x)
        return self.fc2(x)

# =====================
# Load Trained Model
# =====================
model = MultiModalSentimentModel().to(device)
model.load_state_dict(torch.load("multimodal_model.pth", map_location=device))
model.eval()

# =====================
# Streamlit UI
# =====================
st.title("🧠 Multi-modal Sentiment Analysis")
st.write("Sentiment prediction using **Text + Image**")

text_input = st.text_area("Enter post text")
image_file = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"])

if st.button("Predict Sentiment"):
    if text_input and image_file:
        image = Image.open(image_file)

        input_ids, attention_mask = preprocess_text(text_input)
        image_tensor = preprocess_image(image)

        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        image_tensor = image_tensor.to(device)

        with torch.no_grad():
            outputs = model(input_ids, attention_mask, image_tensor)
            prediction = torch.argmax(outputs, dim=1).item()

        st.image(image, caption="Uploaded Image", width=300)
        st.success(f"Predicted Sentiment: **{label_map[prediction]}**")
    else:
        st.warning("Please provide both text and image.")
