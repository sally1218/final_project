from flask import Flask, request, jsonify, render_template
from PIL import Image
import torch
from torchvision import transforms
import os
import requests

app = Flask(__name__)

# 模型檔案名稱與 Google Drive 檔案 ID
MODEL_PATH = "fruit_classifier.pt"
GDRIVE_FILE_ID = "1zURwAkDac3ybnftoRh79GXNIgHjv9EFA"

# 自動下載模型（從 Google Drive）
def download_model():
    if not os.path.exists(MODEL_PATH):
        print("模型不存在，開始從 Google Drive 下載...")
        url = f"https://drive.google.com/uc?export=download&id={GDRIVE_FILE_ID}"
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            with open(MODEL_PATH, "wb") as f:
                for chunk in response.iter_content(1024):
                    f.write(chunk)
            print("模型下載完成！")
        else:
            print("模型下載失敗！請確認連結是否正確。")

# 下載模型
download_model()

# 載入模型
model = torch.jit.load(MODEL_PATH)
model.eval()

# 水果分類類別
class_names = ['apple', 'avocado', 'banana', 'cherry', 'kiwi', 'mango', 'orange', 'pineapple', 'strawberries', 'watermelon']

# 圖片轉換
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

# 首頁
@app.route("/")
def index():
    return render_template("fruit.html")

# 預測 API
@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    img = Image.open(file.stream).convert("RGB")
    img = transform(img).unsqueeze(0)

    with torch.no_grad():
        outputs = model(img)
        _, predicted = torch.max(outputs, 1)
        label = class_names[predicted.item()]

    return jsonify({"prediction": label})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
