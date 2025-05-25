from flask import Flask, request, jsonify, render_template
from PIL import Image
import torch
from torchvision import transforms

app = Flask(__name__)

# 載入模型
model = torch.jit.load("fruit_classifier.pt")
model.eval()

# 水果分類類別
class_names = ['apple', 'avocado', 'banana', 'cherry', 'kiwi', 'mango', 'orange', 'pineapple', 'strawberries', 'watermelon']

# 圖片轉換
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

# ✅ 新增這段：讓首頁顯示 fruit.html 頁面
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