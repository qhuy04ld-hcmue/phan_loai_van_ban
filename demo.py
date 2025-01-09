import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
import re

# 1. Chuẩn bị dữ liệu mẫu (giả lập dữ liệu)
data_ly = ["Định luật Newton", "Công thức tính vận tốc", "Dòng điện trong vật dẫn", "Lực đàn hồi", "Điện từ trường"] * 100
data_hoa = ["Phản ứng oxi hóa khử", "Cấu trúc nguyên tử", "Liên kết hóa học", "Axit và bazơ", "Phương trình hóa học"] * 100

labels_ly = ["ly"] * len(data_ly)
labels_hoa = ["hoa"] * len(data_hoa)

# Kết hợp dữ liệu
data = data_ly + data_hoa
labels = labels_ly + labels_hoa

# Tạo DataFrame
df = pd.DataFrame({"text": data, "label": labels})

# 2. Tiền xử lý văn bản
def preprocess_text(text):
    text = text.lower()  # Chuyển văn bản thành chữ thường
    text = re.sub(r"[\\.,!?\\-]", " ", text)  # Xóa dấu câu
    text = " ".join(text.split())  # Tách từ bằng khoảng trắng
    return text

df["text"] = df["text"].apply(preprocess_text)

# 3. Vector hóa văn bản với TF-IDF
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df["text"])
y = df["label"]

# 4. Chia dữ liệu thành tập train và test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Huấn luyện mô hình
model = MultinomialNB()
model.fit(X_train, y_train)

# 6. Đánh giá mô hình
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# 7. Dự đoán từ đầu vào người dùng
while True:
    user_input = input("Nhập nội dung văn bản (hoặc gõ 'exit' để thoát): ")
    if user_input.lower() == 'exit':
        break
    processed_input = preprocess_text(user_input)
    vectorized_input = vectorizer.transform([processed_input])
    prediction = model.predict(vectorized_input)
    print(f"Môn học dự đoán: {prediction[0]}")
