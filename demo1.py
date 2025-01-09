import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
import re
import os

# 1. Đọc dữ liệu từ tệp ly.txt và hoa.txt
dataset_dir = './dataset'  # Đảm bảo rằng thư mục 'dataset' chứa hai tệp 'ly.txt' và 'hoa.txt'

# Đọc nội dung từ các tệp
with open(os.path.join(dataset_dir, 'ly1.txt'), 'r', encoding='utf-8') as file:
    data_ly = file.readlines()

with open(os.path.join(dataset_dir, 'hoa1.txt'), 'r', encoding='utf-8') as file:
    data_hoa = file.readlines()

# Gán nhãn cho dữ liệu
labels_ly = ['lý'] * len(data_ly)
labels_hoa = ['hoá'] * len(data_hoa)

# Kết hợp dữ liệu và nhãn
data = data_ly + data_hoa
labels = labels_ly + labels_hoa

# 2. Tiền xử lý văn bản
def preprocess_text(text):
    text = text.lower()  # Chuyển văn bản thành chữ thường
    text = re.sub(r"[\\.,!?\\-]", " ", text)  # Xóa dấu câu
    text = " ".join(text.split())  # Tách từ bằng khoảng trắng
    return text

# Tiền xử lý dữ liệu
data = [preprocess_text(text) for text in data]

# Tạo DataFrame
df = pd.DataFrame({"text": data, "label": labels})

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
