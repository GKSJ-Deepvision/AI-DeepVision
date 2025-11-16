
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from PIL import Image
from tqdm import tqdm

csv_path = r"C:\Users\ADMIN\Downloads\dataset_extracted\crowds_counting.csv"
image_base_path = r"C:\Users\ADMIN\Downloads\dataset_extracted\images"


df = pd.read_csv(csv_path)
print("✅ Dataset Loaded Successfully")
print(df.head())


print("\nMissing Values:")
print(df.isnull().sum())

print("\nDuplicate Rows:", df.duplicated().sum())

df.drop_duplicates(inplace=True)

label_encoder = LabelEncoder()
df['label_encoded'] = label_encoder.fit_transform(df['label'])
df['type_encoded'] = label_encoder.fit_transform(df['type'])

print("\nLabel Encoding Done:")
print(df[['label', 'label_encoded', 'type', 'type_encoded']].head())

import cv2

image_data = []
labels = []

for i in tqdm(range(len(df))):
    img_rel_path = df.iloc[i]['image']
    label = df.iloc[i]['label_encoded']
    
    
    img_path = os.path.join(r"C:\Users\ADMIN\Downloads\dataset_extracted", img_rel_path)
    
    if os.path.exists(img_path):
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (128, 128))  
        image_data.append(img)
        labels.append(label)
    else:
        print(f"⚠️ Image not found: {img_rel_path}")


X = np.array(image_data)
y = np.array(labels)

print("\n✅ Image Preprocessing Complete")
print("Image Array Shape:", X.shape)
print("Labels Shape:", y.shape)

X = X / 255.0  # Scale to [0,1] range

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\n✅ Dataset Split Complete")
print("Train set:", X_train.shape)
print("Test set:", X_test.shape)
