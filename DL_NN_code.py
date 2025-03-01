import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping

df = pd.read_csv('IMDB_Dataset.csv')

print(df.head())
print(df.info())

stop_words = set(stopwords.words('english'))
# preprocessing function 
def clean_text(text):
    text = text.lower() # Convert to lowercase
    text = re.sub('<.*?>', '', text)  # Remove HTML tags
    text = re.sub('http\S+|www\S+', '', text)  # Remove URLs
    text = re.sub('[^a-z\s]', '', text)
    tokens = text.split() 
    tokens = [word for word in tokens if word not in stop_words]  # Remove stopwords

    return ' '.join(tokens)

df['cleaned review'] = df['review'].apply(clean_text)

# TF-IDF : converting the data into suitable format for neural network
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df['cleaned review']).toarray()
y = df['sentiment'].map({'positive': 1, 'negative': 0})

#trainning the model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#model building

model = Sequential([
    Dense(256, activation = 'relu', input_shape=(5000,),kernel_regularizer=l2(0.01)),
    Dropout(0.4),
    Dense(128, activation = 'relu',kernel_regularizer=l2(0.01)),
    Dropout(0.3),
    Dense(64, activation = 'relu',kernel_regularizer=l2(0.01)),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer=Adam(learning_rate=0.0005), loss='binary_crossentropy', metrics=['accuracy'])
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

#model training
history = model.fit(X_train, y_train, epochs=15, batch_size=64, validation_data=(X_test, y_test), callbacks=[early_stopping])
test_loss, test_accuracy = model.evaluate(X_test, y_test)

print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")


#plots
# # Sentiment Distribution
# plt.figure(figsize=(10, 6))
# sns.countplot(x='sentiment', data=df)
# plt.title('Sentiment Distribution')
# plt.xlabel('Sentiment')
# plt.ylabel('Count')
# plt.show()

# ploting accuracy and loss over epochs
plt.figure(figsize=(10, 6))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Training and Validation Accuracy')
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Validation Loss')