import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


# Cargar el dataset
data = pd.read_csv('sentimentdataset.csv')

# Mostrar las primeras filas
st.write("### Visualización de los datos")
st.write(data.head())

# Ejemplo de procesamiento de texto
def preprocess_text(text):
    # Convertir a minúsculas
    text = text.lower()
    # Eliminar signos de puntuación
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text

# Crear la columna 'Processed_Text' si no existe
if 'Processed_Text' not in data.columns:
    data['Processed_Text'] = data['Text'].apply(preprocess_text)

# Vectorizar el texto utilizando TF-IDF
tfidf_vectorizer = TfidfVectorizer(max_features=500)
X_tfidf = tfidf_vectorizer.fit_transform(data['Processed_Text'])

# Agrupar las clases de sentimientos
def map_sentiment(sentiment):
    positive_labels = ['Positive', 'Joy', 'Excitement', 'Contentment']
    negative_labels = ['Sadness', 'Anger', 'Regret', 'Sorrow', 'Grief', 'Disgust', 'Fear', 'Disappointment']
    neutral_labels = ['Neutral', 'Acceptance', 'Reflection']

    if sentiment.strip() in positive_labels:
        return 'Positive'
    elif sentiment.strip() in negative_labels:
        return 'Negative'
    else:
        return 'Neutral'

# Aplicar la función de mapeo
data['Grouped_Sentiment'] = data['Sentiment'].apply(map_sentiment)

# Verificar la distribución de las clases agrupadas
st.write("### Distribución de las clases agrupadas")
st.write(data['Grouped_Sentiment'].value_counts())

# Dividir los datos
X_train, X_test, y_train, y_test = train_test_split(X_tfidf, data['Grouped_Sentiment'], test_size=0.2, random_state=42)

# Diccionario para almacenar resultados
model_results = {}

# Función para entrenar, predecir y evaluar el modelo
def evaluate_model(model, model_name):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    confusion_mat = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, zero_division=1)
    
    # Guardar los resultados
    model_results[model_name] = {
        'Accuracy': accuracy,
        'Confusion Matrix': confusion_mat,
        'Classification Report': report
    }
    
    # Mostrar los resultados
    st.write(f'### {model_name} - Accuracy: {accuracy:.2f}')
    st.write('Confusion Matrix:')
    st.write(confusion_mat)
    st.write('Classification Report:')
    st.write(report)

# Evaluar los modelos
st.write("## Evaluación de Modelos")
models = {
    'Decision Tree': DecisionTreeClassifier(),
    'Naive Bayes': MultinomialNB(),
    'K-Nearest Neighbors': KNeighborsClassifier()
}

for model_name, model in models.items():
    evaluate_model(model, model_name)

# Comparar y mostrar el mejor modelo
best_model_name = max(model_results, key=lambda x: model_results[x]['Accuracy'])
st.write(f'## El mejor modelo es: {best_model_name} con una precisión de {model_results[best_model_name]["Accuracy"]:.2f}')

# Visualizar la matriz de confusión del mejor modelo
st.write(f'### Matriz de Confusión para {best_model_name}')
best_confusion_matrix = model_results[best_model_name]['Confusion Matrix']

plt.figure(figsize=(8, 6))
sns.heatmap(best_confusion_matrix, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('Predicción')
plt.ylabel('Real')
st.pyplot(plt)

# Visualización de la distribución de clases
st.write("### Distribución de Clases en el Conjunto de Datos")
class_distribution = {'Positive': 145, 'Negative': 30, 'Neutral': 557}

plt.figure(figsize=(8, 6))
plt.bar(class_distribution.keys(), class_distribution.values(), color=['green', 'red', 'blue'])
plt.title('Distribución de Clases en el Conjunto de Datos')
plt.xlabel('Clase de Sentimiento')
plt.ylabel('Número de Ejemplos')
st.pyplot(plt)
