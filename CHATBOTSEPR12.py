import os
import pandas as pd
import spacy
from sentence_transformers import SentenceTransformer
import numpy as np
import re
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st


# Chargement du modèle spaCy pour le français
@st.cache_resource
def load_spacy_model():
    return spacy.load("fr_core_news_sm")


# Chargement du modèle SentenceTransformer
@st.cache_resource
def load_sentence_transformer_model():
    return SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')


nlp = load_spacy_model()
model = load_sentence_transformer_model()


# Fonction pour charger les données FAQ
@st.cache_data
def load_faq_data(file_path):
    if not os.path.isfile(file_path):
        st.error(f"Erreur : le fichier '{file_path}' n'existe pas.")
        return None

    try:
        faq_data = pd.read_csv(file_path, sep=';', encoding='latin-1', on_bad_lines='warn')
        faq_data.columns = faq_data.columns.str.strip()  # Nettoyer les noms de colonnes
        required_columns = {'Questions', 'Answers'}
        if not required_columns.issubset(faq_data.columns):
            st.error(f"Erreur : le fichier doit contenir les colonnes {required_columns}.")
            return None

    except Exception as e:
        st.error(f"Erreur lors du chargement du fichier FAQ : {e}")
        return None

    st.success("Bonjour et bienvenue ! Quelles informations recherchez-vous ?")
    return faq_data


# Fonction pour prétraiter une question
def preprocess_question(question):
    doc = nlp(question.lower())
    corrected = " ".join(token.lemma_ for token in doc if not token.is_stop)
    corrected = re.sub(r'\W+', ' ', corrected)  # Retirer la ponctuation
    return corrected.strip()


# Fonction pour calculer les embeddings des questions FAQ
@st.cache_data
def calculate_faq_embeddings(faq_data):
    return model.encode(faq_data['Questions'].tolist(), convert_to_tensor=True)


# Fonction pour trouver la question la plus similaire
def find_similar_question(question, faq_embeddings):
    question_embedding = model.encode([question], convert_to_tensor=True)
    similarities = cosine_similarity(question_embedding, faq_embeddings)
    top_index = np.argmax(similarities)
    return top_index, similarities[0][top_index]


# Fonction pour raffiner la réponse
def refine_response(index, similarity_score, faq_data, threshold=0.7):
    if similarity_score > threshold:
        return faq_data['Answers'].iloc[index]
    return "Désolé, je n'ai pas trouvé de réponse à votre question."


# Fonction principale du pipeline
def chatbot_pipeline(question, faq_data, faq_embeddings):
    processed_question = preprocess_question(question)
    index, similarity_score = find_similar_question(processed_question, faq_embeddings)
    return refine_response(index, similarity_score, faq_data)


# Interface Streamlit
def main():
    # CSS pour le fond
    css_code = """
    <style>
    .stApp {
        background-image: url("C:/Users/user/PycharmProjects/pythonProject22/LOGO2.png");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        height: 100vh;
    }
    </style>
    """
    st.markdown(css_code, unsafe_allow_html=True)

    col1, col2 = st.columns([1, 4])
    with col1:
        logo_path = r"C:\Users\user\PycharmProjects\pythonProject22\LOGO SEPR.png"
        st.image(logo_path, use_column_width=True)

    with col2:
        st.title("Chatbot de la Société d'Environnement et de Plantation de Redeyef SEPR")
        file_path = r"C:\Users\user\PycharmProjects\pythonProject22\CHATBOTSEPR12.csv"

        # Chargement des données FAQ
        faq_data = load_faq_data(file_path)
        if faq_data is not None:
            faq_data['Questions'] = faq_data['Questions'].fillna('')
            faq_embeddings = calculate_faq_embeddings(faq_data)


            user_question = st.text_input("                    ")

            if st.button("Obtenir une réponse"):
                if user_question:
                    response = chatbot_pipeline(user_question, faq_data, faq_embeddings)

                    # Affichage de l'image du chatbot avec gestion d'erreur
                    chatbot_image_path = r"C:\Users\user\PycharmProjects\pythonProject22\IMGCH_1804.jpg"
                    try:
                        st.image(chatbot_image_path, caption="Chatbot SEPR", width=80)
                    except Exception as e:
                        st.error(f"Erreur lors du chargement de l'image : {e}")

                    # Affichage de la réponse sous l'image
                    st.write("Chatbot SEPR:", response)

                    # Enregistrer la conversation dans un fichier
                    with open("conversations.txt", "a", encoding="utf-8") as f:
                        f.write(f"Vous: {user_question}\nChatbot: {response}\n\n")

            if st.checkbox("Afficher l'historique"):
                if os.path.exists("conversations.txt"):
                    with open("conversations.txt", "r", encoding="utf-8") as f:
                        st.text_area("Historique", f.read(), height=300)
                else:
                    st.write("Aucune conversation enregistrée.")


if __name__ == "__main__":
    main()