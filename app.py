import streamlit as st
import keras
import keras_nlp
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
import os

# Set the Keras backend
os.environ["KERAS_BACKEND"] = "jax"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "1.0"

# Set random seed
keras.utils.set_random_seed(42)

class HealthChatState():
    __START_TURN_USER__ = "<start_of_turn>user\n"
    __START_TURN_MODEL__ = "<start_of_turn>model\n"
    __END_TURN__ = "<end_of_turn>\n"

    def __init__(self, model, diseases, system=""):
        self.model = model
        self.diseases = diseases
        self.system = system if system else """I am a health assistant trained to help understand symptoms and provide general health information.
        I cannot provide medical diagnosis - please consult healthcare professionals for medical advice."""
        self.history = []

    def add_to_history_as_user(self, message):
        self.history.append(self.__START_TURN_USER__ + message + self.__END_TURN__)

    def add_to_history_as_model(self, message):
        self.history.append(self.__START_TURN_MODEL__ + message)

    def get_history(self):
        return "".join([*self.history])

    def get_full_prompt(self):
        prompt = self.get_history() + self.__START_TURN_MODEL__
        if len(self.system) > 0:
            prompt = self.system + "\n" + prompt
        return prompt

    def enhance_response(self, response, message):
        for disease in self.diseases:
            if disease.lower() in message.lower():
                response += f"\n\nNote: {disease} was mentioned. Please consult a healthcare professional for proper medical evaluation and diagnosis."
        return response

    def send_message(self, message):
        self.add_to_history_as_user(message)
        prompt = self.get_full_prompt()
        response = self.model.generate(prompt, max_length=2048)
        result = response.replace(prompt, "")
        enhanced_result = self.enhance_response(result, message)
        self.add_to_history_as_model(enhanced_result)
        return enhanced_result

def load_and_preprocess_data():
    # Load your datasets here
    # For deployment, you might want to store this data in a more accessible format
    # or use a different data source
    kaggle_data_path = "Disease_symptom_and_patient_profile_dataset.csv"
    mendeley_data_path = "symbipredict_2022.csv"

    kaggle_data = pd.read_csv(kaggle_data_path)
    mendeley_data = pd.read_csv(mendeley_data_path)
    combined_data = pd.concat([kaggle_data, mendeley_data], ignore_index=True)

    X = combined_data.drop(columns=["Disease"])
    y = combined_data["Disease"]

    categorical_columns = X.select_dtypes(include=["object"]).columns
    numerical_columns = X.select_dtypes(include=["number"]).columns

    label_encoders = {}
    for col in categorical_columns:
        label_encoders[col] = LabelEncoder()
        X[col] = label_encoders[col].fit_transform(X[col].astype(str))

    imputer = SimpleImputer(strategy="mean")
    X[numerical_columns] = imputer.fit_transform(X[numerical_columns])

    scaler = StandardScaler()
    X[numerical_columns] = scaler.fit_transform(X[numerical_columns])

    disease_encoder = LabelEncoder()
    y_encoded = disease_encoder.fit_transform(y)
    disease_classes = [str(disease) for disease in disease_encoder.classes_]

    return X, y, disease_classes

def initialize_health_chatbot():
    # Load the Gemma model
    gemma_lm = keras_nlp.models.GemmaCausalLM.from_preset("gemma2_instruct_2b_en")

    # Load and preprocess health data
    _, _, diseases = load_and_preprocess_data()

    system_prompt = """I am a health information assistant trained on medical symptoms and conditions.
    I can provide general health information and discuss symptoms, but I cannot diagnose conditions.
    Always consult healthcare professionals for medical advice and diagnosis."""

    chat = HealthChatState(gemma_lm, diseases, system_prompt)
    return chat

def main():
    st.set_page_config(
        page_title="Health Assistant Chatbot",
        page_icon="🏥",
        layout="wide"
    )

    st.title("🏥 Health Assistant Chatbot")
    st.markdown("""
    Welcome to the Health Assistant Chatbot! I can help you understand symptoms and provide general health information.
    Please remember that I cannot provide medical diagnosis - always consult healthcare professionals for medical advice.
    """)

    # Initialize session state for chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Initialize chatbot
    if "chatbot" not in st.session_state:
        st.session_state.chatbot = initialize_health_chatbot()

    # Chat interface
    chat_container = st.container()
    with chat_container:
        for message in st.session_state.chat_history:
            if message["role"] == "user":
                st.markdown(f"🙋‍♂️ **You:** {message['content']}")
            else:
                st.markdown(f"🤖 **Assistant:** {message['content']}")

    # User input
    user_input = st.text_input("Type your message here...", key="user_input")
    if st.button("Send"):
        if user_input:
            # Add user message to chat history
            st.session_state.chat_history.append({
                "role": "user",
                "content": user_input
            })

            # Get bot response
            response = st.session_state.chatbot.send_message(user_input)

            # Add bot response to chat history
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": response
            })

            # Clear input
            st.session_state.user_input = ""
            st.experimental_rerun()

if __name__ == "__main__":
    main()