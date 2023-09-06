import streamlit as st
from fastai.vision.all import load_learner, PILImage

# Load your trained model
model = load_learner('cat_model.pkl')

st.title("Cat Breed Classifier")

# Upload image through streamlit
uploaded_file = st.file_uploader("Choose a cat image...", type="jpg")

if uploaded_file is not None:
    image = PILImage.create(uploaded_file)
    st.image(image, caption='Uploaded Cat Image.', use_column_width=True)
    st.write("")
    st.write("Classifying...")
    
    # Predict the breed
    pred,pred_idx,probs = model.predict(image)
    st.write(f"Prediction: {pred}; Probability: {probs[pred_idx]:.04f}")