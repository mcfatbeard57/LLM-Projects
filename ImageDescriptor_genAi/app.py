from dotenv import load_dotenv
load_dotenv() ## load all the environment variables from .env

import streamlit as st
import os
from PIL import Image
import google.generativeai as genai

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def get_gemini_response(input,image,user_prompt):
    ## Load Gemini pro vision model
    model=genai.GenerativeModel('gemini-pro-vision')
    response=model.generate_content([input,image[0],user_prompt])
    return response.text

def input_image_details(uploaded_file):
    if uploaded_file is not None:
        # Read the file into bytes
        bytes_data = uploaded_file.getvalue()
        image_parts = [
            {"mime_type": uploaded_file.type,  # Get the mime type of the uploaded file
                "data": bytes_data}
        ]
        return image_parts
    else:
        raise FileNotFoundError("No file uploaded")

##initialize our streamlit app
st.set_page_config(page_title="Image Descriptor")

st.header("Image Descriptor")
input=st.text_input("Prompt. Use this to provide context for the image or the output ",key="input")
uploaded_file = st.file_uploader("Choose any image you like:", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image=Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image.", use_column_width=True)

submit=st.button("Tell me about the image")

## if submit button is clicked

if submit:
    image_data=input_image_details(uploaded_file)
    response=get_gemini_response(input,image_data,input)
    st.subheader("The Response is")
    st.write(response)