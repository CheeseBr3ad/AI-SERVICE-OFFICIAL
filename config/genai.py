import google.generativeai as genai
from config.config import GEMINI_API_KEY, GEMINI_AI_MODEL

# Initialize Gemini
genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel(
    GEMINI_AI_MODEL
)  

# from google import genai

# # The client gets the API key from the environment variable `GEMINI_API_KEY`.
# gemini_model = genai.Client()
