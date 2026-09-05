import os

from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()


def test_gemini_connection():

    assert os.getenv("GOOGLE_API_KEY")

    model = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0,
    )

    response = model.invoke(
        "Reply with exactly: Gemini connection works."
    )

    assert response.text