from dotenv import load_dotenv
load_dotenv()  # this loads values from .env into os.environ


import os
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def detect_doc_type(text: str) -> str:
    """Classify the document type using OpenAI LLM."""
    prompt = f"""
    You are a document classification agent.
    Given the following text, identify what type of document it is 
    (e.g., invoice, prescription, medical bill, resume, ID card, report).

    Text:
    {text[:1000]}
    """

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
    )

    return response.choices[0].message.content.strip()
