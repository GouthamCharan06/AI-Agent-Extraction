import os
from dotenv import load_dotenv
from openai import OpenAI
import json

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

FEW_SHOT_EXAMPLES = """
Example 1:
Text: Payment receipt for tuition fees of student John Doe.
Document Type: fee_receipt

Example 2:
Text: Invoice for medical services rendered to patient Jane Smith.
Document Type: invoice

Example 3:
Text: Prescription for patient Alice Brown.
Document Type: prescription
"""

def classify_document(text: str, model="gpt-4o-mini") -> dict:
    if not text.strip():
        return {"doc_type": "unknown", "confidence": 0.0}

    prompt = f"""
You are a document classification agent.
Identify the type of document from the following text.
Possible types: invoice, medical_bill, prescription, resume, ID card, report, fee_receipt, others.
Return JSON ONLY in the format: {{"doc_type": "...", "confidence": 0.0–1.0}}

{FEW_SHOT_EXAMPLES}

Text to classify:
{text[:2000]}
"""
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
        )
        raw_output = response.choices[0].message.content.strip()
        try:
            doc_info = json.loads(raw_output)
            return {
                "doc_type": doc_info.get("doc_type", "others"),
                "confidence": float(doc_info.get("confidence", 0.9))
            }
        except Exception:
            return {"doc_type": raw_output, "confidence": 0.9}
    except Exception as e:
        print(f"[Error] Failed to classify document: {e}")
        return {"doc_type": "unknown", "confidence": 0.0}
