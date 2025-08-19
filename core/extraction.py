from typing import List, Optional
from core.schema import FieldItem
from openai import OpenAI
import os
from dotenv import load_dotenv
import json

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def extract_fields(
    text: str,
    doc_type: str,
    fields_list: Optional[List[str]] = None,
    model: str = "gpt-4o-mini"
) -> List[FieldItem]:
    if not text.strip():
        return []

    fields_prompt = f"Extract the following fields: {', '.join(fields_list)}." if fields_list else "Extract the most relevant fields for this document type."

    prompt = f"""
You are a document extraction agent.
Document type: {doc_type}
Instructions: {fields_prompt}
Return JSON array of fields with the format:
[{{"name": "...", "value": "...", "confidence": 0.0–1.0, "source": {{"page": 1, "bbox": [0,0,0,0]}}}}]

Text:
{text[:3000]}
"""
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
        )
        raw_output = response.choices[0].message.content.strip()
        extracted_fields = []
        try:
            fields_json = json.loads(raw_output)
            for f in fields_json:
                extracted_fields.append(
                    FieldItem(
                        name=f.get("name", "N/A"),
                        value=f.get("value", "N/A"),
                        confidence=float(f.get("confidence", 0.8)),
                        source=f.get("source", {})
                    )
                )
        except Exception:
            if fields_list:
                for f in fields_list:
                    extracted_fields.append(FieldItem(name=f, value="N/A", confidence=0.5, source={}))
            else:
                extracted_fields.append(FieldItem(name="SampleField", value="SampleValue", confidence=0.7, source={}))
        return extracted_fields
    except Exception as e:
        print(f"[Error] Failed to extract fields: {e}")
        return []
