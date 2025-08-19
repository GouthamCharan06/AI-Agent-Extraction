from typing import List, Optional
from dotenv import load_dotenv
import os
import json

# LangChain
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import HumanMessage

# Pydantic schema
from pydantic import BaseModel, Field, ValidationError
from typing import List as TList, Optional as TOptional

load_dotenv()


# --------------------------
# Pydantic Schema Definition
# --------------------------

class Source(BaseModel):
    page: TOptional[int] = None
    bbox: TList[float] = Field(default_factory=list)

class FieldItem(BaseModel):
    name: str
    value: TOptional[str] = None
    confidence: float
    source: Source

class QASection(BaseModel):
    passed_rules: TList[str]
    failed_rules: TList[str]
    notes: str

class ExtractionOutput(BaseModel):
    doc_type: str
    fields: TList[FieldItem]
    overall_confidence: float
    qa: QASection


# --------------------------
# Extraction Function
# --------------------------

def extract_fields(
    text: str,
    doc_type: Optional[str] = None,
    fields_list: Optional[List[str]] = None,
    model: str = "gpt-4o-mini",
    max_retries: int = 2
) -> ExtractionOutput:

    if not text.strip():
        return ExtractionOutput(
            doc_type="unknown",
            fields=[],
            overall_confidence=0.0,
            qa=QASection(
                passed_rules=[],
                failed_rules=["empty_text"],
                notes="No text provided"
            )
        )

    llm = ChatOpenAI(model_name=model, temperature=0)

    # --------------------------
    # Auto-detect doc_type
    # --------------------------
    if not doc_type:
        detect_prompt = f"""
You are a document classifier. 
Identify the document type from the following text. 
Examples: invoice, resume, fee receipt, prescription, medical bill, certificate, report, etc.
Return ONLY a single word or short phrase without explanation.

Text:
{text[:2000]}
"""
        detect_response = llm([HumanMessage(content=detect_prompt)])
        doc_type = detect_response.content.strip().lower()
        print(f"[Info] Auto-detected doc_type: {doc_type}")

    # Instruction text
    if fields_list:
        instruction_text = f"Extract the following fields: {', '.join(fields_list)}."
    else:
        instruction_text = "Extract all relevant key-value fields for this document type."

    # Prompt template (escape braces for f-string safety)
    prompt_template = f"""
You are an expert document extraction assistant.

Instructions:
1. Read the entire document text carefully.
2. Determine the document type (invoice, medical bill, prescription, certificate, resume, etc.).
3. Extract the most important fields relevant to the identified document type.
4. Return a JSON object exactly matching this structure (no code fences or markdown):

{{
  "doc_type": "{doc_type}",
  "fields": [
    {{
      "name": "<FieldItem.name>",
      "value": "<FieldItem.value>",
      "confidence": 0.0,
      "source": {{"page": 1, "bbox": [0,0,0,0]}}
    }}
  ],
  "overall_confidence": 0.0,
  "qa": {{"passed_rules": [], "failed_rules": [], "notes": ""}}
}}

Rules:
- Only include fields present in the text.
- Confidence reflects certainty of each extracted value (0.0–1.0).
- Do not invent any information not present in the document.
- Accurately extract tables, totals, dates, amounts, or IDs if present.
- Keep JSON valid and parsable.
- {instruction_text}

Text to process:
{text[:10000]}
"""

    def clean_response(raw_response: str) -> str:
        """Remove code fences and cleanup JSON text"""
        raw_response = raw_response.strip()
        if raw_response.startswith("```"):
            raw_response = raw_response.strip("`")
            raw_response = raw_response.replace("json", "").strip()
        return raw_response

    # --------------------------
    # Main extraction loop
    # --------------------------
    for attempt in range(max_retries):
        try:
            # Build chat messages
            chat_prompt = ChatPromptTemplate.from_template(prompt_template)
            messages = chat_prompt.format_messages()

            # Call the LLM
            response_obj = llm(messages)

            # Extract text
            if hasattr(response_obj, "content"):
                raw_response = response_obj.content
            elif hasattr(response_obj, "choices"):
                raw_response = response_obj.choices[0].message.content
            else:
                raw_response = str(response_obj)

            raw_response = clean_response(raw_response)
            print("[Debug] LLM raw output:", raw_response)

            # Try parsing JSON
            try:
                parsed_output = json.loads(raw_response)
            except json.JSONDecodeError:
                # Retry with "fix my JSON" step
                fix_prompt = f"""
The previous response was not valid JSON or did not match schema.
Fix it and return ONLY valid JSON in this exact format:

{{
  "doc_type": "<string>",
  "fields": [
    {{"name":"<string>","value":"<string|null>","confidence":0.0,"source":{{"page":1,"bbox":[0,0,0,0]}}}}
  ],
  "overall_confidence": 0.0,
  "qa": {{"passed_rules": [], "failed_rules": [], "notes": ""}}
}}

Broken JSON to fix:
{raw_response}
"""
                fix_response = llm([HumanMessage(content=fix_prompt)])
                raw_fix = clean_response(
                    getattr(fix_response, "content", str(fix_response))
                )
                parsed_output = json.loads(raw_fix)

            # Validate with Pydantic
            validated = ExtractionOutput.parse_obj(parsed_output)
            return validated

        except (ValidationError, Exception) as e:
            print(f"[Warning] Extraction attempt {attempt + 1} failed: {e}")

    # --------------------------
    # Smart Fallback
    # --------------------------
    print("[Info] Falling back to simple key-value extraction...")

    fallback_prompt = f"""
Extract as many field name/value pairs as possible from this text. 
Do not invent information. 
Return ONLY JSON in this format:

[
  {{"name": "<field_name>", "value": "<field_value>"}}
]

Text:
{text[:3000]}
"""

    fb_response = llm([HumanMessage(content=fallback_prompt)])

    fb_raw = clean_response(getattr(fb_response, "content", str(fb_response)))
    try:
        fb_pairs = json.loads(fb_raw)
    except Exception:
        fb_pairs = [{"name": "SampleField", "value": "SampleValue"}]

    fb_fields = [
        FieldItem(
            name=p.get("name", "N/A"),
            value=p.get("value", "N/A"),
            confidence=0.5,
            source=Source(page=None, bbox=[])
        )
        for p in fb_pairs
    ]

    return ExtractionOutput(
        doc_type=doc_type if doc_type else "unknown",
        fields=fb_fields,
        overall_confidence=0.5,
        qa=QASection(
            passed_rules=[],
            failed_rules=["schema_not_matched"],
            notes="Used fallback simple extraction"
        )
    )
