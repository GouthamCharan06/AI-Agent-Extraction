AI Document Extraction Agent
-----------------------------
This project extracts structured information from unstructured documents (invoices, resumes, prescriptions, etc.) using OCR and AI models. It identifies the document type, extracts key fields, validates them, and returns results in JSON format with confidence scores.

How It Works:

1.Upload a document (PDF or image).
2.OCR + parsing – Text is extracted using PyMuPDF and EasyOCR.
3.Classification – The document type is identified with an OpenAI model.
4.Field extraction – Relevant fields are pulled and validated.
5.Output – Results are shown in the Streamlit UI and can be downloaded (JSON format).

Tech Stack:

1.Python
2.Streamlit (UI)
3.OpenAI API (classification & extraction) + LangChain (LLM orchestration & parsing)
4.PyMuPDF, EasyOCR (text extraction)
5.Pydantic (validation & confidence scoring)

Run Locally:
1.Clone the repository:
git clone <your-repo-url>
cd ai-agent-extraction

2.Install dependencies:
pip install -r requirements.txt

3.Add your OpenAI API Key in .streamlit/secrets.toml:
OPENAI_API_KEY="your_api_key_here"

4.Start the app:
streamlit run app/main.py




