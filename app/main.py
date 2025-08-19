import sys
from pathlib import Path
import json
from io import BytesIO
import pandas as pd
import streamlit as st

# Add project root to Python path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from core.ingestion import extract_text_from_file
from core.routing import classify_document
from core.extraction import extract_fields
from core.validation import validate_fields
from core.confidence import update_document_schema
from core.schema import DocumentSchema

# Streamlit page config
st.set_page_config(page_title="AI Document Extraction Agent", layout="wide")
st.title("AI Document Extraction Agent")
st.write("Upload a PDF or Image to extract structured data.")

# File upload
uploaded_file = st.file_uploader(
    "Upload a document (PDF, PNG, JPG, JPEG)", type=["pdf", "png", "jpg", "jpeg"]
)

# Optional field list input
fields_input = st.text_area(
    "Optional: Enter comma-separated fields to extract", ""
)
fields_list = [f.strip() for f in fields_input.split(",") if f.strip()]

if uploaded_file is not None:
    # Save uploaded file temporarily
    temp_dir = Path("temp")
    temp_dir.mkdir(exist_ok=True)
    file_path = temp_dir / uploaded_file.name
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Show spinner while processing
    with st.spinner("Processing document... This may take a few moments."):
        # 1️⃣ Ingestion
        text = extract_text_from_file(str(file_path))
        if not text.strip():
            st.warning("No text could be extracted from the uploaded file.")
            st.stop()

        # 2️⃣ Routing
        doc_info = classify_document(text)
        doc_type = doc_info.get("doc_type", "unknown")
        st.success(f"Document Type: {doc_type.replace('_',' ').title()}, Confidence: {doc_info.get('confidence',0):.2f}")


        # 3️⃣ Extraction
        fields = extract_fields(text, doc_type, fields_list)

        # 4️⃣ Validation
        qa_info = validate_fields(fields, doc_type)

        # 5️⃣ Confidence & JSON output
        document_schema: DocumentSchema = update_document_schema(doc_type, fields, qa_info)

    # Display overall confidence
    st.subheader("Overall Confidence")
    st.progress(int(document_schema.overall_confidence * 100))

    # Display JSON output
    st.subheader("Extracted Document JSON")
    st.json(document_schema.dict())

    # Display per-field confidence bars
    st.subheader("Per-field Confidence")
    for field in document_schema.fields:
        col1, col2 = st.columns([2, 5])
        with col1:
            st.markdown(f"**{field.name}**")
        with col2:
            st.progress(int(field.confidence * 100))
        st.write(f"Value: `{field.value}`")

    # Live table preview
    st.subheader("Live Preview of Extracted Fields")
    if document_schema.fields:
        df = pd.DataFrame([
            {
                "Field Name": f.name,
                "Value": f.value,
                "Confidence": f"{f.confidence:.2f}",
            } for f in document_schema.fields
        ])
        st.dataframe(df)

    # Download JSON button
    json_bytes = BytesIO(json.dumps(document_schema.dict(), indent=4).encode("utf-8"))
    st.download_button(
        label="Download JSON",
        data=json_bytes,
        file_name=f"{uploaded_file.name}_extracted.json",
        mime="application/json"
    )
