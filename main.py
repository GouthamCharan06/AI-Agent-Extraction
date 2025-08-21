import sys
from pathlib import Path
import json
from io import BytesIO
import pandas as pd
import streamlit as st

# # Add project root to Python path
# sys.path.append(str(Path(__file__).resolve().parent.parent))

from core.ingestion import extract_text_from_file
from core.routing import classify_document
from core.extraction import extract_fields, ExtractionOutput
from core.validation import validate_fields
from core.confidence import update_document_schema
from core.schema import DocumentSchema, FieldItem as SchemaFieldItem


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

    with st.spinner("Processing document... This may take a few moments."):
        
        # 1. Ingestion
        
        text = extract_text_from_file(str(file_path))
        if not text.strip():
            st.warning("No text could be extracted from the uploaded file.")
            st.stop()

        
        # 2. Routing
        
        doc_info = classify_document(text)
        doc_type = doc_info.get("doc_type", "unknown")
        st.success(
            f"Document Type: {doc_type.replace('_',' ').title()}, "
            f"Confidence: {doc_info.get('confidence', 0):.2f}"
        )

        
        # 3. Extraction
        
        extraction_output: ExtractionOutput = extract_fields(text, doc_type, fields_list)

        
        # Convert extraction FieldItems -> schema FieldItems
        
        schema_fields = []
        for f in extraction_output.fields:
            value_str = str(f.value) if f.value is not None else ""
            source_dict = {"page": f.source.page, "bbox": f.source.bbox} if f.source else {}
            schema_fields.append(
                SchemaFieldItem(
                    name=f.name,
                    value=value_str,
                    confidence=f.confidence,
                    source=source_dict
                )
            )

        
        # 4. Validation
        
        qa_info = validate_fields(schema_fields, doc_type)

        
        # 5. Confidence & JSON output
        
        document_schema: DocumentSchema = update_document_schema(doc_type, schema_fields, qa_info)

    
    # Streamlit Display
    
    # Overall Confidence
    st.subheader("Overall Confidence")
    st.progress(int(document_schema.overall_confidence * 100))

    # JSON Output
    st.subheader("Extracted Document JSON")
    st.json(document_schema.model_dump())

    # Per-field confidence bars
    st.subheader("Per-field Confidence")
    for field in document_schema.fields:
        col1, col2 = st.columns([2, 5])
        with col1:
            st.markdown(f"**{field.name}**")
        with col2:
            st.progress(int(field.confidence * 100))
        st.write(f"Value: {field.value}")

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
    json_bytes = BytesIO(
        json.dumps(document_schema.model_dump(), indent=4).encode("utf-8")
    )
    st.download_button(
        label="Download JSON",
        data=json_bytes,
        file_name=f"{uploaded_file.name}_extracted.json",
        mime="application/json"
    )
