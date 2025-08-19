from typing import List
from core.schema import FieldItem, DocumentSchema, QAInfo

def assign_confidence(fields: List[FieldItem], qa: QAInfo) -> float:
    """
    Compute overall document confidence from individual fields and QA results.
    Weighted formula:
    - Average of field confidences
    - Penalty for each failed QA rule
    """
    if not fields:
        return 0.0
    
    field_avg = sum(f.confidence for f in fields) / len(fields)
    
    # Penalty for failed QA rules
    penalty = 0.1 * len(qa.failed_rules)
    overall_conf = max(0.0, min(1.0, field_avg - penalty))
    
    return overall_conf

def update_document_schema(doc_type: str, fields: List[FieldItem], qa: QAInfo) -> DocumentSchema:
    """
    Build final DocumentSchema object with per-field and overall confidence
    """
    overall_conf = assign_confidence(fields, qa)
    return DocumentSchema(doc_type=doc_type, fields=fields, overall_confidence=overall_conf, qa=qa)
