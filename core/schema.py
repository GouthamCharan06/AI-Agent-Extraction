from pydantic import BaseModel, Field
from typing import List, Dict, Optional

class FieldItem(BaseModel):
    name: str
    value: str
    confidence: float
    source: Optional[Dict] = Field(default_factory=dict)  # e.g., {"page":1, "bbox":[x1,y1,x2,y2]}

class QAInfo(BaseModel):
    passed_rules: List[str] = []
    failed_rules: List[str] = []
    notes: Optional[str] = ""

class DocumentSchema(BaseModel):
    doc_type: str
    fields: List[FieldItem] = []
    overall_confidence: float
    qa: QAInfo
