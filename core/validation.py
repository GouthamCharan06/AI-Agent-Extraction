from typing import List
from core.schema import FieldItem, QAInfo
import re

def validate_fields(fields: List[FieldItem], doc_type: str) -> QAInfo:
    """
    Apply validation rules to extracted fields.
    Supports:
    - Date format checking
    - Numeric/total checks
    - Low-confidence notes
    """
    qa = QAInfo()

    for field in fields:
        fname_lower = field.name.lower()
        
        # Date validation (basic regex: YYYY-MM-DD or DD/MM/YYYY)
        if "date" in fname_lower:
            if re.search(r"\d{4}-\d{2}-\d{2}", field.value) or re.search(r"\d{2}/\d{2}/\d{4}", field.value):
                qa.passed_rules.append(f"{field.name}_date_format")
            else:
                qa.failed_rules.append(f"{field.name}_date_format")
        
        # Amount/Total validation
        if "total" in fname_lower or "amount" in fname_lower:
            try:
                float(field.value.replace(",", "").replace("$", ""))
                qa.passed_rules.append(f"{field.name}_numeric")
            except Exception:
                qa.failed_rules.append(f"{field.name}_numeric")
    
    # Notes for low-confidence fields
    low_conf_fields = [f.name for f in fields if f.confidence < 0.6]
    if low_conf_fields:
        qa.notes = f"Low confidence fields: {', '.join(low_conf_fields)}"
    
    return qa
