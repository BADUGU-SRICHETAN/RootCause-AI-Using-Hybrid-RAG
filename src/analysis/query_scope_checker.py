import re
from typing import Dict, Any


class QueryScopeChecker:

    def __init__(self):

        self.out_of_scope_patterns = {

            "financial_information": [
                r"\bcost\b",
                r"\bsalary\b",
                r"\bsalaries\b",
                r"\bprice\b",
                r"\bprices\b",
                r"\bpurchase price\b",
                r"\bhosting cost\b",
                r"\bmonthly cloud hosting\b",
                r"\bbudget\b",
                r"\bannual expenditure\b"
            ],

            "employee_information": [
                r"\bwho is responsible for\b",
                r"\bwho was responsible for\b",
                r"\bwho is the engineer responsible for\b",
                r"\bwhich engineer\b",
                r"\bwhich employee\b",
                r"\bwho approved\b",
                r"\bwho fixed\b",
                r"\bwho resolved\b",
                r"\bemployee\b",
                r"\bannual salary\b"
            ],

            "infrastructure_inventory": [
                r"\bphysical location\b",
                r"\bdata center location\b",
                r"\bwhich production server\b",
                r"\bwhich server\b",
                r"\bcurrently hosts\b"
            ],

            "vendor_information": [
                r"\bmonitoring vendor\b",
                r"\bwhich vendor\b",
                r"\bvendor does the organization\b",
                r"\bcurrently purchase\b"
            ],

            "future_prediction": [
                r"\btomorrow\b",
                r"\bpredict\b",
                r"\bprediction\b",
                r"\bnext production outage\b",
                r"\bexact number of .* requests\b",
                r"\bfuture\b"
            ]
        }

    def check(self, query: str) -> Dict[str, Any]:

        if not query or not query.strip():

            raise ValueError("Query cannot be empty.")

        normalized_query = query.lower().strip()

        for category, patterns in self.out_of_scope_patterns.items():

            for pattern in patterns:

                if re.search(pattern, normalized_query):

                    return {
                        "in_scope": False,
                        "category": category,
                        "matched_pattern": pattern,
                        "reason": (
                            "This question requires information "
                            "outside the historical incident database."
                        )
                    }

        return {
            "in_scope": True,
            "category": "historical_incident_investigation",
            "matched_pattern": None,
            "reason": (
                "The question can proceed to historical "
                "incident evidence retrieval."
            )
        }