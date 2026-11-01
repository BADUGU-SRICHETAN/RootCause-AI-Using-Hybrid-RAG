import os
from typing import List

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI


class RootCauseAnalyzer:

    def __init__(self, model_name: str = "gpt-4o-mini"):

        load_dotenv()

        api_key = os.getenv("OPENAI_API_KEY")

        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY is missing from the environment."
            )

        self.llm = ChatOpenAI(
            model=model_name,
            temperature=0,
            api_key=api_key
        )

        self.prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                """
                You are a software incident investigation analyst.

                Analyze historical incident evidence to identify
                potential causes of the current incident.

                Rules:
                1. Use only the supplied evidence.
                2. Never invent technical facts.
                3. Distinguish confirmed evidence from hypotheses.
                4. Cite evidence using incident IDs or sources.
                5. Explicitly mention when evidence is insufficient.
                6. Never present a potential cause as confirmed fact.
                7. Recommend practical investigation steps.

                Structure your response using these sections:

                ## 1. Incident Summary
                ## 2. Potential Root Causes
                ## 3. Supporting Evidence
                ## 4. Confidence Assessment
                ## 5. Recommended Investigation Steps

                For each potential cause, explain why it is plausible,
                identify supporting evidence, and mention uncertainty.

                Confidence must reflect evidence quality and relevance,
                not speculation.
                """
            ),
            (
                "human",
                """
                Current Incident:

                {query}

                Retrieved Historical Evidence:

                {evidence}

                Analyze the incident using the supplied evidence.
                """
            )
        ])

        self.chain = self.prompt | self.llm | StrOutputParser()

    @staticmethod
    def build_evidence_context(
        documents: List[Document]
    ) -> str:

        evidence_sections = []

        for index, document in enumerate(documents, start=1):

            metadata = document.metadata

            incident_id = metadata.get(
                "incident_id",
                metadata.get("source", f"Evidence_{index}")
            )

            evidence = (
                f"Evidence {index}\n"
                f"Incident ID: {incident_id}\n"
                f"Content: {document.page_content}\n"
            )

            evidence_sections.append(evidence)

        return "\n".join(evidence_sections)

    def analyze(
        self,
        query: str,
        retrieved_documents: List[Document]
    ) -> str:

        if not query.strip():
            raise ValueError("Incident query cannot be empty.")

        if not retrieved_documents:
            return (
                "Insufficient evidence: No relevant historical "
                "incidents were retrieved. More incident records "
                "are required for analysis."
            )

        evidence_context = self.build_evidence_context(
            retrieved_documents
        )

        response = self.chain.invoke({
            "query": query,
            "evidence": evidence_context
        })

        return response.strip()
