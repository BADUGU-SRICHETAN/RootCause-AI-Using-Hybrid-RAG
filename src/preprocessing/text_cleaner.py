import re
from typing import List

from langchain_core.documents import Document


class TextCleaner:

    ANSI_PATTERN = re.compile(
        r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])"
    )

    def clean_text(self, text: str) -> str:

        if not text:
            return ""

        # Remove terminal color codes
        text = self.ANSI_PATTERN.sub("", text)

        # Normalize line endings
        text = text.replace("\r\n", "\n").replace("\r", "\n")

        # Remove trailing spaces
        lines = [line.rstrip() for line in text.split("\n")]

        text = "\n".join(lines)

        # Remove excessive blank lines
        text = re.sub(r"\n{3,}", "\n\n", text)

        return text.strip()

    def clean_documents(
        self,
        documents: List[Document]
    ) -> List[Document]:

        cleaned_documents = []

        for document in documents:

            cleaned_text = self.clean_text(document.page_content)

            if not cleaned_text:
                continue

            cleaned_documents.append(
                Document(
                    page_content=cleaned_text,
                    metadata=document.metadata.copy()
                )
            )

        return cleaned_documents