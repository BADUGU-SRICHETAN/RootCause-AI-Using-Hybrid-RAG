import json
import csv

from pathlib import Path
from typing import List

from langchain_core.documents import Document


class DocumentLoader:

    SUPPORTED_EXTENSIONS = {
        ".txt",
        ".log",
        ".md",
        ".pdf",
        ".json",
        ".jsonl",
        ".csv"
    }

    def __init__(self, data_directory: str = "data/raw"):

        self.data_directory = Path(data_directory)

        if not self.data_directory.exists():
            raise FileNotFoundError(
                f"Data directory not found: {self.data_directory}"
            )

    def load_documents(self) -> List[Document]:

        documents = []

        for file_path in self.data_directory.rglob("*"):

            if not file_path.is_file():
                continue

            if file_path.suffix.lower() not in self.SUPPORTED_EXTENSIONS:
                continue

            print(f"Loading: {file_path.name}")

            try:

                extension = file_path.suffix.lower()

                if extension == ".pdf":
                    documents.extend(self._load_pdf(file_path))

                elif extension in {".txt", ".log", ".md"}:
                    documents.extend(self._load_text(file_path))

                elif extension == ".json":
                    documents.extend(self._load_json(file_path))

                elif extension == ".jsonl":
                    documents.extend(self._load_jsonl(file_path))

                elif extension == ".csv":
                    documents.extend(self._load_csv(file_path))

            except Exception as error:

                print(
                    f"Failed to load {file_path.name}: {error}"
                )

        print(f"\nTotal documents loaded: {len(documents)}")

        return documents

    def _load_text(self, file_path: Path) -> List[Document]:

        content = file_path.read_text(
            encoding="utf-8",
            errors="replace"
        )

        if not content.strip():
            return []

        return [
            Document(
                page_content=content,
                metadata={
                    "source": str(file_path),
                    "file_type": file_path.suffix.lower()
                }
            )
        ]

    def _load_pdf(self, file_path: Path) -> List[Document]:

        import fitz

        documents = []

        with fitz.open(file_path) as pdf:

            for page_number, page in enumerate(pdf, start=1):

                content = page.get_text()

                if not content.strip():
                    continue

                documents.append(
                    Document(
                        page_content=content,
                        metadata={
                            "source": str(file_path),
                            "file_type": ".pdf",
                            "page_number": page_number
                        }
                    )
                )

        return documents

    def _load_json(self, file_path: Path) -> List[Document]:

        with open(file_path, "r", encoding="utf-8") as file:
            data = json.load(file)

        records = data if isinstance(data, list) else [data]

        documents = []

        for index, record in enumerate(records):

            content = json.dumps(
                record,
                ensure_ascii=False,
                indent=2
            )

            metadata = {
                "source": str(file_path),
                "file_type": ".json",
                "record_index": index
            }

            if isinstance(record, dict) and record.get("incident_id"):
                metadata["incident_id"] = str(record["incident_id"])

            documents.append(
                Document(
                    page_content=content,
                    metadata=metadata
                )
            )

        return documents

    def _load_jsonl(self, file_path: Path) -> List[Document]:

        documents = []

        with open(file_path, "r", encoding="utf-8") as file:

            for index, line in enumerate(file):

                if not line.strip():
                    continue

                record = json.loads(line)

                documents.append(
                    Document(
                        page_content=json.dumps(
                            record,
                            ensure_ascii=False,
                            indent=2
                        ),
                        metadata={
                            "source": str(file_path),
                            "file_type": ".jsonl",
                            "record_index": index
                        }
                    )
                )

        return documents

    def _load_csv(self, file_path: Path) -> List[Document]:

        documents = []

        with open(
            file_path,
            "r",
            encoding="utf-8-sig",
            newline=""
        ) as file:

            reader = csv.DictReader(file)

            for index, row in enumerate(reader):

                content = "\n".join(
                    f"{key}: {value}"
                    for key, value in row.items()
                    if value is not None
                )

                documents.append(
                    Document(
                        page_content=content,
                        metadata={
                            "source": str(file_path),
                            "file_type": ".csv",
                            "record_index": index
                        }
                    )
                )

        return documents