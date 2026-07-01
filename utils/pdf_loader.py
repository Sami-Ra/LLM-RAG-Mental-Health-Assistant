import re
import fitz
from pathlib import Path


class PDFLoader:
    """Loads PDF documents and extracts clean text page by page."""

    def __init__(self, pdf_directory: Path) -> None:
        self.pdf_directory = pdf_directory

    @staticmethod
    def clean_text(text: str) -> str:
        """
        Clean extracted PDF text.

        Normalizes spaces and tabs to a single space while preserving
        paragraph breaks (double newlines).
        """
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = text.strip()
        return text

    def load_documents(self) -> list[dict]:
        """
        Load all PDFs from the configured directory.

        Returns
        -------
        list[dict]
            Each entry: {"source": filename, "page": page_number, "text": cleaned_text}

        Notes
        -----
        Corrupted or unreadable files are skipped with a logged warning.
        Pages with fewer than 20 characters are also skipped.
        """
        documents: list[dict] = []
        pdf_files = sorted(self.pdf_directory.glob("*.pdf"))

        if not pdf_files:
            print(f"No PDF files found in {self.pdf_directory}.")
            return documents

        print(f"Found {len(pdf_files)} PDF file(s).\n")

        for pdf_file in pdf_files:
            print(f"Loading {pdf_file.name}")

            try:
                doc = fitz.open(pdf_file)
            except Exception as e:
                print(f"  Skipping {pdf_file.name}: could not open file ({e})")
                continue

            for page_num, page in enumerate(doc, start=1):
                try:
                    text = page.get_text()
                    text = self.clean_text(text)
                    if len(text) > 20:
                        documents.append({
                            "source": pdf_file.name,
                            "page": page_num,
                            "text": text,
                        })
                    else:
                        print(f"  Skipping page {page_num} (insufficient text).")
                except Exception as e:
                    print(f"  Skipping page {page_num} of {pdf_file.name}: {e}")

            doc.close()

        print(f"\nLoaded {len(documents)} pages.\n")
        return documents
