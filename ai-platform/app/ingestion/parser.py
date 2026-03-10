import io
from PyPDF2 import PdfReader

class Parser:
    """
    Parse documents: PDF or raw text
    """
    def parse(self, docs):
        """
        docs: list of tuples [(name, content)]
        Returns: list of strings (parsed text)
        """
        parsed_docs = []
        for name, content in docs:
            if name.lower().endswith(".pdf"):
                try:
                    pdf = PdfReader(io.BytesIO(content))
                    text = "\n".join([page.extract_text() or "" for page in pdf.pages])
                    parsed_docs.append(text)
                except Exception as e:
                    print(f"Failed to parse {name}: {e}")
            else:
                # assume text content
                if isinstance(content, bytes):
                    content = content.decode("utf-8", errors="ignore")
                parsed_docs.append(content)
        return parsed_docs