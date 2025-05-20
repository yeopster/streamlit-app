import os
import PyPDF2

def extract_text_from_pdfs(pdf_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for pdf_file in os.listdir(pdf_dir):
        if pdf_file.endswith(".pdf"):
            pdf_path = os.path.join(pdf_dir, pdf_file)
            with open(pdf_path, "rb") as file:
                reader = PyPDF2.PdfReader(file)
                text = ""
                for page in reader.pages:
                    text += page.extract_text() or ""
            output_path = os.path.join(output_dir, pdf_file.replace(".pdf", ".txt"))
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(text)
            print(f"Extracted text from {pdf_file}")

if __name__ == "__main__":
    extract_text_from_pdfs("data/", "data/processed/")