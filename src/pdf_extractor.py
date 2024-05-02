import fitz


def extract_text_from_pdf(path):
    """
    Extracts all text from PDF file
    """
    document = fitz.open(path)
    text = ""

    for page in document:
        text += page.get_text()

    document.close()

    return text


if __name__ == "__main__":
    pdf_path = "data\input\Amazon-Q4_2023_Transcript.pdf"
    extracted_text = extract_text_from_pdf(pdf_path)
    print(extracted_text)
