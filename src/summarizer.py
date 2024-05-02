from pdf_extractor import extract_text_from_pdf
from transformers import pipeline


def summarize_text(text):
    summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
    summary = summarizer(text, max_length=1024, min_length=100, do_sample=False)
    return " ".join([s["summary_text"] for s in summary])


# Main function to extract and summarize the PDF content
def main(pdf_path):
    extracted_text = extract_text_from_pdf(pdf_path)
    summary_text = summarize_text(extracted_text)
    return summary_text


if __name__ == "__main__":
    pdf_path = "data\input\Amazon-Q4_2023_Transcript.pdf"
    summary = main(pdf_path)
    print(summary)
