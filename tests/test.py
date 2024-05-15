import fitz
import openai
from dotenv_vault import load_dotenv
import os

load_dotenv()

client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def extract_text_from_pdf(pdf_path):
    document = fitz.open(pdf_path)

    text = ""
    for page_num in range(document.page_count):
        page = document.load_page(page_num)
        text += page.get_text()

    return text


def summarize_text(text):
    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": "You are a financial expert skilled in summarizing financial earnings document and provide whether the user should invest in the company or not",
            },
            {
                "role": "user",
                "content": f"Summarize the following financial document: {text}",
            },
        ],
    )

    summary = completion.choices[0].message.content
    return summary


if __name__ == "__main__":
    pdf_path = "data\\input\\Amazon-Q4_2023_Transcript.pdf"

    pdf_text = extract_text_from_pdf(pdf_path)

    summary = summarize_text(pdf_text)
    print(summary)
