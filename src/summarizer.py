from pdf_extractor import extract_text_from_pdf
from transformers import BartTokenizer, pipeline

tokenizer = BartTokenizer.from_pretrained("sshleifer/distilbart-cnn-12-6")


def create_segments(text, tokenizer, max_length, overlap=200):
    tokens = tokenizer(
        text,
        add_special_tokens=True,
        truncation=True,
        max_length=None,
        return_tensors="pt",
    ).input_ids[0]
    segments = []
    for i in range(0, len(tokens), max_length - overlap):
        end = min(i + max_length, len(tokens))
        segments.append(tokens[i:end])
    return segments


def summarize_text(text, tokenizer):
    summarizer = pipeline(
        "summarization", model="sshleifer/distilbart-cnn-12-6", framework="pt"
    )
    segments = create_segments(text, tokenizer, 1024, 200)
    summaries = []

    for segment in segments:
        segment_text = tokenizer.decode(segment, skip_special_tokens=True)
        summary = summarizer(
            segment_text, max_length=60, min_length=30, do_sample=False
        )
        summaries.append(summary[0]["summary_text"])

    # re-summarize if there are multiple segments
    if len(summaries) > 1:
        full_text = " ".join(summaries)
        summary = summarizer(full_text, max_length=60, min_length=30, do_sample=False)
        return summary[0]["summary_text"]

    return " ".join(summaries)


def main(pdf_path):
    extracted_text = extract_text_from_pdf(pdf_path)
    summary_text = summarize_text(extracted_text, tokenizer)
    return summary_text


if __name__ == "__main__":
    pdf_path = "data/input/Amazon-Q4_2023_Transcript.pdf"
    summary = main(pdf_path)
    print(summary)
