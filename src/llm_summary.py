from transformers import pipeline
import functools

@functools.lru_cache
def get_summarizer():
    return pipeline("summarization", model="google/flan-t5-small")

def summarize_transcript(text):
    summarizer = get_summarizer()
    result = summarizer(text[:512])
    return result[0]["summary_text"]
