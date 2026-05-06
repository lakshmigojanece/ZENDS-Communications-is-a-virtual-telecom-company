import re   # Regular Expression module in Python.,text processing tool, clean and matching pattern..
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    return text
