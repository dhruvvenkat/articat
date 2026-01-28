#!/usr/bin/env python3
import sys
import requests
from readability import Document
from bs4 import BeautifulSoup

UA = "Mozilla/5.0"

if len(sys.argv) < 2:
    print("usage: articat <url>")
    sys.exit(1)

url = sys.argv[1]

html = requests.get(url, headers={"User-Agent": UA}).text

doc = Document(html)
article_html = doc.summary(html_partial=True)

soup = BeautifulSoup(article_html, "lxml")

for tag in soup(["script", "style", "noscript", "iframe"]):
    tag.decompose()

text = soup.get_text("\n")

lines = [line.strip() for line in text.splitlines()]
paragraphs = []
current = []

for line in lines:
    if not line:
        if current:
            paragraphs.append(" ".join(current))
            current = []
        continue
    current.append(line)

if current:
    paragraphs.append(" ".join(current))

normalized = "\n\n".join(paragraphs)

print(normalized.strip())
