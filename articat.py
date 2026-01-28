#!/usr/bin/env python3
import sys
import io
import asyncio
from urllib.parse import urljoin
from urllib.request import Request, urlopen
from playwright.async_api import async_playwright
from PIL import Image
from readability import Document
from bs4 import BeautifulSoup

UA = "Mozilla/5.0"
HALF_BLOCK = "\u2580"
MAX_IMAGE_WIDTH = 120
MIN_IMAGE_AREA = 64 * 64
MIN_TEXT_CHARS = 200
HEADER_MARKER = "__HDR__"

async def fetch_html(url: str) -> str:
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(user_agent=UA)
        page = await context.new_page()
        await page.goto(url, wait_until="domcontentloaded")
        html = await page.content()
        await browser.close()
        return html


def pick_image_src(tag):
    src = tag.get("src")
    if src:
        return src

    for key in ("data-src", "data-original", "data-lazy-src"):
        src = tag.get(key)
        if src:
            return src

    srcset = tag.get("srcset") or tag.get("data-srcset")
    if srcset:
        candidates = [part.strip().split(" ")[0] for part in srcset.split(",") if part.strip()]
        if candidates:
            return candidates[-1]

    return None


def find_meta_image(soup):
    for key in ("og:image", "og:image:url", "twitter:image", "twitter:image:src"):
        tag = soup.find("meta", attrs={"property": key}) or soup.find("meta", attrs={"name": key})
        if tag and tag.get("content"):
            return tag["content"].strip()
    return None


def find_meta_content(soup, keys):
    for key in keys:
        tag = soup.find("meta", attrs={"property": key}) or soup.find("meta", attrs={"name": key})
        if tag and tag.get("content"):
            return tag["content"].strip()
    return None


def normalize_publication(name: str | None) -> str | None:
    if not name:
        return None
    name = name.strip()
    if name.startswith("@") and len(name) > 1:
        name = name[1:]
    return name or None


def mark_headings(soup):
    for heading in soup.find_all(["h1", "h2", "h3", "h4", "h5", "h6"]):
        text = heading.get_text(" ", strip=True)
        if not text:
            continue
        heading.clear()
        heading.append(f"{HEADER_MARKER}{text}")


def count_headings(soup):
    return len(soup.find_all(["h1", "h2", "h3", "h4", "h5", "h6"]))


def extract_fallback_text(full_soup):
    container = full_soup.find("article") or full_soup
    for tag in container(["script", "style", "noscript", "iframe"]):
        tag.decompose()

    has_testid = container.find(attrs={"data-testid": True}) is not None

    def classify_node(node):
        if not getattr(node, "name", None):
            return None
        tag = node.name
        if tag in ("h1", "h2", "h3", "h4", "h5", "h6"):
            return "heading"
        if tag == "p":
            return "paragraph"
        testid = (node.get("data-testid") or "").lower()
        if testid:
            if "heading" in testid or "title" in testid or "subhead" in testid:
                return "heading"
            if "paragraph" in testid or "body" in testid or "text" in testid:
                return "paragraph"
        if tag == "div":
            if node.find(["p", "h1", "h2", "h3", "h4", "h5", "h6"], recursive=True):
                return None
            class_name = " ".join(node.get("class", [])).lower()
            if "paragraph" in class_name or "body" in class_name or "text" in class_name:
                return "paragraph"
        if has_testid and testid:
            return "paragraph"
        return None

    nodes = container.find_all(lambda n: classify_node(n) is not None)
    parts = []
    for node in nodes:
        kind = classify_node(node)
        text = node.get_text(" ", strip=True)
        if not text:
            continue
        if kind == "heading":
            parts.append(text.upper())
        else:
            parts.append(text)

    if not parts:
        return container.get_text("\n")

    return "\n\n".join(parts)


def _download(url: str) -> bytes:
    req = Request(url, headers={"User-Agent": UA})
    with urlopen(req, timeout=15) as response:
        return response.read()


async def fetch_image_bytes(url: str) -> bytes:
    return await asyncio.to_thread(_download, url)


def image_bytes_to_ansi(image_bytes: bytes, width: int = MAX_IMAGE_WIDTH) -> str | None:
    with Image.open(io.BytesIO(image_bytes)) as img:
        img = img.convert("RGB")
        w, h = img.size
        if w * h < MIN_IMAGE_AREA:
            return None
        new_width = min(width, w)
        new_height = max(2, int((h / w) * new_width))
        if new_height % 2:
            new_height += 1
        img = img.resize((new_width, new_height))
        pixels = img.load()
        lines = []
        for y in range(0, new_height, 2):
            row = []
            for x in range(new_width):
                r1, g1, b1 = pixels[x, y]
                r2, g2, b2 = pixels[x, y + 1]
                row.append(
                    f"\x1b[38;2;{r1};{g1};{b1}m\x1b[48;2;{r2};{g2};{b2}m{HALF_BLOCK}"
                )
            row.append("\x1b[0m")
            lines.append("".join(row))
        return "\n".join(lines)


async def build_image_ansi_map(image_info: dict[str, dict[str, str]]) -> dict[str, str]:
    semaphore = asyncio.Semaphore(4)

    async def worker(token: str, info: dict[str, str]) -> tuple[str, str | None]:
        async with semaphore:
            try:
                image_bytes = await fetch_image_bytes(info["url"])
                ansi_art = image_bytes_to_ansi(image_bytes)
            except Exception:
                return token, None

        if not ansi_art:
            return token, None

        alt = info.get("alt", "")
        if alt:
            return token, f"[Image: {alt}]\n{ansi_art}"
        return token, ansi_art

    tasks = [worker(token, info) for token, info in image_info.items()]
    results = await asyncio.gather(*tasks)
    return {token: art for token, art in results if art}


async def main() -> int:
    if len(sys.argv) < 2:
        print("usage: articat <url>")
        return 1

    url = sys.argv[1]

    html = await fetch_html(url)

    full_soup = BeautifulSoup(html, "lxml")

    doc = Document(html, url=url)
    article_html = doc.summary(html_partial=True, keep_all_images=True)

    soup = BeautifulSoup(article_html, "lxml")

    for tag in soup(["script", "style", "noscript", "iframe"]):
        tag.decompose()

    mark_headings(soup)

    title = doc.short_title() or doc.title() or find_meta_content(full_soup, ("og:title", "twitter:title"))
    author = doc.author() or find_meta_content(
        full_soup,
        ("author", "article:author", "og:article:author", "twitter:creator"),
    )
    publication = normalize_publication(
        find_meta_content(
            full_soup,
            ("og:site_name", "application-name", "publisher", "article:publisher", "twitter:site", "apple-mobile-web-app-title"),
        )
    )
    header = f"{title or 'Unknown'} | {author or 'Unknown'} | {publication or 'Unknown'}"

    image_info = {}
    for i, img in enumerate(soup.find_all("img")):
        src = pick_image_src(img)
        if not src or src.startswith("data:"):
            img.decompose()
            continue
        alt = (img.get("alt") or "").strip()
        token = f"__IMG_{i}__"
        img.replace_with(soup.new_string(f"\n{token}\n"))
        image_info[token] = {"url": urljoin(url, src), "alt": alt}

    fallback_tokens = []
    if not image_info:
        meta_image = find_meta_image(full_soup)
        if meta_image:
            token = "__IMG_FALLBACK_0__"
            image_info[token] = {"url": urljoin(url, meta_image), "alt": "Top image"}
            fallback_tokens.append(token)
        else:
            article_tag = full_soup.find("article")
            if article_tag:
                img = article_tag.find("img")
                src = pick_image_src(img) if img else None
                if src and not src.startswith("data:"):
                    token = "__IMG_FALLBACK_0__"
                    image_info[token] = {"url": urljoin(url, src), "alt": (img.get("alt") or "").strip()}
                    fallback_tokens.append(token)

    image_ansi_map = await build_image_ansi_map(image_info) if image_info else {}

    text = soup.get_text("\n")
    text_len = len(text.strip())
    summary_headings = count_headings(soup)
    full_container = full_soup.find("article") or full_soup
    full_headings = count_headings(full_container)
    if text_len < MIN_TEXT_CHARS or (full_headings > 1 and summary_headings < full_headings):
        text = extract_fallback_text(full_soup)

    lines = [line.strip() for line in text.splitlines()]
    paragraphs = []
    current = []

    for line in lines:
        if line.startswith(HEADER_MARKER):
            heading = line[len(HEADER_MARKER):].strip().upper()
            if heading:
                if current:
                    paragraphs.append(" ".join(current))
                    current = []
                paragraphs.append(heading)
            continue
        if line in image_info:
            if current:
                paragraphs.append(" ".join(current))
                current = []
            paragraphs.append(line)
            continue
        if not line:
            if current:
                paragraphs.append(" ".join(current))
                current = []
            continue
        current.append(line)

    if current:
        paragraphs.append(" ".join(current))

    normalized = "\n\n".join(paragraphs)
    if fallback_tokens:
        normalized = "\n\n".join(fallback_tokens) + "\n\n" + normalized
    for token in image_info:
        normalized = normalized.replace(token, image_ansi_map.get(token, ""))

    print("\n\n")
    normalized = normalized.strip("\n")
    if normalized:
        print(header)
        print()
        print(normalized)
    else:
        print(header)
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
