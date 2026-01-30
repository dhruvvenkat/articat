#!/usr/bin/env python3
import sys
import io
import asyncio
import json
import math
from datetime import datetime
from urllib.parse import urljoin
from urllib.request import Request, urlopen
from playwright.async_api import async_playwright
from PIL import Image
from readability import Document
import re
from bs4 import BeautifulSoup, Tag

UA = "Mozilla/5.0"
HALF_BLOCK = "\u2580"
MAX_IMAGE_WIDTH = 120
MIN_IMAGE_AREA = 64 * 64
MIN_TEXT_CHARS = 200
HEADER_MARKER = "__HDR__"
SPINNER_FRAMES = ["|", "/", "-", "\\"]
JUNK_RE = re.compile(
    r"(nav|menu|breadcrumb|header|footer|masthead|sidebar|related|promo|sponsor|"
    r"subscribe|newsletter|social|share|signin|login|cookie|advert|ads?|banner|"
    r"widget|search|comment|comments|trending|popular|recommend)",
    re.I,
)

async def fetch_html(url: str, debug: bool = False) -> str:
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(user_agent=UA)
        page = await context.new_page()
        await page.goto(url, wait_until="domcontentloaded")
        previous_height = 0
        for _ in range(12):
            height = await page.evaluate("document.body.scrollHeight")
            await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
            await page.wait_for_timeout(1000)
            if height == previous_height:
                break
            previous_height = height
        html = await page.content()
        if debug:
            print("[debug] fetched html length:", len(html), file=sys.stderr)
        await browser.close()
        return html


async def _spinner(label: str, stop_event: asyncio.Event) -> None:
    if not sys.stderr.isatty():
        return
    i = 0
    while not stop_event.is_set():
        frame = SPINNER_FRAMES[i % len(SPINNER_FRAMES)]
        sys.stderr.write(f"\r{label} {frame}")
        sys.stderr.flush()
        await asyncio.sleep(0.1)
        i += 1
    sys.stderr.write("\r" + (" " * (len(label) + 2)) + "\r")
    sys.stderr.flush()


async def run_with_spinner(coro, label: str):
    if not sys.stderr.isatty():
        return await coro
    stop_event = asyncio.Event()
    spinner_task = asyncio.create_task(_spinner(label, stop_event))
    try:
        return await coro
    finally:
        stop_event.set()
        await spinner_task


def pick_image_src(tag):
    if not isinstance(tag, Tag):
        return None

    def parse_srcset(srcset):
        best_any = (None, -1.0)
        best_non_webp = (None, -1.0)
        for part in srcset.split(","):
            part = part.strip()
            if not part:
                continue
            bits = part.split()
            url = bits[0]
            score = 0.0
            if len(bits) > 1:
                descriptor = bits[-1]
                if descriptor.endswith("w") and descriptor[:-1].isdigit():
                    score = float(descriptor[:-1])
                elif descriptor.endswith("x"):
                    try:
                        score = float(descriptor[:-1]) * 1000.0
                    except ValueError:
                        score = 0.0
            is_webp = ".webp" in url.lower() or "format=webp" in url.lower() or "fm=webp" in url.lower()
            if score >= best_any[1]:
                best_any = (url, score)
            if not is_webp and score >= best_non_webp[1]:
                best_non_webp = (url, score)
        return best_non_webp[0] or best_any[0]

    if tag.name == "picture":
        for source in tag.find_all("source"):
            srcset = source.get("srcset") or source.get("data-srcset") or source.get("data-lazy-srcset")
            if srcset:
                return parse_srcset(srcset)
        img = tag.find("img")
        if img:
            return pick_image_src(img)
        return None

    if tag.name == "source":
        srcset = tag.get("srcset") or tag.get("data-srcset") or tag.get("data-lazy-srcset")
        if srcset:
            return parse_srcset(srcset)
        return None

    srcset = (
        tag.get("srcset")
        or tag.get("data-srcset")
        or tag.get("data-lazy-srcset")
        or tag.get("data-src-set")
        or tag.get("data-bgset")
    )
    if srcset:
        return parse_srcset(srcset)

    for key in (
        "data-src",
        "data-original",
        "data-lazy-src",
        "data-url",
        "data-bg",
        "data-background",
        "data-background-image",
        "data-image",
    ):
        src = tag.get(key)
        if src:
            return src

    src = tag.get("src")
    if src and not src.startswith("data:"):
        return src

    if tag.parent and tag.parent.name == "picture":
        return pick_image_src(tag.parent)

    style = tag.get("style") or ""
    match = re.search(r"background-image\\s*:\\s*url\\(([^)]+)\\)", style, re.I)
    if match:
        return match.group(1).strip(" \"'")

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


def infer_cdn_prefix(soup):
    hero = find_meta_image(soup)
    if not hero:
        return None
    match = re.match(r"^(https?://[^/]+/.*/image/upload/)", hero)
    if match:
        return match.group(1)
    return None


def resolve_image_url(src: str | None, base_url: str, cdn_prefix: str | None) -> str | None:
    if not src:
        return None
    src = src.strip()
    if not src:
        return None
    if src.startswith("//"):
        return "https:" + src
    if src.startswith("http://") or src.startswith("https://"):
        return src
    if cdn_prefix and re.match(r"^(w_|c_|q_|t_|g_)", src):
        return normalize_image_url(cdn_prefix + src.lstrip("/"))
    return normalize_image_url(urljoin(base_url, src))


def normalize_image_url(url: str) -> str:
    if "/image/upload/" not in url:
        return url
    if "images.complex.com" not in url and "cloudinary" not in url:
        return url
    prefix, rest = url.split("/image/upload/", 1)
    if "/" in rest:
        first, remainder = rest.split("/", 1)
        if "f_auto" in first:
            first = first.replace("f_auto", "f_jpg")
            return f"{prefix}/image/upload/{first}/{remainder}"
        if first.startswith("v") and first[1:].isdigit():
            return f"{prefix}/image/upload/f_jpg/{rest}"
        return f"{prefix}/image/upload/f_jpg,{first}/{remainder}"
    return f"{prefix}/image/upload/f_jpg/{rest}"


def normalize_publication(name: str | None) -> str | None:
    if not name:
        return None
    name = name.strip()
    if name.startswith("@") and len(name) > 1:
        name = name[1:]
    return name or None


def extract_ld_json(soup):
    blocks = []
    for tag in soup.find_all("script", attrs={"type": "application/ld+json"}):
        raw = tag.string or tag.get_text(strip=True)
        if not raw:
            continue
        try:
            data = json.loads(raw)
        except Exception:
            continue
        blocks.append(data)
    return blocks


def extract_published_date(soup):
    meta_date = find_meta_content(
        soup,
        (
            "article:published_time",
            "og:article:published_time",
            "datePublished",
            "publish_date",
            "pubdate",
            "publishdate",
            "timestamp",
            "date",
        ),
    )
    if meta_date:
        return meta_date

    for block in extract_ld_json(soup):
        items = block if isinstance(block, list) else [block]
        for item in items:
            if not isinstance(item, dict):
                continue
            if "datePublished" in item:
                return item.get("datePublished")
            if "@graph" in item and isinstance(item["@graph"], list):
                for node in item["@graph"]:
                    if isinstance(node, dict) and "datePublished" in node:
                        return node.get("datePublished")
    return None


def format_date(value: str | None) -> str | None:
    if not value:
        return None
    raw = value.strip()
    if not raw:
        return None
    try:
        cleaned = raw.replace("Z", "+00:00")
        dt = datetime.fromisoformat(cleaned)
        return dt.strftime("%Y-%m-%d")
    except Exception:
        return raw


def extract_canonical_url(soup, fallback_url: str) -> str:
    link = soup.find("link", attrs={"rel": "canonical"})
    href = link.get("href") if link else None
    if href:
        return href.strip()
    meta = find_meta_content(soup, ("og:url",))
    if meta:
        return meta.strip()
    return fallback_url


def mark_headings(soup):
    for heading in soup.find_all(["h1", "h2", "h3", "h4", "h5", "h6"]):
        text = heading.get_text(" ", strip=True)
        if not text:
            continue
        heading.clear()
        heading.append(f"{HEADER_MARKER}{text}")


def unwrap_noscript_images(soup):
    for noscript in soup.find_all("noscript"):
        try:
            inner = BeautifulSoup(noscript.decode_contents(), "lxml")
        except Exception:
            noscript.decompose()
            continue
        for node in inner.find_all(["img", "picture", "source", "figure"]):
            noscript.insert_before(node)
        noscript.decompose()


def strip_junk_nodes(container):
    for tag in container.find_all(True):
        if not getattr(tag, "attrs", None):
            continue
        ident = " ".join(tag.get("class", [])) + " " + (tag.get("id") or "")
        if not ident or not JUNK_RE.search(ident):
            continue
        if tag.name in ("article", "main"):
            continue
        if tag.find(["article", "main"]):
            continue
        text_len = len(tag.get_text(" ", strip=True))
        if text_len > 200:
            continue
        tag.decompose()


def count_headings(soup):
    return len(soup.find_all(["h1", "h2", "h3", "h4", "h5", "h6"]))


def count_images(soup):
    sources = set()
    for node in soup.find_all(True):
        if node.name == "source":
            continue
        if node.name == "img" and node.find_parent("picture"):
            continue
        src = pick_image_src(node)
        if src and not src.startswith("data:"):
            sources.add(src)
    return len(sources)


def extract_fallback_blocks(full_soup, base_url, cdn_prefix):
    container = full_soup.find("article") or full_soup.find("main") or full_soup
    unwrap_noscript_images(container)
    strip_junk_nodes(container)
    for tag in container(
        ["script", "style", "iframe", "nav", "header", "footer", "aside"]
    ):
        tag.decompose()
    for tag in container.find_all(attrs={"role": ["navigation", "banner", "contentinfo"]}):
        tag.decompose()

    has_testid = container.find(attrs={"data-testid": True}) is not None

    def classify_node(node):
        if not isinstance(node, Tag):
            return None
        tag = node.name
        if tag not in ("source", "script", "style") and pick_image_src(node):
            return "image"
        if tag in ("h1", "h2", "h3", "h4", "h5", "h6"):
            return "heading"
        if tag == "p":
            return "paragraph"
        if tag == "figure" and node.find(["img", "picture"]):
            return "image"
        testid = (node.get("data-testid") or "").lower()
        if testid:
            if "heading" in testid or "title" in testid or "subhead" in testid:
                return "heading"
            if "paragraph" in testid or "body" in testid or "text" in testid:
                return "paragraph"
        if tag == "div":
            if node.find(
                ["p", "h1", "h2", "h3", "h4", "h5", "h6", "img", "figure", "picture"],
                recursive=True,
            ):
                return None
            class_name = " ".join(node.get("class", [])).lower()
            if "paragraph" in class_name or "body" in class_name or "text" in class_name:
                return "paragraph"
            if "heading" in class_name or "headline" in class_name or "title" in class_name:
                return "heading"
        if has_testid and testid:
            return "paragraph"
        return None

    def iter_blocks(node):
        kind = classify_node(node)
        if kind:
            if kind == "image":
                img = None
                src = pick_image_src(node)
                if node.name == "img":
                    img = node
                if not src:
                    img = node.find("img") or node.find("picture")
                    src = pick_image_src(img) if img else None
                resolved = resolve_image_url(src, base_url, cdn_prefix)
                if not resolved or resolved.startswith("data:"):
                    return
                alt = (img.get("alt") or "").strip() if img else ""
                yield ("image", resolved, alt)
                return
            text = node.get_text(" ", strip=True)
            if text:
                yield (kind, text)
            return

        for child in getattr(node, "children", []):
            if isinstance(child, Tag):
                yield from iter_blocks(child)

    blocks = []
    image_info = {}
    for kind, value, *rest in iter_blocks(container):
        if kind == "image":
            token = f"__IMG_FALLBACK_{len(image_info)}__"
            alt = rest[0] if rest else ""
            image_info[token] = {"url": value, "alt": alt, "referer": base_url}
            blocks.append(("image", token))
        elif kind == "heading":
            blocks.append(("heading", value.upper()))
        else:
            blocks.append(("paragraph", value))

    if not blocks:
        text = container.get_text("\n")
        if text.strip():
            blocks.append(("paragraph", text))

    return blocks, image_info


def _download(url: str, referer: str | None = None, debug: bool = False) -> bytes:
    headers = {"User-Agent": UA}
    if referer:
        headers["Referer"] = referer
    req = Request(url, headers=headers)
    with urlopen(req, timeout=15) as response:
        data = response.read()
        if debug:
            print(f"[debug] downloaded {len(data)} bytes: {url}", file=sys.stderr)
        return data


async def fetch_image_bytes(url: str, referer: str | None = None, debug: bool = False) -> bytes:
    return await asyncio.to_thread(_download, url, referer, debug)


def image_bytes_to_ansi(
    image_bytes: bytes,
    width: int = MAX_IMAGE_WIDTH,
    debug: bool = False,
) -> str | None:
    with Image.open(io.BytesIO(image_bytes)) as img:
        if debug:
            print(f"[debug] image format={img.format} size={img.size}", file=sys.stderr)
        img = img.convert("RGB")
        w, h = img.size
        if w * h < MIN_IMAGE_AREA:
            if debug:
                print(f"[debug] image skipped small size={w}x{h}", file=sys.stderr)
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


async def build_image_ansi_map(image_info: dict[str, dict[str, str]], debug: bool = False) -> dict[str, str]:
    semaphore = asyncio.Semaphore(4)

    async def worker(token: str, info: dict[str, str]) -> tuple[str, str | None]:
        async with semaphore:
            try:
                image_bytes = await fetch_image_bytes(info["url"], info.get("referer"), debug)
                ansi_art = image_bytes_to_ansi(image_bytes, debug=debug)
            except Exception as exc:
                if debug:
                    print(f"[debug] image failed {info.get('url')}: {exc}", file=sys.stderr)
                return token, None

        if not ansi_art:
            if debug:
                print(f"[debug] image produced no render {info.get('url')}", file=sys.stderr)
            return token, None

        alt = info.get("alt", "")
        if alt:
            return token, f"[Image: {alt}]\n{ansi_art}"
        return token, ansi_art

    if debug:
        for token, info in image_info.items():
            print(f"[debug] image token {token}: {info.get('url')}", file=sys.stderr)
    tasks = [worker(token, info) for token, info in image_info.items()]
    results = await asyncio.gather(*tasks)
    return {token: art for token, art in results if art}


async def main() -> int:
    if len(sys.argv) < 2:
        print("usage: articat <url>")
        return 1

    debug = "--debug-images" in sys.argv
    args = [arg for arg in sys.argv[1:] if not arg.startswith("--")]
    if not args:
        print("usage: articat <url> [--debug-images]")
        return 1

    url = args[0]

    html = await run_with_spinner(fetch_html(url, debug), "Fetching page")

    full_soup = BeautifulSoup(html, "lxml")
    unwrap_noscript_images(full_soup)
    cdn_prefix = infer_cdn_prefix(full_soup)

    doc = Document(html, url=url)
    article_html = doc.summary(html_partial=True, keep_all_images=True)

    soup = BeautifulSoup(article_html, "lxml")

    unwrap_noscript_images(soup)
    strip_junk_nodes(soup)
    for tag in soup(["script", "style", "iframe"]):
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
    published = format_date(extract_published_date(full_soup))
    canonical_url = extract_canonical_url(full_soup, url)

    text = soup.get_text("\n")
    text_len = len(text.strip())
    summary_headings = count_headings(soup)
    summary_images = count_images(soup)
    full_container = full_soup.find("article") or full_soup.find("main") or full_soup
    full_headings = count_headings(full_container)
    full_images = count_images(full_container)
    use_fallback = (
        text_len < MIN_TEXT_CHARS
        or (full_headings > 1 and summary_headings < full_headings)
        or (full_images > summary_images)
    )

    fallback_blocks = []
    image_info = {}
    fallback_tokens = []
    if use_fallback:
        fallback_blocks, image_info = extract_fallback_blocks(full_soup, url, cdn_prefix)
        if not image_info:
            meta_image = find_meta_image(full_soup)
            resolved_meta = resolve_image_url(meta_image, url, cdn_prefix)
            if resolved_meta:
                token = "__IMG_FALLBACK_0__"
                image_info[token] = {"url": resolved_meta, "alt": "Top image", "referer": url}
                fallback_blocks.insert(0, ("image", token))
            else:
                article_tag = full_soup.find("article")
                if article_tag:
                    img = article_tag.find("img")
                    src = pick_image_src(img) if img else None
                    resolved = resolve_image_url(src, url, cdn_prefix)
                    if resolved and not resolved.startswith("data:"):
                        token = "__IMG_FALLBACK_0__"
                        image_info[token] = {
                            "url": resolved,
                            "alt": (img.get("alt") or "").strip(),
                            "referer": url,
                        }
                        fallback_blocks.insert(0, ("image", token))
    else:
        for i, img in enumerate(soup.find_all("img")):
            src = pick_image_src(img)
            if not src or src.startswith("data:"):
                img.decompose()
                continue
            alt = (img.get("alt") or "").strip()
            token = f"__IMG_{i}__"
            img.replace_with(soup.new_string(f"\n{token}\n"))
            resolved = resolve_image_url(src, url, cdn_prefix)
            if not resolved:
                continue
            image_info[token] = {"url": resolved, "alt": alt, "referer": url}

        if not image_info:
            meta_image = find_meta_image(full_soup)
            resolved_meta = resolve_image_url(meta_image, url, cdn_prefix)
            if resolved_meta:
                token = "__IMG_FALLBACK_0__"
                image_info[token] = {"url": resolved_meta, "alt": "Top image", "referer": url}
                fallback_tokens.append(token)
            else:
                article_tag = full_soup.find("article")
                if article_tag:
                    img = article_tag.find("img")
                    src = pick_image_src(img) if img else None
                    resolved = resolve_image_url(src, url, cdn_prefix)
                    if resolved and not resolved.startswith("data:"):
                        token = "__IMG_FALLBACK_0__"
                        image_info[token] = {
                            "url": resolved,
                            "alt": (img.get("alt") or "").strip(),
                            "referer": url,
                        }
                        fallback_tokens.append(token)

        text = soup.get_text("\n")

    image_ansi_map = (
        await run_with_spinner(build_image_ansi_map(image_info, debug), "Rendering images")
        if image_info
        else {}
    )
    if not image_ansi_map:
        hero = find_meta_image(full_soup)
        hero = resolve_image_url(hero, url, cdn_prefix)
        if not hero:
            article_tag = full_soup.find("article") or full_soup.find("main")
            if article_tag:
                hero = resolve_image_url(
                    pick_image_src(article_tag.find("img") or article_tag.find("picture")),
                    url,
                    cdn_prefix,
                )
        if hero:
            token = "__IMG_HERO__"
            image_info = {token: {"url": hero, "alt": "Top image", "referer": url}}
            image_ansi_map = await run_with_spinner(
                build_image_ansi_map(image_info, debug),
                "Rendering images",
            )

    paragraphs = []
    current = []
    if use_fallback:
        for kind, value in fallback_blocks:
            if kind == "heading":
                if current:
                    paragraphs.append(" ".join(current))
                    current = []
                paragraphs.append(value)
            elif kind == "image":
                if current:
                    paragraphs.append(" ".join(current))
                    current = []
                paragraphs.append(value)
            else:
                if value:
                    paragraphs.append(value)
    else:
        lines = [line.strip() for line in text.splitlines()]
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

    hero_token = "__IMG_HERO__"
    if hero_token in image_info and hero_token not in paragraphs:
        paragraphs.insert(0, hero_token)

    normalized = "\n\n".join(paragraphs)
    if fallback_tokens:
        normalized = "\n\n".join(fallback_tokens) + "\n\n" + normalized
    for token in image_info:
        normalized = normalized.replace(token, image_ansi_map.get(token, ""))

    normalized = normalized.strip("\n")
    content_for_reading = "\n\n".join(p for p in paragraphs if not p.startswith("__IMG_"))
    words = re.findall(r"[A-Za-z0-9']+", content_for_reading)
    read_minutes = math.ceil(len(words) / 200) if words else None

    header_lines = []
    header_lines.append(title or "Unknown title")
    byline_parts = [part for part in [author, publication] if part]
    if byline_parts:
        header_lines.append(" | ".join(byline_parts))
    meta_parts = []
    if published:
        meta_parts.append(f"Published: {published}")
    if read_minutes:
        meta_parts.append(f"Read time: {read_minutes} min")
    if meta_parts:
        header_lines.append(" | ".join(meta_parts))
    if canonical_url:
        header_lines.append(f"URL: {canonical_url}")

    print("\n\n")
    print("\n".join(header_lines))
    if normalized:
        print()
        print(normalized)
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
