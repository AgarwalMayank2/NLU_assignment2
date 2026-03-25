"""Shared utilities for Word2Vec assignment tasks."""

import hashlib
import json
import re
from collections import Counter
from io import BytesIO
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple
from urllib.parse import urldefrag, urljoin, urlparse

import matplotlib.pyplot as plt
import numpy as np
import requests
from bs4 import BeautifulSoup
from PyPDF2 import PdfReader
from wordcloud import WordCloud

# Constants
HTML_TAGS_TO_REMOVE = ["script", "style", "noscript", "svg", "footer", "header", "nav"]
BOILERPLATE_PATTERNS = [
    r"privacy policy",
    r"copyright",
    r"all rights reserved",
    r"skip to content",
    r"click here",
    r"cookie",
    r"terms and conditions",
]

TARGET_WORDS = ["research", "student", "phd", "exam"]
ANALOGIES = [
    ("ug", "btech", "pg"),
    ("btech", "undergraduate", "mtech"),
    ("research", "lab", "teaching"),
]
SEED_WORDS = [
    "research",
    "student",
    "phd",
    "exam",
    "faculty",
    "department",
    "course",
    "lab",
    "project",
    "admission",
    "curriculum",
    "semester",
    "btech",
    "mtech",
    "undergraduate",
    "postgraduate",
]

NON_TEXT_RE = re.compile(r"[^a-zA-Z\s]")
MULTISPACE_RE = re.compile(r"\s+")
MULTISPACE_INLINE_RE = re.compile(r"[ \t]+")


# Directory helpers
def root_dir() -> Path:
    """Get project root directory."""
    return Path(__file__).resolve().parent


def deliverables_dir() -> Path:
    """Get deliverables directory."""
    return root_dir() / "deliverables"


def models_dir() -> Path:
    """Get models (non-deliverables) directory."""
    return root_dir() / "models"


def raw_downloads_dir() -> Path:
    """Get raw downloads directory."""
    return root_dir() / "raw_downloads"


# File I/O helpers
def ensure_dirs(paths: Iterable[Path]) -> None:
    """Create directories if they don't exist."""
    for p in paths:
        p.mkdir(parents=True, exist_ok=True)


def read_lines(path: Path) -> List[str]:
    """Read non-empty, non-comment lines from a file."""
    with path.open("r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip() and not line.strip().startswith("#")]


def write_json(data: dict, path: Path) -> None:
    """Write a dict to JSON file."""
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def load_json(path: Path) -> dict:
    """Load a JSON file."""
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


# Text processing
def is_probably_english(text: str, threshold: float = 0.65) -> bool:
    """Check if text is probably English based on ASCII letter ratio."""
    if not text.strip():
        return False
    letters = sum(ch.isalpha() for ch in text)
    ascii_letters = sum(("a" <= ch.lower() <= "z") for ch in text if ch.isalpha())
    if letters == 0:
        return False
    return (ascii_letters / letters) >= threshold


def normalize_text(text: str) -> str:
    """Normalize whitespace while preserving line boundaries."""
    text = text.replace("\u00a0", " ").replace("\t", " ").replace("\r", "\n")
    lines = [MULTISPACE_INLINE_RE.sub(" ", line).strip() for line in text.splitlines()]
    return "\n".join(line for line in lines if line)


def tokenize_clean(text: str) -> List[str]:
    """Tokenize and clean text (lowercase, remove non-letters)."""
    text = text.lower()
    text = NON_TEXT_RE.sub(" ", text)
    text = MULTISPACE_RE.sub(" ", text).strip()
    if not text:
        return []
    return [tok for tok in text.split(" ") if len(tok) > 1]


def corpus_stats(doc_tokens: List[List[str]]) -> dict:
    """Compute corpus statistics."""
    token_count = sum(len(doc) for doc in doc_tokens)
    vocab = set(tok for doc in doc_tokens for tok in doc)
    freq = Counter(tok for doc in doc_tokens for tok in doc)
    return {
        "total_documents": len(doc_tokens),
        "total_tokens": token_count,
        "vocabulary_size": len(vocab),
        "top_30_words": freq.most_common(30),
    }


def safe_filename(text: str) -> str:
    """Convert text to a safe filename."""
    return re.sub(r"[^a-zA-Z0-9._-]", "_", text)


# URL and web helpers
def fetch_url(url: str, timeout: int = 25) -> bytes:
    """Fetch URL content."""
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    return response.content


def normalize_url(url: str) -> str:
    """Normalize URL by removing fragment and trailing slash."""
    clean, _ = urldefrag(url)
    return clean.rstrip("/")


def is_html_url(url: str) -> bool:
    """Check if URL is likely HTML (not PDF)."""
    low = url.lower()
    if low.endswith(".pdf"):
        return False
    return True


def same_allowed_domain(url: str, allowed_hosts: List[str]) -> bool:
    """Check if URL belongs to allowed domain."""
    host = (urlparse(url).hostname or "").lower()
    if not host:
        return False
    return any(host == allowed or host.endswith(f".{allowed}") for allowed in allowed_hosts)


def extract_links_from_html(content: bytes, base_url: str, allowed_hosts: List[str]) -> List[str]:
    """Extract and normalize links from HTML content."""
    soup = BeautifulSoup(content, "html.parser")
    links: List[str] = []
    for anchor in soup.find_all("a", href=True):
        raw = anchor.get("href", "").strip()
        if not raw:
            continue
        absolute = normalize_url(urljoin(base_url, raw))
        parsed = urlparse(absolute)
        if parsed.scheme not in {"http", "https"}:
            continue
        if not same_allowed_domain(absolute, allowed_hosts):
            continue
        links.append(absolute)
    return links


def extract_html_text(content: bytes) -> str:
    """Extract text from HTML content."""
    soup = BeautifulSoup(content, "html.parser")
    for tag in HTML_TAGS_TO_REMOVE:
        for item in soup.find_all(tag):
            item.decompose()
    text = soup.get_text("\n")
    return normalize_text(text)


def extract_pdf_text(content: bytes) -> str:
    """Extract text from PDF content."""
    reader = PdfReader(BytesIO(content))
    chunks = []
    for page in reader.pages:
        try:
            chunks.append(page.extract_text() or "")
        except Exception:
            continue
    return normalize_text("\n".join(chunks))


def crawl_urls(seed_urls: List[str], max_depth: int, max_pages: int) -> List[Tuple[str, int]]:
    """Crawl URLs up to max_depth and max_pages."""
    normalized_seeds = [normalize_url(url) for url in seed_urls]
    allowed_hosts = sorted(
        {
            (urlparse(url).hostname or "").lower()
            for url in normalized_seeds
            if (urlparse(url).hostname or "")
        }
    )

    stack: List[Tuple[str, int]] = [(url, 0) for url in normalized_seeds]
    visited: set = set()
    crawled: List[Tuple[str, int]] = []

    while stack and len(crawled) < max_pages:
        current_url, depth = stack.pop()
        if current_url in visited:
            continue
        visited.add(current_url)
        crawled.append((current_url, depth))

        if depth >= max_depth or not is_html_url(current_url):
            continue

        try:
            content = fetch_url(current_url)
            child_links = extract_links_from_html(content, current_url, allowed_hosts)
        except Exception:
            continue

        for link in reversed(child_links):
            if link not in visited:
                stack.append((link, depth + 1))

    return crawled


def load_corpus(path: Path) -> List[List[str]]:
    """Load tokenized corpus from file."""
    docs = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            tokens = line.strip().split()
            if tokens:
                docs.append(tokens)
    return docs


def clean_boilerplate(text: str) -> str:
    """Remove boilerplate patterns and low-quality lines."""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    cleaned = []
    for line in lines:
        low = line.lower()
        if any(re.search(pattern, low) for pattern in BOILERPLATE_PATTERNS):
            continue
        letters = sum(ch.isalpha() for ch in line)
        symbols = sum(not ch.isalnum() and not ch.isspace() for ch in line)
        if letters < 5 and symbols > letters:
            continue
        cleaned.append(line)
    return "\n".join(cleaned)


def process_document(url: str, content: bytes) -> Tuple[str, List[str]]:
    """Process a document (HTML or PDF) and extract tokens."""
    text = extract_pdf_text(content) if url.lower().endswith(".pdf") else extract_html_text(content)
    text = clean_boilerplate(text)
    english_lines = [line for line in text.splitlines() if is_probably_english(line)]
    merged = " ".join(english_lines) if english_lines else text
    tokens = tokenize_clean(merged)
    return merged, tokens


def save_raw(raw_dir: Path, url: str, content: bytes) -> Path:
    """Save raw content to file."""
    digest = hashlib.md5(url.encode("utf-8")).hexdigest()[:8]
    suffix = ".pdf" if url.lower().endswith(".pdf") else ".html"
    name = safe_filename(url.split("//", 1)[-1])[:80]
    path = raw_dir / f"{name}_{digest}{suffix}"
    with path.open("wb") as f:
        f.write(content)
    return path


def generate_wordcloud(freq: Counter, out_path: Path) -> None:
    """Generate and save wordcloud visualization."""
    cloud = WordCloud(width=1400, height=800, background_color="white", colormap="viridis")
    cloud.generate_from_frequencies(freq)
    plt.figure(figsize=(14, 8))
    plt.imshow(cloud, interpolation="bilinear")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()





# From-scratch Word2Vec implementation
def stable_sigmoid(x: float) -> float:
    """Numerically stable sigmoid."""
    x = float(np.clip(x, -10.0, 10.0))
    return 1.0 / (1.0 + np.exp(-x))


class ScratchKeyedVectors:
    """Word vector storage with similarity search."""

    def __init__(self, index_to_key: List[str], vectors: np.ndarray):
        self.index_to_key = index_to_key
        self.key_to_index = {w: i for i, w in enumerate(index_to_key)}
        self.vectors = vectors.astype(np.float32, copy=False)
        self._norm_vectors: Optional[np.ndarray] = None

    def __getitem__(self, word: str) -> np.ndarray:
        return self.vectors[self.key_to_index[word]]

    def _normalized_vectors(self) -> np.ndarray:
        if self._norm_vectors is None:
            norms = np.linalg.norm(self.vectors, axis=1, keepdims=True) + 1e-12
            self._norm_vectors = self.vectors / norms
        return self._norm_vectors

    def most_similar(
        self,
        word: Optional[str] = None,
        topn: int = 10,
        positive: Optional[List[str]] = None,
        negative: Optional[List[str]] = None,
    ) -> List[Tuple[str, float]]:
        """Find most similar words (supports word arithmetic)."""
        if word is not None:
            positive = [word]
        positive = positive or []
        negative = negative or []
        if not positive and not negative:
            return []

        for token in positive + negative:
            if token not in self.key_to_index:
                return []

        query = np.zeros(self.vectors.shape[1], dtype=np.float32)
        for token in positive:
            query += self[token]
        for token in negative:
            query -= self[token]

        q_norm = np.linalg.norm(query)
        if q_norm < 1e-12:
            return []
        query = query / q_norm

        sims = self._normalized_vectors() @ query
        banned = set(positive + negative)
        ranked = np.argsort(-sims)
        out: List[Tuple[str, float]] = []
        for idx in ranked:
            token = self.index_to_key[idx]
            if token in banned:
                continue
            out.append((token, float(sims[idx])))
            if len(out) >= topn:
                break
        return out


class ScratchWord2Vec:
    """From-scratch Word2Vec model (CBOW and Skip-gram)."""

    def __init__(self, vector_size: int, window: int, negative: int, sg: int, epochs: int, min_count: int):
        self.vector_size = vector_size
        self.window = window
        self.negative = negative
        self.sg = sg
        self.epochs = epochs
        self.min_count = min_count
        self.wv: Optional[ScratchKeyedVectors] = None

    def save(self, path: str) -> None:
        """Save model to npz file."""
        if self.wv is None:
            raise RuntimeError("Model is not trained yet.")
        payload = {
            "vector_size": np.array([self.vector_size], dtype=np.int32),
            "window": np.array([self.window], dtype=np.int32),
            "negative": np.array([self.negative], dtype=np.int32),
            "sg": np.array([self.sg], dtype=np.int32),
            "epochs": np.array([self.epochs], dtype=np.int32),
            "min_count": np.array([self.min_count], dtype=np.int32),
            "index_to_key": np.array(self.wv.index_to_key, dtype=object),
            "vectors": self.wv.vectors,
        }
        with open(path, "wb") as f:
            np.savez_compressed(f, **payload)

    @classmethod
    def load(cls, path: str) -> "ScratchWord2Vec":
        """Load model from npz file."""
        with np.load(path, allow_pickle=True) as data:
            model = cls(
                vector_size=int(data["vector_size"][0]),
                window=int(data["window"][0]),
                negative=int(data["negative"][0]),
                sg=int(data["sg"][0]),
                epochs=int(data["epochs"][0]),
                min_count=int(data["min_count"][0]),
            )
            index_to_key = data["index_to_key"].tolist()
            vectors = data["vectors"].astype(np.float32)
            model.wv = ScratchKeyedVectors(index_to_key=index_to_key, vectors=vectors)
            return model


def build_vocab(corpus: List[List[str]], min_count: int) -> Tuple[List[str], Dict[str, int]]:
    """Build vocabulary with min_count threshold."""
    freq = Counter(tok for sent in corpus for tok in sent)
    index_to_key = [w for w, c in freq.items() if c >= min_count]
    index_to_key.sort()
    key_to_index = {w: i for i, w in enumerate(index_to_key)}
    return index_to_key, key_to_index


def corpus_to_indices(corpus: List[List[str]], key_to_index: Dict[str, int]) -> List[List[int]]:
    """Convert corpus tokens to word indices."""
    encoded = []
    for sent in corpus:
        row = [key_to_index[w] for w in sent if w in key_to_index]
        if len(row) >= 2:
            encoded.append(row)
    return encoded


def negative_sampling_probs(index_to_key: List[str], corpus: List[List[str]]) -> np.ndarray:
    """Compute unigram^0.75 probability distribution for negative sampling."""
    vocab_set = set(index_to_key)
    freq = Counter(tok for sent in corpus for tok in sent if tok in vocab_set)
    weights = np.array([freq[w] ** 0.75 for w in index_to_key], dtype=np.float64)
    denom = weights.sum()
    if denom == 0:
        return np.full(len(index_to_key), 1.0 / max(1, len(index_to_key)), dtype=np.float64)
    return weights / denom


def train_scratch_word2vec(
    corpus: List[List[str]],
    model_type: str,
    vector_size: int,
    window: int,
    negative: int,
    epochs: int,
    min_count: int,
    seed: int = 42,
) -> ScratchWord2Vec:
    """Train from-scratch CBOW or Skip-gram model."""
    sg = 0 if model_type.lower() == "cbow" else 1
    model = ScratchWord2Vec(
        vector_size=vector_size,
        window=window,
        negative=negative,
        sg=sg,
        epochs=epochs,
        min_count=min_count,
    )

    index_to_key, key_to_index = build_vocab(corpus, min_count=min_count)
    if not index_to_key:
        raise RuntimeError("No vocabulary after min_count filtering. Consider lowering --min-count.")

    encoded = corpus_to_indices(corpus, key_to_index)
    if not encoded:
        raise RuntimeError("No trainable sentences after vocabulary filtering.")

    rng = np.random.default_rng(seed)
    vocab_size = len(index_to_key)
    in_vectors = (rng.random((vocab_size, vector_size), dtype=np.float32) - 0.5) / vector_size
    out_vectors = np.zeros((vocab_size, vector_size), dtype=np.float32)
    neg_probs = negative_sampling_probs(index_to_key, corpus)

    total_centers = epochs * sum(len(sent) for sent in encoded)
    step_count = 0
    start_alpha = 0.025
    min_alpha = 0.0001

    def current_alpha() -> float:
        frac = step_count / max(1, total_centers)
        return max(min_alpha, start_alpha * (1.0 - frac))

    for _ in range(epochs):
        rng.shuffle(encoded)
        for sent in encoded:
            n = len(sent)
            for center_pos, center_idx in enumerate(sent):
                dyn_window = int(rng.integers(1, window + 1))
                left = max(0, center_pos - dyn_window)
                right = min(n, center_pos + dyn_window + 1)
                context = [sent[i] for i in range(left, right) if i != center_pos]
                if not context:
                    step_count += 1
                    continue

                lr = current_alpha()

                if sg == 0:
                    # CBOW: predict center word from mean context vector.
                    hidden = in_vectors[context].mean(axis=0)
                    grad_hidden = np.zeros(vector_size, dtype=np.float32)
                    targets = [center_idx]
                    labels = [1.0]
                    neg_samples = rng.choice(vocab_size, size=negative, p=neg_probs)
                    for neg_idx in neg_samples:
                        if int(neg_idx) == center_idx:
                            continue
                        targets.append(int(neg_idx))
                        labels.append(0.0)

                    for target_idx, label in zip(targets, labels):
                        score = float(np.dot(hidden, out_vectors[target_idx]))
                        pred = stable_sigmoid(score)
                        g = lr * (label - pred)
                        grad_hidden += g * out_vectors[target_idx]
                        out_vectors[target_idx] += g * hidden

                    shared_grad = grad_hidden / len(context)
                    for ctx_idx in context:
                        in_vectors[ctx_idx] += shared_grad
                else:
                    # Skip-gram: predict each context word from center word.
                    for target_word in context:
                        hidden = in_vectors[center_idx]
                        grad_in = np.zeros(vector_size, dtype=np.float32)
                        targets = [target_word]
                        labels = [1.0]
                        neg_samples = rng.choice(vocab_size, size=negative, p=neg_probs)
                        for neg_idx in neg_samples:
                            if int(neg_idx) == target_word:
                                continue
                            targets.append(int(neg_idx))
                            labels.append(0.0)

                        for target_idx, label in zip(targets, labels):
                            score = float(np.dot(hidden, out_vectors[target_idx]))
                            pred = stable_sigmoid(score)
                            g = lr * (label - pred)
                            grad_in += g * out_vectors[target_idx]
                            out_vectors[target_idx] += g * hidden

                        in_vectors[center_idx] += grad_in

                step_count += 1

    model.wv = ScratchKeyedVectors(index_to_key=index_to_key, vectors=in_vectors)
    return model
