# Importing all the required libraries
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

from shared_utils import (
    crawl_urls,
    corpus_stats,
    deliverables_dir,
    ensure_dirs,
    fetch_url,
    generate_wordcloud,
    load_json,
    process_document,
    read_lines,
    root_dir,
    save_raw,
    write_json,
)


class CorpusAssembler:

    def __init__(self, proj_root: Path, config: Dict):  # instializing necessary variables
        self.proj_root = proj_root
        self.config = config
        self.raw_dir = self.proj_root / "raw_downloads"
        self.output_dir = self.proj_root / "deliverables" / "task1"
        self.corpus_path = self.output_dir / "clean_corpus.txt"

    def build(self) -> Dict:
        ensure_dirs([self.raw_dir, self.output_dir, self.corpus_path.parent])

        initial_urls = read_lines(self.proj_root / self.config["sources_rel"])  # reading the seed URLs from sources.txt
        discovered = crawl_urls(initial_urls, max_depth=self.config["crawl_depth"], max_pages=self.config["max_pages"])
        # explore more
        unique_urls = [url for url, _ in discovered]  # getting all the unique URLs from the explored URLs

        tokenized_docs, doc_metadata, url_depth = self._retrieve_and_process(unique_urls, discovered)

        if not tokenized_docs:
            raise RuntimeError("No documents processed. Check sources.txt for valid URLs.")

        self._write_corpus(tokenized_docs)  # writing the cleaned and tokenized corpus to clean_corpus.txt
        stats = self._generate_statistics(tokenized_docs, doc_metadata, url_depth, len(initial_urls), len(unique_urls))
        # generating the statistics for the corpus and saving it to dataset_stats.json, also generating the wordcloud and summary text file for the corpus
        self._generate_artifacts(tokenized_docs, stats)  # generating the artifacts for the corpus assembly stage, including stats, wordcloud, and summary

        return stats

    def _retrieve_and_process(self, urls: List[str], depth_map: List[Tuple[str, int]]) -> Tuple[List[List[str]], List[Dict], Dict]:
        tokenized_docs = []
        doc_metadata = []
        depth_lookup = {url: d for url, d in depth_map}

        for url in urls:
            try:
                content = fetch_url(url)  # fetch content from the URL webpage
                raw_path = save_raw(self.raw_dir, url, content)
                extracted_text, word_tokens = process_document(url, content)  # saving all the extracted text and tokens

                if not word_tokens:
                    continue

                tokenized_docs.append(word_tokens)
                doc_metadata.append({
                    "source_url": url,
                    "crawl_level": depth_lookup.get(url, 0),
                    "raw_file": str(raw_path.relative_to(self.proj_root)),
                    "token_count": len(word_tokens),
                })
                print(f"✓ Processed: {url} ({len(word_tokens)} tokens)")
            except Exception as err:
                print(f"✗ Skipped: {url} ({err})")

        return tokenized_docs, doc_metadata, depth_lookup

    def _write_corpus(self, tokenized_docs: List[List[str]]) -> None:
        with self.corpus_path.open("w", encoding="utf-8") as fp:
            for tokens in tokenized_docs:
                fp.write(" ".join(tokens) + "\n")

    def _generate_statistics(self, tokenized_docs: List[List[str]], metadata: List[Dict], 
                            depth_map: Dict, seed_count: int, discovered_count: int) -> Dict:
        stats = corpus_stats(tokenized_docs)
        stats.update({
            "sourced_documents": len(metadata),
            "seed_sources": seed_count,
            "crawl_depth": self.config["crawl_depth"],
            "max_pages": self.config["max_pages"],
            "unique_urls": discovered_count,
            "documents": metadata,
        })
        return stats

    def _generate_artifacts(self, tokenized_docs: List[List[str]], stats: Dict) -> None:
        # Statistics JSON
        stats_path = self.output_dir / "dataset_stats.json"
        write_json(stats, stats_path)

        # Word frequency visualization
        term_freq = Counter(term for doc in tokenized_docs for term in doc)
        cloud_path = self.output_dir / "wordcloud_task1.png"
        generate_wordcloud(term_freq, cloud_path)

        # Human-readable summary
        summary_path = self.output_dir / "task1_summary.txt"
        self._write_summary(summary_path, stats)

    def _write_summary(self, path: Path, stats: Dict) -> None:
        """Write human-readable corpus summary."""
        with path.open("w", encoding="utf-8") as fp:
            fp.write("CORPUS ASSEMBLY SUMMARY\n")
            fp.write(f"Total documents: {stats['total_documents']}\n")
            fp.write(f"Total tokens: {stats['total_tokens']}\n")
            fp.write(f"Vocabulary size: {stats['vocabulary_size']}\n")
            fp.write(f"Sources processed: {stats['sourced_documents']}\n")
            fp.write(f"Seed URLs: {stats['seed_sources']}\n")
            fp.write(f"Discovered URLs: {stats['unique_urls']}\n")
            fp.write(f"Crawl depth: {stats['crawl_depth']}\n")
            fp.write("Most frequent terms:\n")
            for term, count in stats["top_30_words"][:20]:
                fp.write(f"  {term}: {count}\n")


def task1_prepare_dataset(sources_rel: str = "sources.txt", crawl_depth: int = 2, max_pages: int = 120) -> Dict:
    root = root_dir()
    config = {
        "sources_rel": sources_rel,
        "crawl_depth": crawl_depth,
        "max_pages": max_pages,
    }  # Takes the configuration for the corpus assembly stage, including source file, crawl depth, and page limit
    assembler = CorpusAssembler(root, config)  # calling corpus assembler instance
    return assembler.build()  # calling the main .build() method


if __name__ == "__main__":
    task1_prepare_dataset()
