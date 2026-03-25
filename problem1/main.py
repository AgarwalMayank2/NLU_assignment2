import argparse
from typing import List, Dict, Any

from task1 import task1_prepare_dataset
from task2 import task2_train_word2vec
from task3 import task3_semantic_analysis
from task4 import task4_visualization


class PipelineConfig:

    def __init__(
        self,
        sources: str = "sources.txt",
        viz_method: str = "pca",
        dim_sizes: List[int] = None,
        win_sizes: List[int] = None,
        neg_counts: List[int] = None,
        num_epochs: int = 20,
        vocab_threshold: int = 2,
        depth: int = 2,
        page_limit: int = 120,
    ):
        self.sources = sources
        self.viz_method = viz_method
        self.dim_sizes = dim_sizes or [100, 200, 300]
        self.win_sizes = win_sizes or [3, 5, 8]
        self.neg_counts = neg_counts or [5, 10]
        self.num_epochs = num_epochs
        self.vocab_threshold = vocab_threshold
        self.depth = depth
        self.page_limit = page_limit


class WordEmbeddingPipeline:
    def __init__(self, config: PipelineConfig):
        self.config = config

    def execute(self) -> Dict[str, Any]:  # Execute all pipeline stages in sequence.
        results = {}

        self._print_stage("DATA ACQUISITION: Web Crawling, Downloading & Processing")
        results["stage1"] = task1_prepare_dataset(
            sources_rel=self.config.sources,
            crawl_depth=self.config.depth,
            max_pages=self.config.page_limit,
        )  # Prepare the dataset by crawling web sources, processing documents, and generating statistics and artifacts.

        self._print_stage("MODEL TRAINING: CBOW & Skip-gram with Hyperparameter Tuning")
        results["stage2"] = task2_train_word2vec(
            corpus_rel="deliverables/task1/clean_corpus.txt",
            dimensions=self.config.dim_sizes,
            windows=self.config.win_sizes,
            negatives=self.config.neg_counts,
            epochs=self.config.num_epochs,
            min_count=self.config.vocab_threshold,
        )  # Train Word2Vec models (CBOW and Skip-gram) across a grid of hyperparameters and select the best configurations.

        self._print_stage("SEMANTIC ANALYSIS: Nearest Neighbors & Analogies")
        results["stage3"] = task3_semantic_analysis(best_models_rel="deliverables/task2/task2_best_models.json")
        # Perform semantic evaluation of the best models by analyzing nearest neighbors and solving analogy tasks, compiling results into structured formats and human-readable reports.

        self._print_stage("DIMENSIONALITY REDUCTION: Embedding Visualization")
        task4_visualization(best_models_rel="deliverables/task2/task2_best_models.json", method=self.config.viz_method)
        # Visualize the embedding spaces of the best models using dimensionality reduction techniques (PCA or t-SNE) and save the resulting plots.

        self._print_completion()
        return results

    @staticmethod
    def _print_stage(label: str) -> None:
        """Pretty-print stage header."""
        print("\n" + "=" * 70)
        print(label)
        print("=" * 70)

    @staticmethod
    def _print_completion() -> None:
        """Print completion message."""
        print("\n" + "=" * 70)
        print("✓ PIPELINE EXECUTION COMPLETED")
        print("=" * 70)
        print("Results are available in the 'deliverables/' directory.")


def _build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Word2Vec Embedding Pipeline: Multi-stage Execution")

    parser.add_argument("--sources", default="sources.txt", help="Sources file location")
    parser.add_argument("--crawl-depth", type=int, default=2, help="Maximum depth for web crawling")
    parser.add_argument("--max-pages", type=int, default=120, help="Upper limit on pages to retrieve")
    parser.add_argument("--viz", choices=["pca", "tsne"], default="pca", help="Dimensionality reduction technique")
    parser.add_argument("--dimensions", nargs="+", type=int, default=[100, 200, 300], help="Vector sizes to test")
    parser.add_argument("--windows", nargs="+", type=int, default=[3, 5, 8], help="Context window sizes to test")
    parser.add_argument("--negatives", nargs="+", type=int, default=[5, 10], help="Negative sampling counts to test")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training iterations")
    parser.add_argument("--min-count", type=int, default=2, help="Minimum frequency threshold for vocabulary")

    return parser


def main():
    parser = _build_cli_parser()  # Build the command-line argument parser
    args = parser.parse_args()  # Parse the command-line arguments into a structured format

    pipeline_settings = PipelineConfig(
        sources=args.sources,
        viz_method=args.viz,
        dim_sizes=args.dimensions,
        win_sizes=args.windows,
        neg_counts=args.negatives,
        num_epochs=args.epochs,
        vocab_threshold=args.min_count,
        depth=args.crawl_depth,
        page_limit=args.max_pages,
    )  # Create a PipelineConfig instance using the parsed command-line arguments

    pipeline = WordEmbeddingPipeline(pipeline_settings)  # Initialize the WordEmbeddingPipeline with the specified configuration
    pipeline.execute()


if __name__ == "__main__":
    main()
