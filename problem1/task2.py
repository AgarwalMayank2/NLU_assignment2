import itertools
import time
from pathlib import Path
from typing import Dict, List

import pandas as pd

from shared_utils import (
    deliverables_dir,
    ensure_dirs,
    load_corpus,
    models_dir,
    root_dir,
    train_scratch_word2vec,
    write_json,
)


class HyperparameterExperiment:

    def __init__(self, architecture: str, vector_size: int, window: int, negative: int, epochs: int, min_count: int):
        self.architecture = architecture
        self.vector_size = vector_size
        self.window = window
        self.negative = negative
        self.epochs = epochs
        self.min_count = min_count
        self.model = None
        self.elapsed_time = 0

    def train(self, corpus: List[List[str]]) -> None:  # Train the Word2Vec model with the specified hyperparameters and measure training time.
        start_time = time.time()
        self.model = train_scratch_word2vec(
            corpus=corpus,
            model_type=self.architecture,
            vector_size=self.vector_size,
            window=self.window,
            negative=self.negative,
            epochs=self.epochs,
            min_count=self.min_count,
            seed=42,
        )
        self.elapsed_time = time.time() - start_time

    def save(self, output_path: Path) -> None:
        self.model.save(str(output_path))

    def to_record(self, model_path: Path) -> Dict:
        return {
            "model_type": self.architecture,
            "vector_size": self.vector_size,
            "window": self.window,
            "negative": self.negative,
            "epochs": self.epochs,
            "vocab_size": len(self.model.wv.key_to_index),
            "train_seconds": round(self.elapsed_time, 3),
            "model_path": str(model_path),
        }


class EmbbeddingExperimenter:

    def __init__(self, corpus: List[List[str]], config: Dict, models_output: Path):
        self.corpus = corpus
        self.config = config
        self.models_dir = models_output
        self.experiment_results = []

    def execute_grid_search(self) -> List[Dict]:  # Run experiments across all hyperparameter combinations and architectures.
        for architecture in ["cbow", "skipgram"]:
            self._run_architecture_search(architecture)
        return self.experiment_results

    def _run_architecture_search(self, architecture: str) -> None:  # Train models for a specific architecture across all hyperparameter combinations.
        param_combinations = itertools.product(
            self.config["dimensions"],
            self.config["windows"],
            self.config["negatives"],
        )

        for vec_dim, ctx_window, neg_samples in param_combinations:
            exp = HyperparameterExperiment(
                architecture=architecture,
                vector_size=vec_dim,
                window=ctx_window,
                negative=neg_samples,
                epochs=self.config["epochs"],
                min_count=self.config["min_count"],
            )  # Initialize experiment with current hyperparameters

            print(f"Training {architecture}_dim{vec_dim}_win{ctx_window}_neg{neg_samples}...", end=" ")
            exp.train(self.corpus)

            model_filename = f"{architecture}_dim{vec_dim}_win{ctx_window}_neg{neg_samples}.model"
            save_location = self.models_dir / model_filename
            exp.save(save_location)

            record = exp.to_record(save_location)
            self.experiment_results.append(record)
            print(f"✓ ({exp.elapsed_time:.2f}s)")

    def select_best(self, architecture: str) -> Dict:  # Select the best model for a given architecture based on vocab size and training time.
        arch_results = [r for r in self.experiment_results if r["model_type"] == architecture]
        ranked = sorted(arch_results, key=lambda r: (-r["vocab_size"], r["train_seconds"]))
        return ranked[0]


def task2_train_word2vec(
    corpus_rel: str = "deliverables/task1/clean_corpus.txt",
    dimensions: List[int] = None,
    windows: List[int] = None,
    negatives: List[int] = None,
    epochs: int = 20,
    min_count: int = 2,
) -> Dict:
    dimensions = dimensions or [100, 200, 300]
    windows = windows or [3, 5, 8]
    negatives = negatives or [5, 10]

    root = root_dir()
    corpus_location = root / corpus_rel
    model_storage = models_dir()
    task_dir = deliverables_dir() / "task2"
    ensure_dirs([model_storage, task_dir])  # Ensure output directories exist

    corpus = load_corpus(corpus_location)  # Load the cleaned corpus from stage 1 output
    if not corpus:
        raise RuntimeError("Corpus is empty. Run stage 1 first.")

    config = {
        "dimensions": dimensions,
        "windows": windows,
        "negatives": negatives,
        "epochs": epochs,
        "min_count": min_count,
    }

    experimenter = EmbbeddingExperimenter(corpus, config, model_storage)  # Initialize experimenter with corpus and hyperparameter config
    all_results = experimenter.execute_grid_search()  # Run the grid search over all hyperparameter combinations and collect results

    # Save detailed results
    results_df = pd.DataFrame(all_results)
    results_csv = task_dir / "task2_hyperparameter_results.csv"
    results_df.to_csv(results_csv, index=False)  # Save the hyperparameter tuning results to a CSV file for analysis

    # Save best models config
    best_selections = {
        "cbow": experimenter.select_best("cbow"),
        "skipgram": experimenter.select_best("skipgram"),
    }
    best_json = task_dir / "task2_best_models.json"
    write_json(best_selections, best_json)

    print(f"\n✓ Results table: {results_csv}")
    print(f"✓ Best model config: {best_json}")
    return best_selections


if __name__ == "__main__":
    task2_train_word2vec()
