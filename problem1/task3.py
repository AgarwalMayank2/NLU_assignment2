from pathlib import Path
from typing import Dict, List, Tuple

from shared_utils import (
    ANALOGIES,
    TARGET_WORDS,
    ScratchWord2Vec,
    deliverables_dir,
    ensure_dirs,
    load_json,
    root_dir,
    write_json,
)


class EmbeddingEvaluator:

    def __init__(self, model_location: Path, eval_targets: List[str], test_triplets: List[Tuple[str, str, str]]):
        self.model = ScratchWord2Vec.load(str(model_location))
        self.model_path = model_location
        self.eval_targets = eval_targets
        self.test_triplets = test_triplets

    def compute_similarities(self, query_word: str, neighbor_count: int = 5) -> List[Tuple[str, float]]:  # Find semantically similar words using cosine distance.
        if query_word not in self.model.wv.key_to_index:
            return []
        return self.model.wv.most_similar(query_word, topn=neighbor_count)

    def test_analogies(self, source: str, intermediate: str, query: str, candidate_count: int = 5) -> List[Tuple[str, float]]:
        # Test analogy relationships using vector arithmetic (e.g., "
        for word in (source, intermediate, query):
            if word not in self.model.wv.key_to_index:
                return []
        return self.model.wv.most_similar(positive=[query, intermediate], negative=[source], topn=candidate_count)

    def full_evaluation(self) -> Dict:  # Perform the full semantic evaluation, including similarity and analogy tests, and compile results into a structured format.
        similarity_results = {}
        for target in self.eval_targets:
            similarity_results[target] = self.compute_similarities(target, neighbor_count=5)

        analogy_results = {}
        for src, med, qry in self.test_triplets:
            analogy_key = f"{src}:{med}::{qry}:?"
            analogy_results[analogy_key] = self.test_analogies(src, med, qry, candidate_count=5)

        return {
            "model_path": str(self.model_path),
            "neighbors": similarity_results,
            "analogies": analogy_results,
        }


class EvaluationReporter:
    @staticmethod
    def compose_summary(analysis_results: Dict[str, Dict]) -> str:  # Generate a human-readable report summarizing the semantic evaluation results for each model.
        """Generate formatted evaluation report."""
        lines = ["SEMANTIC QUALITY EVALUATION", ""]

        for model_label, payload in analysis_results.items():
            lines.append(f"Model: {model_label}")
            lines.append(f"Location: {payload['model_path']}")

            # Similarity report
            lines.append("Semantic Neighbors:")
            for word, neighbors in payload["neighbors"].items():
                if neighbors:
                    similarity_list = ", ".join(f"{w} ({s:.4f})" for w, s in neighbors)
                    lines.append(f"  {word}: {similarity_list}")
                else:
                    lines.append(f"  {word}: [not in vocabulary]")

            # Analogy report
            lines.append("Analogy Solutions:")
            for expression, solutions in payload["analogies"].items():
                if solutions:
                    solution_list = ", ".join(f"{w} ({s:.4f})" for w, s in solutions)
                    lines.append(f"  {expression} -> {solution_list}")
                else:
                    lines.append(f"  {expression} -> [vocabulary gap]")

            lines.append("")

        return "\n".join(lines)


def task3_semantic_analysis(best_models_rel: str = "deliverables/task2/task2_best_models.json") -> Dict:
    # Load best model configurations
    root = root_dir()
    best_cfg_path = root / best_models_rel
    output_location = deliverables_dir() / "task3"
    ensure_dirs([output_location])

    if not best_cfg_path.exists():
        raise RuntimeError("Best model configuration not found. Run stage 2 first.")

    best_models_cfg = load_json(best_cfg_path)
    cbow_model_path = Path(best_models_cfg["cbow"]["model_path"])
    skipgram_model_path = Path(best_models_cfg["skipgram"]["model_path"])

    # Evaluate both models
    cbow_eval = EmbeddingEvaluator(cbow_model_path, TARGET_WORDS, ANALOGIES).full_evaluation()  # Evaluate CBOW model
    skipgram_eval = EmbeddingEvaluator(skipgram_model_path, TARGET_WORDS, ANALOGIES).full_evaluation()  # Evaluate Skip-gram model

    cumulative_results = {
        "CBOW": cbow_eval,
        "Skip-gram": skipgram_eval,
    }

    # Save structured results
    json_output = output_location / "task3_semantic_results.json"
    write_json(cumulative_results, json_output)

    # Save formatted report
    report = EvaluationReporter.compose_summary(cumulative_results)  # Generate a human-readable report from the evaluation results
    txt_output = output_location / "task3_semantic_results.txt"
    with txt_output.open("w", encoding="utf-8") as f:
        f.write(report)

    print(f"✓ Saved results: {json_output}")
    print(f"✓ Saved report: {txt_output}")
    return cumulative_results


if __name__ == "__main__":
    task3_semantic_analysis()
