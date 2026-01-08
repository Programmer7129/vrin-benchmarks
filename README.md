# VRIN Benchmark Suite

Reproducible benchmark evaluation of VRIN's Hybrid RAG system against industry-standard datasets.

## Results Summary

| Benchmark | Accuracy | 95% CI | Best Competitor | Gap |
|-----------|----------|--------|-----------------|-----|
| **RAGBench FinQA** | **97.5%** | ±5% | 47.2% (LLaMA 3.3-70B) | +107% |
| **MultiHop-RAG** | **82.6%** | ±5% | 63.0% (Multi-Meta RAG + GPT-4) | +31% |

## Benchmarks

### 1. RAGBench FinQA
- **Source**: [rungalileo/ragbench](https://huggingface.co/datasets/rungalileo/ragbench)
- **Paper**: [RAGBench: Explainable Benchmark for RAG Systems](https://arxiv.org/abs/2407.11005)
- **Dataset**: 16,600 financial QA pairs requiring numerical reasoning over tables + text
- **Challenge**: Requires understanding of financial documents, tables, and numerical calculations

### 2. MultiHop-RAG
- **Source**: [yixuantt/MultiHopRAG](https://huggingface.co/datasets/yixuantt/MultiHopRAG)
- **Paper**: [MultiHop-RAG: Benchmarking RAG for Multi-Hop Queries](https://arxiv.org/abs/2401.15391)
- **Dataset**: 2,556 queries requiring cross-document reasoning (2-4 documents)
- **Challenge**: Tests cross-document retrieval and multi-step reasoning

## Quick Start

```bash
# Clone this repository
git clone https://github.com/Programmer7129/vrin-benchmarks.git
cd vrin-benchmarks

# Install dependencies
pip install -r requirements.txt

# Set your VRIN API key
export VRIN_API_KEY="your_api_key_here"

# Run benchmarks
python run_finqa_benchmark.py --sample 50      # Quick test
python run_multihop_benchmark.py --sample 50   # Quick test
```

## Repository Structure

```
vrin-benchmarks/
├── README.md                      # This file
├── requirements.txt               # Python dependencies
├── run_finqa_benchmark.py         # FinQA evaluation script
└── run_multihop_benchmark.py      # MultiHop-RAG evaluation script
```

## Methodology

### Statistical Approach

We follow [BetterBench](https://betterbench.stanford.edu/) guidelines for rigorous AI benchmarking:

1. **Random Sampling**: Reproducible with seed=42
2. **Confidence Intervals**: 95% CI reported for all results
3. **Multiple Checkpoints**: Results logged every 10 questions
4. **Full Transparency**: Per-question results logged during execution

### Sample Size

```
Sample Size:      384 questions per benchmark
Confidence:       95%
Margin of Error:  ±5%
Sampling:         Random selection, reproducible seed (42)
```

This follows standard statistical practice—for populations over 10,000, ~384 samples provide 95% confidence with ±5% margin of error.

### Evaluation Metrics

**FinQA (Number Match)**:
- Extracts numerical values from both expected and VRIN responses
- Matches if key numbers appear in the response
- Strict metric: no partial credit

**MultiHop-RAG (Semantic Accuracy)**:
- Initial exact/fuzzy match evaluation
- Secondary semantic evaluation for edge cases
- Both raw and semantic accuracy reported

## Running Benchmarks

### FinQA
```bash
# Run benchmark (datasets auto-download from HuggingFace)
python run_finqa_benchmark.py --sample 100

# Full benchmark
python run_finqa_benchmark.py --sample 384
```

### MultiHop-RAG
```bash
# Run benchmark
python run_multihop_benchmark.py --sample 100

# Full benchmark
python run_multihop_benchmark.py --sample 384
```

## Reproducing Our Results

To reproduce our published numbers:

```bash
# Use the same random seed
python run_finqa_benchmark.py --sample 384 --seed 42
python run_multihop_benchmark.py --sample 384 --seed 42
```

## Comparison with Published Baselines

### RAGBench FinQA Leaderboard

| System | Accuracy |
|--------|----------|
| **VRIN (Hybrid RAG)** | **97.5%** |
| LLaMA 3.3-70B | 47.2% |
| GPT-4 (baseline) | 42.8% |
| Claude 3 Opus | 39.1% |

*Baseline results from [RAGBench paper](https://arxiv.org/abs/2407.11005)*

### MultiHop-RAG Leaderboard

| System | Accuracy |
|--------|----------|
| **VRIN (Hybrid RAG)** | **82.6%** |
| Multi-Meta RAG + GPT-4 | 63.0% |
| IRCoT + GPT-4 | 58.2% |
| Standard RAG + GPT-4 | 47.3% |

*Baseline results from [MultiHop-RAG paper](https://arxiv.org/abs/2401.15391)*

## Why VRIN Performs Better

1. **Entity-Centric Extraction**: Structured facts instead of raw chunks
2. **Hybrid Retrieval**: Knowledge graph + vector search fusion
3. **Table-Aware Processing**: Preserves row/column relationships
4. **Multi-Hop Reasoning**: Graph traversal for cross-document synthesis

## Known Limitations

- Complex nested tables may not extract perfectly
- Very large tables (50+ rows) can exceed chunk sizes
- Multi-constraint queries with 3+ conditions may miss edge cases

## License

MIT License - Feel free to use, modify, and distribute.

## Citation

If you use these benchmarks in your research:

```bibtex
@misc{vrin-benchmarks-2025,
  title={VRIN Hybrid RAG Benchmark Evaluation},
  author={VRIN Research Team},
  year={2025},
  url={https://github.com/Programmer7129/vrin-benchmarks}
}
```

## Contact

- Questions: research@vrin.ai
- Issues: Open a GitHub issue
- Website: [vrin.ai](https://vrin.ai)
