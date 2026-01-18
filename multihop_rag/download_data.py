#!/usr/bin/env python3
"""
Download MultiHop-RAG dataset from HuggingFace.

Dataset: yixuantt/MultiHopRAG
Paper: https://arxiv.org/abs/2401.15391
GitHub: https://github.com/yixuantt/MultiHop-RAG
"""

import os
import json
from pathlib import Path
from datasets import load_dataset

def download_multihop_rag():
    """Download MultiHop-RAG dataset and save to data/ directory."""

    print("🔽 Downloading MultiHop-RAG dataset...")
    print("   Source: yixuantt/MultiHopRAG")
    print("   Size: 2,556 queries with 2-4 document hops")
    print()

    # Set up paths
    script_dir = Path(__file__).parent
    data_dir = script_dir.parent / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Download the dataset (queries)
        print("📥 Loading MultiHopRAG queries from HuggingFace...")
        dataset = load_dataset("yixuantt/MultiHopRAG", "MultiHopRAG")

        # Also download the corpus
        print("📥 Loading MultiHopRAG corpus from HuggingFace...")
        corpus = load_dataset("yixuantt/MultiHopRAG", "corpus")

        print(f"✅ Queries loaded successfully!")
        for split_name in dataset.keys():
            print(f"   {split_name.capitalize()}: {len(dataset[split_name])} queries")
        print()

        print(f"✅ Corpus loaded successfully!")
        for split_name in corpus.keys():
            print(f"   {split_name.capitalize()}: {len(corpus[split_name])} documents")
        print()

        # Save query dataset splits as JSON
        for split_name, split_data in dataset.items():
            output_file = data_dir / f"queries_{split_name}.json"
            print(f"💾 Saving {split_name} queries to {output_file}...")

            # Convert to list of dictionaries
            examples = []
            for example in split_data:
                examples.append(dict(example))

            # Save as JSON
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(examples, f, indent=2, ensure_ascii=False)

            print(f"   ✅ Saved {len(examples)} queries")

        # Save corpus
        for split_name, split_data in corpus.items():
            output_file = data_dir / f"corpus_{split_name}.json"
            print(f"💾 Saving {split_name} corpus to {output_file}...")

            # Convert to list of dictionaries
            examples = []
            for example in split_data:
                examples.append(dict(example))

            # Save as JSON
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(examples, f, indent=2, ensure_ascii=False)

            print(f"   ✅ Saved {len(examples)} documents")

        # Get the first split for info (usually 'train' or 'test')
        first_split = list(dataset.keys())[0]

        # Save dataset info
        info_file = data_dir / "dataset_info.json"
        info = {
            "name": "MultiHop-RAG",
            "source": "yixuantt/MultiHopRAG",
            "paper": "https://arxiv.org/abs/2401.15391",
            "github": "https://github.com/yixuantt/MultiHop-RAG",
            "splits": {split_name: len(split_data) for split_name, split_data in dataset.items()},
            "total": sum(len(split_data) for split_data in dataset.values()),
            "description": "Multi-hop QA requiring retrieval and reasoning across 2-4 documents",
            "features": list(dataset[first_split].features.keys())
        }

        with open(info_file, 'w', encoding='utf-8') as f:
            json.dump(info, f, indent=2)

        print()
        print(f"📊 Dataset info saved to {info_file}")
        print()
        print("✅ MultiHop-RAG dataset downloaded successfully!")
        print(f"   Data directory: {data_dir}")
        print()

        # Print sample example
        print("📝 Sample example:")
        sample = dataset[first_split][0]
        for key, value in sample.items():
            if isinstance(value, str) and len(value) > 100:
                print(f"   {key}: {value[:100]}...")
            elif isinstance(value, list) and len(str(value)) > 100:
                print(f"   {key}: [list with {len(value)} items]...")
            else:
                print(f"   {key}: {value}")

        return True

    except Exception as e:
        print(f"❌ Error downloading dataset: {str(e)}")
        print(f"   Please ensure you have 'datasets' library installed:")
        print(f"   pip install datasets")
        return False

if __name__ == "__main__":
    download_multihop_rag()
