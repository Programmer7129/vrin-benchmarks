#!/usr/bin/env python3
"""
Download RAGBench FinQA dataset from HuggingFace.

Dataset: rungalileo/ragbench (finqa subset)
Paper: https://arxiv.org/abs/2407.11005
"""

import os
import json
from pathlib import Path
from datasets import load_dataset

def download_ragbench_finqa():
    """Download RAGBench FinQA dataset and save to data/ directory."""

    print("🔽 Downloading RAGBench FinQA dataset...")
    print("   Source: rungalileo/ragbench (finqa subset)")
    print("   Size: ~16.6k examples")
    print()

    # Set up paths
    script_dir = Path(__file__).parent
    data_dir = script_dir.parent / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Download the finqa subset
        print("📥 Loading dataset from HuggingFace...")
        dataset = load_dataset("rungalileo/ragbench", "finqa")

        print(f"✅ Dataset loaded successfully!")
        print(f"   Train: {len(dataset['train'])} examples")
        print(f"   Validation: {len(dataset['validation'])} examples")
        print(f"   Test: {len(dataset['test'])} examples")
        print()

        # Save dataset splits as JSON
        for split_name, split_data in dataset.items():
            output_file = data_dir / f"{split_name}.json"
            print(f"💾 Saving {split_name} split to {output_file}...")

            # Convert to list of dictionaries
            examples = []
            for example in split_data:
                examples.append(dict(example))

            # Save as JSON
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(examples, f, indent=2, ensure_ascii=False)

            print(f"   ✅ Saved {len(examples)} examples")

        # Save dataset info
        info_file = data_dir / "dataset_info.json"
        info = {
            "name": "RAGBench FinQA",
            "source": "rungalileo/ragbench",
            "subset": "finqa",
            "paper": "https://arxiv.org/abs/2407.11005",
            "splits": {
                "train": len(dataset['train']),
                "validation": len(dataset['validation']),
                "test": len(dataset['test']),
                "total": len(dataset['train']) + len(dataset['validation']) + len(dataset['test'])
            },
            "description": "Financial question answering with numerical reasoning over hybrid tabular and text data",
            "features": list(dataset['train'].features.keys())
        }

        with open(info_file, 'w', encoding='utf-8') as f:
            json.dump(info, f, indent=2)

        print()
        print(f"📊 Dataset info saved to {info_file}")
        print()
        print("✅ RAGBench FinQA dataset downloaded successfully!")
        print(f"   Data directory: {data_dir}")
        print()

        # Print sample example
        print("📝 Sample example:")
        sample = dataset['test'][0]
        for key, value in sample.items():
            if isinstance(value, str) and len(value) > 100:
                print(f"   {key}: {value[:100]}...")
            else:
                print(f"   {key}: {value}")

        return True

    except Exception as e:
        print(f"❌ Error downloading dataset: {str(e)}")
        print(f"   Please ensure you have 'datasets' library installed:")
        print(f"   pip install datasets")
        return False

if __name__ == "__main__":
    download_ragbench_finqa()
