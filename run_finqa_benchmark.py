#!/usr/bin/env python3
"""
RAGBench FinQA Benchmark - Large Sample with Continuous Logging

Dataset: rungalileo/ragbench (FinQA split)
Sample Size: 384 questions
Statistical: Calculated margin of error with finite population correction
"""

import os
import sys
import json
import random
import time
from pathlib import Path
from datetime import datetime
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent))

from vrin import VRINClient
from benchmark_utils import (
    calculate_margin_of_error,
    get_api_key,
    evaluate_finqa_answer,
    format_duration
)

# Configuration
SAMPLE_SIZE = 384
LOG_FILE = Path(__file__).parent / "ragbench_finqa" / "logs" / f"finqa_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
RESULTS_FILE = Path(__file__).parent / "ragbench_finqa" / "results" / f"finqa_{SAMPLE_SIZE}_sample_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"


def log(message):
    """Write to log file and print"""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    log_msg = f"[{timestamp}] {message}"
    print(log_msg)
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_FILE, 'a') as f:
        f.write(log_msg + '\n')


def run_benchmark():
    # Load full dataset
    data_file = Path(__file__).parent / "ragbench_finqa" / "data" / "test.json"
    log(f"Loading dataset from: {data_file}")

    with open(data_file, 'r') as f:
        full_dataset = json.load(f)

    population_size = len(full_dataset)
    log(f"Full dataset size: {population_size} questions")

    # Calculate margin of error
    margin_of_error = calculate_margin_of_error(SAMPLE_SIZE, population_size)

    log("=" * 80)
    log("RAGBench FinQA Benchmark")
    log(f"Sample Size: {SAMPLE_SIZE} questions")
    log(f"Population Size: {population_size} questions")
    log(f"Margin of Error: ±{margin_of_error}% at 95% confidence")
    log(f"Sampling Method: Random (seed=42)")
    log(f"Log File: {LOG_FILE}")
    log("=" * 80)

    # Random sampling (FinQA doesn't have question_type for stratification)
    random.seed(42)
    if len(full_dataset) > SAMPLE_SIZE:
        sample = random.sample(full_dataset, SAMPLE_SIZE)
        log(f"Randomly sampled {SAMPLE_SIZE} questions")
    else:
        sample = full_dataset
        log(f"Using all {len(sample)} questions (dataset smaller than sample size)")

    # Initialize client (API key required from environment)
    api_key = get_api_key()
    client = VRINClient(api_key=api_key)
    log(f"VRIN Client initialized")

    # Tracking
    results = []
    successful_insertions = 0
    successful_queries = 0
    correct_answers = 0
    match_types = defaultdict(int)
    start_time = time.time()

    for idx, item in enumerate(sample, 1):
        question_start = time.time()

        log(f"\n{'='*60}")
        log(f"Question {idx}/{len(sample)} (ID: {item['id']})")
        log(f"Query: {item['question'][:80]}...")

        # Combine documents
        combined_content = f"FinQA Document {item['id']}\n\n"
        for i, doc in enumerate(item['documents']):
            if isinstance(doc, str):
                combined_content += f"Document {i+1}:\n{doc}\n\n"
            elif isinstance(doc, list):
                # Table format - preserve structure
                combined_content += f"Table {i+1}:\n"
                for row in doc:
                    combined_content += " | ".join(str(cell) for cell in row) + "\n"
                combined_content += "\n"

        # Insert
        try:
            insert_result = client.insert(combined_content, wait=True)
            facts_extracted = insert_result.get('facts_extracted', 0)
            facts_stored = insert_result.get('facts_stored', 0)
            log(f"INSERT: {facts_extracted} facts extracted, {facts_stored} stored")
            successful_insertions += 1
        except Exception as e:
            log(f"INSERT FAILED: {str(e)}")
            results.append({
                'id': item['id'],
                'question': item['question'],
                'expected': item['response'],
                'vrin_response': '',
                'insertion_success': False,
                'query_success': False,
                'correct': False,
                'match_type': 'insertion_failed',
                'elapsed': time.time() - question_start
            })
            continue

        # Query using research mode (full knowledge graph + vector search)
        try:
            query_result = client.query(item['question'], response_mode='research')
            vrin_answer = query_result.get('summary', query_result.get('response', ''))
            log(f"QUERY: Got response ({len(vrin_answer)} chars)")
            successful_queries += 1

            # Evaluate with improved numerical matching
            correct, match_type = evaluate_finqa_answer(item['response'], vrin_answer)
            match_types[match_type] += 1

            if correct:
                log(f"CORRECT ({match_type})")
                correct_answers += 1
            else:
                log(f"INCORRECT")
                log(f"   Expected (first 100 chars): {item['response'][:100]}")
                log(f"   VRIN (first 100 chars): {vrin_answer[:100]}")

            question_elapsed = time.time() - question_start
            log(f"Time: {format_duration(question_elapsed)}")

            results.append({
                'id': item['id'],
                'question': item['question'],
                'expected': item['response'],
                'vrin_response': vrin_answer,
                'insertion_success': True,
                'query_success': True,
                'correct': correct,
                'match_type': match_type,
                'elapsed': question_elapsed
            })

        except Exception as e:
            log(f"QUERY FAILED: {str(e)}")
            results.append({
                'id': item['id'],
                'question': item['question'],
                'expected': item['response'],
                'vrin_response': '',
                'insertion_success': True,
                'query_success': False,
                'correct': False,
                'match_type': 'query_failed',
                'elapsed': time.time() - question_start
            })

        # Progress update every 10 questions
        if idx % 10 == 0:
            elapsed = time.time() - start_time
            avg_time = elapsed / idx
            remaining = (len(sample) - idx) * avg_time
            current_accuracy = correct_answers / idx * 100
            log(f"\nPROGRESS:")
            log(f"   Completed: {idx}/{len(sample)} ({idx/len(sample)*100:.1f}%)")
            log(f"   Correct: {correct_answers}/{idx} ({current_accuracy:.1f}%)")
            log(f"   Avg time: {format_duration(avg_time)}/question")
            log(f"   ETA: {format_duration(remaining)}")

    # Final summary
    total_elapsed = time.time() - start_time
    accuracy = (correct_answers / len(sample) * 100) if len(sample) > 0 else 0

    log(f"\n{'='*80}")
    log("FINAL RESULTS")
    log(f"{'='*80}")
    log(f"Accuracy: {accuracy:.1f}% ({correct_answers}/{len(sample)})")
    log(f"Margin of Error: ±{margin_of_error}% at 95% confidence")
    log(f"Confidence Interval: [{accuracy - margin_of_error:.1f}%, {accuracy + margin_of_error:.1f}%]")
    log(f"Successful insertions: {successful_insertions}/{len(sample)}")
    log(f"Successful queries: {successful_queries}/{len(sample)}")
    log(f"Total time: {format_duration(total_elapsed)}")
    log(f"Avg time/question: {format_duration(total_elapsed/len(sample))}")
    log(f"\nMatch Types: {dict(match_types)}")

    # Save results
    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    final_results = {
        'benchmark': 'RAGBench FinQA',
        'dataset_source': 'rungalileo/ragbench',
        'timestamp': datetime.now().isoformat(),

        # Sample info
        'sample_size': len(sample),
        'population_size': population_size,
        'sampling_method': 'random',
        'random_seed': 42,

        # Statistical validity
        'confidence_level': '95%',
        'margin_of_error': f"±{margin_of_error}%",
        'confidence_interval': {
            'lower': round(accuracy - margin_of_error, 1),
            'upper': round(accuracy + margin_of_error, 1)
        },

        # Results
        'accuracy': round(accuracy, 2),
        'correct_answers': correct_answers,
        'successful_insertions': successful_insertions,
        'successful_queries': successful_queries,

        # Match type breakdown
        'match_types': dict(match_types),

        # Timing
        'total_time_seconds': round(total_elapsed, 1),
        'avg_time_per_question': round(total_elapsed / len(sample), 1),

        # Detailed results
        'detailed_results': results
    }

    with open(RESULTS_FILE, 'w') as f:
        json.dump(final_results, f, indent=2)

    log(f"\nResults: {RESULTS_FILE}")
    log(f"Log: {LOG_FILE}")


if __name__ == "__main__":
    try:
        run_benchmark()
    except KeyboardInterrupt:
        log("\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        log(f"\nFATAL: {str(e)}")
        import traceback
        log(traceback.format_exc())
        sys.exit(1)
