#!/usr/bin/env python3
"""
MultiHop-RAG Benchmark - Large Sample with Continuous Logging
Sample Size: 200 questions (±7% error margin)
"""

import os
import sys
import json
import random
import time
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from vrin import VRINClient

# Configuration
SAMPLE_SIZE = 384  # ±5% error margin at 95% confidence
LOG_FILE = Path(__file__).parent / "multihop_rag" / "logs" / f"multihop_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
RESULTS_FILE = Path(__file__).parent / "multihop_rag" / "results" / f"multihop_384_sample_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

def log(message):
    """Write to log file and print"""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    log_msg = f"[{timestamp}] {message}"
    print(log_msg)
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_FILE, 'a') as f:
        f.write(log_msg + '\n')

def run_benchmark():
    log("=" * 80)
    log("MultiHop-RAG Benchmark - Large Sample Run")
    log(f"Sample Size: {SAMPLE_SIZE} questions")
    log(f"Error Margin: ±5% at 95% confidence")
    log(f"Log File: {LOG_FILE}")
    log("=" * 80)

    # Load dataset
    data_file = Path(__file__).parent / "multihop_rag" / "data" / "queries_train.json"
    log(f"Loading dataset from: {data_file}")

    with open(data_file, 'r') as f:
        full_dataset = json.load(f)

    log(f"Full dataset size: {len(full_dataset)} questions")

    # Random sampling
    random.seed(42)
    if len(full_dataset) > SAMPLE_SIZE:
        sample = random.sample(full_dataset, SAMPLE_SIZE)
        log(f"Randomly sampled {SAMPLE_SIZE} questions")
    else:
        sample = full_dataset
        log(f"Using all {len(sample)} questions")

    # Initialize client
    api_key = os.getenv('TEST_ACC_API_KEY', 'vrin_4926d56cff3a2adc')
    client = VRINClient(api_key=api_key)
    log(f"VRIN Client initialized")

    # Tracking
    results = []
    successful_insertions = 0
    successful_queries = 0
    correct_answers = 0
    start_time = time.time()

    for idx, item in enumerate(sample, 1):
        question_start = time.time()

        log(f"\n{'='*60}")
        log(f"Question {idx}/{SAMPLE_SIZE}")
        log(f"Query: {item['query'][:80]}...")
        log(f"Expected: {item['answer']}")
        log(f"Type: {item.get('question_type', 'unknown')}")

        # Combine evidence
        combined_content = f"MultiHop Evidence {idx}\n\n"
        for i, evidence in enumerate(item['evidence_list']):
            combined_content += f"Document {i+1} - {evidence.get('title', 'Untitled')}\n"
            combined_content += f"Source: {evidence.get('source', 'Unknown')}\n"
            combined_content += f"Fact: {evidence['fact']}\n\n"

        # Insert
        try:
            insert_result = client.insert(combined_content, wait=True)
            facts_extracted = insert_result.get('facts_extracted', 0)
            facts_stored = insert_result.get('facts_stored', 0)
            log(f"✅ INSERT: {facts_extracted} facts extracted, {facts_stored} stored")
            successful_insertions += 1
        except Exception as e:
            log(f"❌ INSERT FAILED: {str(e)}")
            results.append({
                'query': item['query'],
                'expected': item['answer'],
                'vrin_response': '',
                'insertion_success': False,
                'query_success': False,
                'correct': False,
                'elapsed': time.time() - question_start
            })
            continue

        # Query
        try:
            query_result = client.query(item['query'])
            vrin_answer = query_result.get('summary', query_result.get('response', ''))
            log(f"✅ QUERY: Got response")
            successful_queries += 1

            # Check correctness (case-insensitive substring or semantic match)
            expected_lower = item['answer'].lower()
            vrin_lower = vrin_answer.lower()

            # Direct match or semantic equivalents
            correct = (
                expected_lower in vrin_lower or
                (expected_lower == "yes" and any(word in vrin_lower[:50] for word in ["yes", "absolutely", "indeed", "correct"])) or
                (expected_lower == "no" and any(word in vrin_lower[:50] for word in ["no", "not", "inconsistent"])) or
                ("insufficient" in expected_lower and ("not available" in vrin_lower or "insufficient" in vrin_lower))
            )

            if correct:
                log(f"🎯 CORRECT")
                correct_answers += 1
            else:
                log(f"❌ INCORRECT")
                log(f"   VRIN (first 100 chars): {vrin_answer[:100]}")

            question_elapsed = time.time() - question_start
            log(f"⏱️  Question time: {question_elapsed:.1f}s")

            results.append({
                'query': item['query'],
                'expected': item['answer'],
                'vrin_response': vrin_answer,
                'question_type': item.get('question_type'),
                'insertion_success': True,
                'query_success': True,
                'correct': correct,
                'elapsed': question_elapsed
            })

        except Exception as e:
            log(f"❌ QUERY FAILED: {str(e)}")
            results.append({
                'query': item['query'],
                'expected': item['answer'],
                'vrin_response': '',
                'insertion_success': True,
                'query_success': False,
                'correct': False,
                'elapsed': time.time() - question_start
            })

        # Progress every 10 questions
        if idx % 10 == 0:
            elapsed = time.time() - start_time
            avg_time = elapsed / idx
            remaining = (SAMPLE_SIZE - idx) * avg_time
            log(f"\n📊 PROGRESS:")
            log(f"   Completed: {idx}/{SAMPLE_SIZE} ({idx/SAMPLE_SIZE*100:.1f}%)")
            log(f"   Correct: {correct_answers}/{idx} ({correct_answers/idx*100:.1f}%)")
            log(f"   Avg time: {avg_time:.1f}s/question")
            log(f"   ETA: {remaining/60:.1f} min")

    # Final summary
    total_elapsed = time.time() - start_time
    accuracy = (correct_answers / SAMPLE_SIZE * 100) if SAMPLE_SIZE > 0 else 0

    log(f"\n{'='*80}")
    log("FINAL RESULTS")
    log(f"{'='*80}")
    log(f"Accuracy: {accuracy:.1f}% ({correct_answers}/{SAMPLE_SIZE})")
    log(f"Total time: {total_elapsed/60:.1f} minutes")

    # Save
    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_FILE, 'w') as f:
        json.dump({
            'benchmark': 'MultiHop-RAG',
            'sample_size': SAMPLE_SIZE,
            'error_margin': '±5%',
            'timestamp': datetime.now().isoformat(),
            'correct_answers': correct_answers,
            'accuracy': accuracy,
            'total_time_seconds': total_elapsed,
            'results': results
        }, f, indent=2)

    log(f"\n✅ Results: {RESULTS_FILE}")
    log(f"✅ Log: {LOG_FILE}")

if __name__ == "__main__":
    try:
        run_benchmark()
    except KeyboardInterrupt:
        log("\n⚠️  Interrupted by user")
        sys.exit(1)
    except Exception as e:
        log(f"\n❌ FATAL: {str(e)}")
        import traceback
        log(traceback.format_exc())
        sys.exit(1)
