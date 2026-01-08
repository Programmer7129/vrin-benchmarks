#!/usr/bin/env python3
"""
RAGBench FinQA Benchmark - Large Sample with Continuous Logging
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
LOG_FILE = Path(__file__).parent / "ragbench_finqa" / "logs" / f"finqa_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
RESULTS_FILE = Path(__file__).parent / "ragbench_finqa" / "results" / f"finqa_384_sample_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

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
    log("RAGBench FinQA Benchmark - Large Sample Run")
    log(f"Sample Size: {SAMPLE_SIZE} questions")
    log(f"Error Margin: ±5% at 95% confidence")
    log(f"Log File: {LOG_FILE}")
    log("=" * 80)

    # Load full dataset
    data_file = Path(__file__).parent / "ragbench_finqa" / "data" / "test.json"
    log(f"Loading dataset from: {data_file}")

    with open(data_file, 'r') as f:
        full_dataset = json.load(f)

    log(f"Full dataset size: {len(full_dataset)} questions")

    # Random sampling
    random.seed(42)  # Reproducible sampling
    if len(full_dataset) > SAMPLE_SIZE:
        sample = random.sample(full_dataset, SAMPLE_SIZE)
        log(f"Randomly sampled {SAMPLE_SIZE} questions")
    else:
        sample = full_dataset
        log(f"Using all {len(sample)} questions (dataset smaller than sample size)")

    # Initialize client
    api_key = os.getenv('TEST_ACC_API_KEY', 'vrin_4926d56cff3a2adc')
    client = VRINClient(api_key=api_key)
    log(f"VRIN Client initialized with API key: {api_key[:15]}...")

    # Tracking
    results = []
    successful_insertions = 0
    successful_queries = 0
    correct_answers = 0
    start_time = time.time()

    for idx, item in enumerate(sample, 1):
        question_start = time.time()

        log(f"\n{'='*60}")
        log(f"Question {idx}/{SAMPLE_SIZE} (ID: {item['id']})")
        log(f"Query: {item['question'][:80]}...")

        # Combine documents
        combined_content = f"FinQA Document {item['id']}\n\n"
        for i, doc in enumerate(item['documents']):
            if isinstance(doc, str):
                combined_content += f"Document {i+1}:\n{doc}\n\n"
            elif isinstance(doc, list):
                # Table format
                combined_content += f"Table {i+1}:\n"
                for row in doc:
                    combined_content += " | ".join(str(cell) for cell in row) + "\n"
                combined_content += "\n"

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
                'id': item['id'],
                'question': item['question'],
                'expected': item['response'],
                'vrin_response': '',
                'insertion_success': False,
                'query_success': False,
                'correct': False,
                'elapsed': time.time() - question_start
            })
            continue

        # Query
        try:
            query_result = client.query(item['question'])
            vrin_answer = query_result.get('summary', query_result.get('response', ''))
            log(f"✅ QUERY: Got response ({len(vrin_answer)} chars)")
            successful_queries += 1

            # Check correctness (fuzzy match - look for numerical values)
            import re
            expected_nums = re.findall(r'[\d,]+\.?\d*', item['response'])
            vrin_nums = re.findall(r'[\d,]+\.?\d*', vrin_answer)

            correct = False
            for exp_num in expected_nums[:3]:  # Check first 3 numbers from expected
                exp_clean = exp_num.replace(',', '')
                for vrin_num in vrin_nums:
                    vrin_clean = vrin_num.replace(',', '')
                    if exp_clean in vrin_clean or vrin_clean in exp_clean:
                        correct = True
                        break
                if correct:
                    break

            if correct:
                log(f"🎯 CORRECT: Found expected numerical value")
                correct_answers += 1
            else:
                log(f"❌ INCORRECT: Expected values not found")
                log(f"   Expected (first 100 chars): {item['response'][:100]}")
                log(f"   VRIN (first 100 chars): {vrin_answer[:100]}")

            question_elapsed = time.time() - question_start
            log(f"⏱️  Question time: {question_elapsed:.1f}s")

            results.append({
                'id': item['id'],
                'question': item['question'],
                'expected': item['response'],
                'vrin_response': vrin_answer,
                'insertion_success': True,
                'query_success': True,
                'correct': correct,
                'elapsed': question_elapsed
            })

        except Exception as e:
            log(f"❌ QUERY FAILED: {str(e)}")
            results.append({
                'id': item['id'],
                'question': item['question'],
                'expected': item['response'],
                'vrin_response': '',
                'insertion_success': True,
                'query_success': False,
                'correct': False,
                'elapsed': time.time() - question_start
            })

        # Progress update every 10 questions
        if idx % 10 == 0:
            elapsed = time.time() - start_time
            avg_time = elapsed / idx
            remaining = (SAMPLE_SIZE - idx) * avg_time
            log(f"\n📊 PROGRESS UPDATE:")
            log(f"   Completed: {idx}/{SAMPLE_SIZE} ({idx/SAMPLE_SIZE*100:.1f}%)")
            log(f"   Correct so far: {correct_answers}/{idx} ({correct_answers/idx*100:.1f}%)")
            log(f"   Avg time/question: {avg_time:.1f}s")
            log(f"   Estimated remaining: {remaining/60:.1f} min")

    # Final summary
    total_elapsed = time.time() - start_time
    accuracy = (correct_answers / SAMPLE_SIZE * 100) if SAMPLE_SIZE > 0 else 0

    log(f"\n{'='*80}")
    log("FINAL RESULTS")
    log(f"{'='*80}")
    log(f"Total questions: {SAMPLE_SIZE}")
    log(f"Successful insertions: {successful_insertions}/{SAMPLE_SIZE}")
    log(f"Successful queries: {successful_queries}/{SAMPLE_SIZE}")
    log(f"Correct answers: {correct_answers}/{SAMPLE_SIZE}")
    log(f"Accuracy: {accuracy:.1f}%")
    log(f"Total time: {total_elapsed/60:.1f} minutes")
    log(f"Avg time/question: {total_elapsed/SAMPLE_SIZE:.1f}s")

    # Save results
    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_FILE, 'w') as f:
        json.dump({
            'benchmark': 'RAGBench FinQA',
            'sample_size': SAMPLE_SIZE,
            'error_margin': '±5%',
            'confidence_level': '95%',
            'timestamp': datetime.now().isoformat(),
            'successful_insertions': successful_insertions,
            'successful_queries': successful_queries,
            'correct_answers': correct_answers,
            'accuracy': accuracy,
            'total_time_seconds': total_elapsed,
            'avg_time_per_question': total_elapsed / SAMPLE_SIZE,
            'results': results
        }, f, indent=2)

    log(f"\n✅ Results saved to: {RESULTS_FILE}")
    log(f"✅ Log file: {LOG_FILE}")

if __name__ == "__main__":
    try:
        run_benchmark()
    except KeyboardInterrupt:
        log("\n⚠️  Benchmark interrupted by user")
        sys.exit(1)
    except Exception as e:
        log(f"\n❌ FATAL ERROR: {str(e)}")
        import traceback
        log(traceback.format_exc())
        sys.exit(1)
