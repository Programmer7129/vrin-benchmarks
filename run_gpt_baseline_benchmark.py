#!/usr/bin/env python3
"""
GPT Baseline Benchmark - For comparison with VRIN on MultiHop-RAG

Dataset: yixuantt/MultiHopRAG
Sample Size: 100 questions
Purpose: Compare raw GPT performance (with evidence in context) vs VRIN Hybrid RAG

This gives GPT the same evidence documents that VRIN ingests, simulating
what a user would do without VRIN: copy/paste documents into ChatGPT.
"""

import os
import sys
import json
import time
import requests
from pathlib import Path
from datetime import datetime
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent))

from benchmark_utils import (
    calculate_margin_of_error,
    stratified_sample,
    evaluate_multihop_answer,
    format_duration
)

# Configuration
SAMPLE_SIZE = 100
MODEL = "gpt-4o"  # Change to the model you want to benchmark
LOG_FILE = Path(__file__).parent / "gpt_baseline" / "logs" / f"gpt_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
RESULTS_FILE = Path(__file__).parent / "gpt_baseline" / "results" / f"gpt_{SAMPLE_SIZE}_sample_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"


def log(message):
    """Write to log file and print"""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    log_msg = f"[{timestamp}] {message}"
    print(log_msg)
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_FILE, 'a') as f:
        f.write(log_msg + '\n')


def get_openai_key():
    """Get OpenAI API key from environment or .env file"""
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        env_path = Path(__file__).parent / '.env'
        if env_path.exists():
            with open(env_path, 'r') as f:
                for line in f:
                    if line.startswith('OPENAI_API_KEY='):
                        api_key = line.split('=', 1)[1].strip()
                        break
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in environment or .env file")
    return api_key


def query_gpt(question: str, evidence_docs: list, api_key: str) -> str:
    """
    Query GPT directly with evidence documents in context.

    This simulates what a user would do without VRIN - copy/paste documents
    into ChatGPT and ask a question.
    """
    context = "## Evidence Documents\n\n"
    for i, evidence in enumerate(evidence_docs):
        context += f"### Document {i+1}: {evidence.get('title', 'Untitled')}\n"
        context += f"Source: {evidence.get('source', 'Unknown')}\n"
        context += f"Content: {evidence['fact']}\n\n"

    prompt = f"""{context}

## Question
{question}

## Instructions
Based ONLY on the evidence documents provided above, answer the question.
- For Yes/No questions: Start with "Yes" or "No" then explain
- For comparison questions: State if they are "Similar" or "Different"
- For entity questions: State the entity name clearly
- If information is insufficient: Say "Insufficient information"

Your answer:"""

    try:
        response = requests.post(
            'https://api.openai.com/v1/chat/completions',
            headers={
                'Authorization': f'Bearer {api_key}',
                'Content-Type': 'application/json'
            },
            json={
                'model': MODEL,
                'messages': [
                    {
                        'role': 'system',
                        'content': 'You are a helpful assistant that answers questions based on provided evidence documents. Be precise and cite evidence when possible.'
                    },
                    {'role': 'user', 'content': prompt}
                ],
                'max_tokens': 1000,
                'temperature': 0.1
            },
            timeout=120
        )

        if response.status_code == 200:
            result = response.json()
            return result['choices'][0]['message']['content']
        else:
            return f"Error: {response.status_code} - {response.text[:200]}"

    except Exception as e:
        return f"Error: {str(e)}"


def run_benchmark():
    # Load dataset
    data_file = Path(__file__).parent / "multihop_rag" / "data" / "queries_train.json"
    log(f"Loading dataset from: {data_file}")

    with open(data_file, 'r') as f:
        full_dataset = json.load(f)

    population_size = len(full_dataset)
    log(f"Full dataset size: {population_size} questions")

    # Calculate margin of error
    margin_of_error = calculate_margin_of_error(SAMPLE_SIZE, population_size)

    log("=" * 80)
    log(f"GPT Baseline Benchmark (Model: {MODEL})")
    log(f"Sample Size: {SAMPLE_SIZE} questions")
    log(f"Population Size: {population_size} questions")
    log(f"Margin of Error: ±{margin_of_error}% at 95% confidence")
    log(f"Sampling Method: Stratified by question_type")
    log(f"Log File: {LOG_FILE}")
    log("=" * 80)

    # Stratified sampling - use SAME seed as VRIN for fair comparison
    sample, question_type_distribution = stratified_sample(
        full_dataset,
        SAMPLE_SIZE,
        stratify_key='question_type',
        seed=42  # Same seed as VRIN benchmark
    )
    log(f"Stratified sample of {len(sample)} questions by question_type")
    log(f"Distribution: {json.dumps(question_type_distribution)}")

    # Get API key
    api_key = get_openai_key()
    log(f"OpenAI API key loaded")

    # Tracking
    results = []
    correct_answers = 0
    match_types = defaultdict(int)
    accuracy_by_type = defaultdict(lambda: {'correct': 0, 'total': 0})
    start_time = time.time()

    for idx, item in enumerate(sample, 1):
        question_start = time.time()
        question_type = item.get('question_type', 'unknown')

        log(f"\n{'='*60}")
        log(f"Question {idx}/{len(sample)}")
        log(f"Query: {item['query'][:80]}...")
        log(f"Expected: {item['answer']}")
        log(f"Type: {question_type}")

        # Query GPT directly with evidence
        try:
            gpt_answer = query_gpt(item['query'], item['evidence_list'], api_key)
            log(f"GPT: Got response ({len(gpt_answer)} chars)")

            # Evaluate with same LLM normalizer as VRIN
            correct, match_type = evaluate_multihop_answer(
                item['answer'],
                gpt_answer,
                question=item['query']
            )
            match_types[match_type] += 1

            if correct:
                log(f"CORRECT ({match_type})")
                correct_answers += 1
                accuracy_by_type[question_type]['correct'] += 1
            else:
                log(f"INCORRECT")
                log(f"   GPT (first 150 chars): {gpt_answer[:150]}")

            accuracy_by_type[question_type]['total'] += 1
            question_elapsed = time.time() - question_start
            log(f"Time: {format_duration(question_elapsed)}")

            results.append({
                'query': item['query'],
                'expected': item['answer'],
                'gpt_response': gpt_answer,
                'question_type': question_type,
                'correct': correct,
                'match_type': match_type,
                'elapsed': question_elapsed
            })

        except Exception as e:
            log(f"QUERY FAILED: {str(e)}")
            accuracy_by_type[question_type]['total'] += 1
            results.append({
                'query': item['query'],
                'expected': item['answer'],
                'gpt_response': '',
                'question_type': question_type,
                'correct': False,
                'match_type': 'query_failed',
                'elapsed': time.time() - question_start
            })

        # Progress every 10 questions
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

    # Calculate accuracy by question type
    accuracy_by_type_pct = {}
    for q_type, stats in accuracy_by_type.items():
        if stats['total'] > 0:
            accuracy_by_type_pct[q_type] = {
                'correct': stats['correct'],
                'total': stats['total'],
                'accuracy': round(stats['correct'] / stats['total'] * 100, 1)
            }

    log(f"\n{'='*80}")
    log("FINAL RESULTS")
    log(f"{'='*80}")
    log(f"Model: {MODEL}")
    log(f"Accuracy: {accuracy:.1f}% ({correct_answers}/{len(sample)})")
    log(f"Margin of Error: ±{margin_of_error}% at 95% confidence")
    log(f"Confidence Interval: [{accuracy - margin_of_error:.1f}%, {accuracy + margin_of_error:.1f}%]")
    log(f"Total time: {format_duration(total_elapsed)}")
    log(f"\nMatch Types: {dict(match_types)}")
    log(f"\nAccuracy by Question Type:")
    for q_type, stats in sorted(accuracy_by_type_pct.items()):
        log(f"   {q_type}: {stats['accuracy']:.1f}% ({stats['correct']}/{stats['total']})")

    # Save results
    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    final_results = {
        'benchmark': f'GPT Baseline ({MODEL})',
        'model': MODEL,
        'dataset_source': 'yixuantt/MultiHopRAG',
        'timestamp': datetime.now().isoformat(),

        # Sample info
        'sample_size': len(sample),
        'population_size': population_size,
        'sampling_method': 'stratified_by_question_type',
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

        # Breakdown
        'match_types': dict(match_types),
        'question_type_distribution': question_type_distribution,
        'accuracy_by_question_type': accuracy_by_type_pct,

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
