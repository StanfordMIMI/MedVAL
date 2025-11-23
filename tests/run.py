import json
import os
import sys
import argparse
from pathlib import Path
from collections import defaultdict
import time

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import dspy
from datasets import load_dataset
from medval.validator import DetectTask, MedVAL_Validator


with open("utils/task_prompts.json", "r") as f:
    task_prompts = json.load(f)


def load_test_data():
    """Load test cases from HuggingFace MedVAL-Bench dataset"""

    hf_dataset = load_dataset("stanfordmimi/MedVAL-Bench")
    dataset_split = hf_dataset["test"]
    df = dataset_split.to_pandas()

    test_cases = []
    for _, row in df.iterrows():
        risk_grade = row.get('physician_risk_grade', '')

        if risk_grade is None:
            continue

        test_cases.append({
            'id': row.get('id', row.get('#', '')),
            'task': row['task'],
            'reference': row['input'],  # input field is the reference
            'candidate': row['output'],  # output is the AI-generated candidate
            'risk_level': risk_grade,
            'physician_assessment': row.get('physician_error_assessment', ''),
            'reference_output': row.get('reference_output', None)
        })

    return test_cases


def configure_validator(model="openai/gpt-4o-mini"):
    """Initialize DSPy and validator modules"""
    api_key = os.environ.get("API_KEY") or os.environ.get("OPENAI_API_KEY")

    if not api_key:
        print("  Error: No API key found.")
        print("   Set API_KEY or OPENAI_API_KEY environment variable.")
        sys.exit(1)

    lm = dspy.LM(model=model, api_key=api_key)
    dspy.configure(lm=lm)

    return {
        "task_detector": dspy.ChainOfThought(DetectTask),
        "validator": dspy.ChainOfThought(MedVAL_Validator)
    }


def run_validation(modules, reference, candidate, expected_task=None):
    """Run complete validation workflow"""
    try:
        detected = modules['task_detector'](reference=reference, candidate=candidate)

        instruction_text = task_prompts.get(detected.task, "")

        result = modules['validator'](
            instruction=instruction_text,
            reference=reference,
            candidate=candidate
        )

        return {
            "success": True,
            "detected_task": detected.task,
            "expected_task": expected_task,
            "task_correct": (detected.task == expected_task) if expected_task else None,
            "detected_instruction": instruction_text,
            "risk_level": result.risk_level,
            "errors": result.structured_errors,
            "error": None
        }
    except Exception as e:
        return {
            "success": False,
            "detected_task": None,
            "expected_task": expected_task,
            "task_correct": False,
            "error": str(e)
        }


def print_header(text):
    """Print formatted header"""
    print("\n" + "=" * 70)
    print(f"  {text}")
    print("=" * 70)


def print_case_result(case, result, verbose=False):
    """Print individual test case result"""
    status = "✓" if result['success'] else "✗"
    risk_indicator = ""
    task_indicator = ""

    if result['success']:
        expected_risk = case['risk_level']
        actual_risk = result['risk_level']
        risk_diff = abs(expected_risk - actual_risk)

        if risk_diff == 0:
            risk_indicator = "EXACT"
        elif risk_diff == 1:
            risk_indicator = "NEAR"
        else:
            risk_indicator = "MISMATCH"

        # indicate if task was detected correctly
        if result['task_correct']:
            task_indicator = f"✓ (detected: {result['detected_task']}, expected: {case['task']})"
        else:
            task_indicator = f"✗ (detected: {result['detected_task']}, expected: {case['task']})"

        print(f"{status} {risk_indicator} {case['id']} | Task: {task_indicator}")
        print(f"    Expected Risk: {expected_risk}/4  |  Actual Risk: {actual_risk}/4")

        if verbose and isinstance(result['errors'], list) and len(result['errors']) > 0:
            print(f"    Errors Found: {len(result['errors'])}")
            for i, error in enumerate(result['errors'][:2], 1):
                print(f"      {i}. {error.category}")
    else:
        print(f"{status} {case['id']}")
        print(f"    ERROR: {result['error']}")

    print()


def calculate_metrics(results):
    """Calculate performance metrics"""
    total = len(results)
    successful = sum(1 for r in results if r['result']['success'])

    if successful == 0:
        return {
            "total": total,
            "successful": 0,
            "failed": total,
            "exact_accuracy": 0,
            "within_1_accuracy": 0,
            "task_detection_accuracy": 0
        }

    # risk level accuracy
    exact_matches = 0
    within_1 = 0
    task_correct = 0

    for r in results:
        if not r['result']['success']:
            continue

        expected = r['case']['risk_level']
        actual = r['result']['risk_level']
        diff = abs(expected - actual)

        if diff == 0:
            exact_matches += 1
            within_1 += 1
        elif diff == 1:
            within_1 += 1

        # task detection accuracy
        if r['result'].get('task_correct'):
            task_correct += 1

    return {
        "total": total,
        "successful": successful,
        "failed": total - successful,
        "exact_accuracy": exact_matches / successful,
        "within_1_accuracy": within_1 / successful,
        "task_detection_accuracy": task_correct / successful
    }


def print_summary(results):
    """Print summary statistics"""
    metrics = calculate_metrics(results)

    print_header("SUMMARY")

    print(f"\nTotal Tests: {metrics['total']}")
    print(f"  Successful: {metrics['successful']}")
    print(f"  Failed: {metrics['failed']}")

    print(f"\nTask Detection Accuracy: {metrics['task_detection_accuracy']:.1%}")

    print(f"\nRisk Level Accuracy:")
    print(f"  EXACT Match: {metrics['exact_accuracy']:.1%}")
    print(f"  NEAR ±1:   {metrics['within_1_accuracy']:.1%}")

    # breakdown by risk level
    by_risk = defaultdict(lambda: {"total": 0, "exact": 0, "within_1": 0})

    for r in results:
        if not r['result']['success']:
            continue

        expected = r['case']['risk_level']
        actual = r['result']['risk_level']
        diff = abs(expected - actual)

        by_risk[expected]['total'] += 1
        if diff == 0:
            by_risk[expected]['exact'] += 1
            by_risk[expected]['within_1'] += 1
        elif diff == 1:
            by_risk[expected]['within_1'] += 1

    print(f"\nBy Risk Level:")
    for level in sorted(by_risk.keys()):
        stats = by_risk[level]
        if stats['total'] > 0:
            exact_pct = stats['exact'] / stats['total']
            within_pct = stats['within_1'] / stats['total']
            print(f"  Level {level}: {stats['total']} cases | "
                  f"Exact: {exact_pct:.1%} | Within ±1: {within_pct:.1%}")


def main():
    parser = argparse.ArgumentParser(
        description="Run MedVAL validator tests using HuggingFace MedVAL-Bench dataset",
        epilog="""
Environment Setup (choose one):
  Option 1 - Using conda:
    1. conda activate medval
    2. export API_KEY="your-openai-api-key"
    3. python tests/run.py [options]

  Option 2 - Using uv:
    1. export API_KEY="your-openai-api-key"
    2. uv run tests/run.py [options]

Examples with conda:
  python tests/run.py --limit 5
  python tests/run.py --task dialogue2note --risk-level 4
  python tests/run.py --model openai/gpt-4o --output results.json

Examples with uv:
  uv run tests/run.py --limit 5
  uv run tests/run.py --task dialogue2note --risk-level 4
  uv run tests/run.py --model openai/gpt-4o --output results.json
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--risk-level", type=int, choices=[1, 2, 3, 4],
                        help="Test only specific risk level")
    parser.add_argument("--task", type=str,
                        help="Test only specific task type")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Show detailed output")
    parser.add_argument("--output", "-o", type=str,
                        help="Save results to JSON file")
    parser.add_argument("--model", type=str, default="openai/gpt-4o-mini",
                        help="Model to use for validation (default: openai/gpt-4o-mini)")
    parser.add_argument("--limit", type=int,
                        help="Limit number of test cases to run (useful for quick testing)")

    args = parser.parse_args()

    print("Loading MedVAL-Bench test dataset from HuggingFace...")
    test_cases = load_test_data()

    if args.risk_level:
        test_cases = [tc for tc in test_cases if tc['risk_level'] == args.risk_level]
        print(f"Filtered to risk level {args.risk_level}: {len(test_cases)} cases")

    if args.task:
        test_cases = [tc for tc in test_cases if tc['task'] == args.task]
        print(f"Filtered to task {args.task}: {len(test_cases)} cases")

    if args.limit:
        test_cases = test_cases[:args.limit]
        print(f"Limited to first {args.limit} cases")

    if not test_cases:
        print("No test cases match the filters.")
        return

    print(f"\nInitializing validator with model: {args.model}")
    modules = configure_validator(args.model)

    print_header(f"Running {len(test_cases)} Test Cases")

    results = []
    start_time = time.time()

    for i, case in enumerate(test_cases, 1):
        print(f"[{i}/{len(test_cases)}] Testing {case['id']}...")

        result = run_validation(
            modules,
            reference=case['reference'],
            candidate=case['candidate'],
            expected_task=case['task']
        )

        results.append({
            "case": case,
            "result": result
        })

        print_case_result(case, result, verbose=args.verbose)

    elapsed_time = time.time() - start_time

    print_summary(results)

    print(f"\nTotal Time: {elapsed_time:.1f}s")
    print(f"Average Time per Test: {elapsed_time/len(test_cases):.1f}s")

    if args.output:
        output_data = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "model": args.model,
            "metrics": calculate_metrics(results),
            "results": [
                {
                    "id": r['case']['id'],
                    "expected_task": r['case']['task'],
                    "detected_task": r['result'].get('detected_task'),
                    "task_correct": r['result'].get('task_correct'),
                    "expected_risk": r['case']['risk_level'],
                    "actual_risk": r['result'].get('risk_level'),
                    "success": r['result']['success'],
                    "error": r['result'].get('error')
                }
                for r in results
            ]
        }

        with open(args.output, 'w') as f:
            json.dump(output_data, f, indent=2)

        print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
