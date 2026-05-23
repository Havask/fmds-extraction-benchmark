import json
import argparse
import os
from typing import Dict, List, Any, Tuple, Optional

def calculate_field_metrics(output_cases: List[Dict[str, Any]], gold_cases: List[Dict[str, Any]]) -> Tuple[Dict[str, Dict[str, float]], float]:
    field_metrics = {}
    all_fields = set(
        key for case in gold_cases for key in case.keys() if key not in ('case_number', 'skadekommune')
    )

    for field in all_fields:
        correct = 0
        total = 0
        for out_case, gold_case in zip(output_cases, gold_cases):
            if out_case.get(field) == gold_case.get(field):
                correct += 1
            total += 1
        field_accuracy = correct / total if total > 0 else 0
        field_metrics[field] = {"accuracy": field_accuracy}

    average_accuracy = sum(m["accuracy"] for m in field_metrics.values()) / len(field_metrics) if field_metrics else 0.0
    return field_metrics, average_accuracy

def calculate_overall_accuracy(output_cases: List[Dict[str, Any]], gold_cases: List[Dict[str, Any]]) -> float:
    total_cases = len(gold_cases)
    if total_cases == 0:
        return 0.0

    exact_matches = 0
    for out_case, gold_case in zip(output_cases, gold_cases):
        fields_to_check = {key for key in gold_case.keys() if key not in ('case_number', 'skadekommune')}
        if not fields_to_check:
            continue
        if all(out_case.get(key) == gold_case.get(key) for key in fields_to_check):
            exact_matches += 1

    return exact_matches / total_cases

def compare_json_files(
    output_file: str,
    gold_standard_file: str,
    model_name: str,
    performance_stats: Optional[Dict[str, float]] = None,
    report_filename: Optional[str] = None
):
    try:
        with open(output_file, 'r', encoding='utf-8') as f_out:
            output_data = json.load(f_out)
        with open(gold_standard_file, 'r', encoding='utf-8') as f_gold:
            gold_data = json.load(f_gold)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error reading files: {e}")
        return

    output_cases = output_data.get("cases", [])
    gold_cases = gold_data.get("cases", [])

    mismatch_lines = []
    for i, (out_case, gold_case) in enumerate(zip(output_cases, gold_cases)):
        case_id = out_case.get("case_number", f"Case {i+1}")
        for key in gold_case.keys():
            if key in ('case_number', 'skadekommune'):
                continue
            if out_case.get(key) != gold_case.get(key):
                mismatch_lines.append(
                    f"Case: {case_id}, Field: '{key}'\n"
                    f"  Output: {out_case.get(key)}\n"
                    f"  Gold:   {gold_case.get(key)}\n"
                    f"{'-'*20}"
                )

    overall_accuracy = calculate_overall_accuracy(output_cases, gold_cases)
    field_accuracies, average_field_accuracy = calculate_field_metrics(output_cases, gold_cases)

    metrics_lines = []
    metrics_lines.append("="*60)
    metrics_lines.append("              PERFORMANCE & ACCURACY METRICS REPORT")
    metrics_lines.append("="*60 + "\n")

    if performance_stats:
        metrics_lines.append("--- Performance Metrics (seconds per case) ---")
        for key, value in performance_stats.items():
            if key == "N (Count)":
                 metrics_lines.append(f"  {key:<20} {int(value)}")
            else:
                 metrics_lines.append(f"  {key:<20} {value:.4f}")
        metrics_lines.append("")

    metrics_lines.append("--- Overall Exact-Match Accuracy ---")
    metrics_lines.append(f"{overall_accuracy:.4f} (percentage of perfectly correct cases)\n")

    metrics_lines.append("--- Average Field Accuracy ---")
    metrics_lines.append(f"{average_field_accuracy:.4f} (the average of all per-field accuracy scores)\n")

    metrics_lines.append("--- Accuracy (Per Field) ---")
    if not field_accuracies:
        metrics_lines.append("  No fields found to evaluate.")
    else:
        for field, scores in sorted(field_accuracies.items()):
            metrics_lines.append(f"  {field:<25} {scores['accuracy']:.4f}")
    
    metrics_lines.append("\n" + "="*60)

    full_report = "\n".join(metrics_lines + ["\n\n--- MISMATCH DETAILS ---\n"] + mismatch_lines)

    if report_filename:
        final_report_path = report_filename
    else:
        fallback_dir = os.path.join("output", "reports")
        os.makedirs(fallback_dir, exist_ok=True)
        final_report_path = os.path.join(fallback_dir, f"{model_name}_report.txt")

    with open(final_report_path, "w", encoding="utf-8") as f:
        f.write(full_report)

    print(f"Report for '{model_name}' has been generated at: {final_report_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Compare a model's JSON output with a gold standard JSON file.")
    parser.add_argument("output_file", help="Path to the model's output JSON file.")
    parser.add_argument("gold_file", help="Path to the gold standard JSON file.")
    parser.add_argument("-n", "--name", dest="model_name", default="model", help="Name of the model for reporting purposes.")
    
    args = parser.parse_args()
    
    compare_json_files(args.output_file, args.gold_file, args.model_name)