import json
import argparse
from typing import Dict, List, Any
from models.gpt_azure import generate_response_azure_gpt
from models.azure import generate_response_azure
from models.other_models.gemini import generate_response_gemini
from models.other_models.claude_azure import generate_response_azure_claude
from evaluate import compare_json_files
import sys
import time
import statistics
import os
from datetime import datetime


def create_prompt(args) -> str:
    if args.few_shot:
        with open('data/prompts/few_shot.json', 'r') as file:
            prompt = json.load(file)
            return prompt
    else:
        with open('data/prompts/prompt.json', 'r') as file:
            prompt = json.load(file)
            return prompt

def save_json_output(filename: str, data: Dict[str, Any]):
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
        print(f"Successfully saved output to {filename}")
    except IOError as e:
        print(f"Error writing to file {filename}: {e}")

def main():
    parser = argparse.ArgumentParser(description="Run an NLP model extraction experiment.")
    parser.add_argument("model", type=str, help="The model to test (e.g., 'chatgpt').")
    parser.add_argument("--few_shot", action="store_true", help="Enable few-shot prompting.")
    
    args = parser.parse_args()
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    model_dir = os.path.join("output", args.model)
    output_subdir = os.path.join(model_dir, "validation_set_outputs")
    report_subdir = os.path.join(model_dir, "validation_set_reports")

    os.makedirs(output_subdir, exist_ok=True)
    os.makedirs(report_subdir, exist_ok=True)
    
    print(f"Starting experiment with model: {args.model}")
    print(f"Outputs will be saved in: {output_subdir}")
    print(f"Reports will be saved in: {report_subdir}")
    print("-" * 30)
    
    prompt = create_prompt(args)
    print("Creating prompt:")

    try:
        with open('data/training_data/case.json', 'r') as file:
            cases_data = json.load(file)
            cases_list = cases_data.get("cases", [])
    except FileNotFoundError:
        print("Error: The cases was not found.")
        return
    
    all_cases = []
    processing_times = [] 

    for case in cases_list:
        print(f"\nProcessing {case['case']}...")
        start_time = time.time()
        
        if args.model in ["gpt-5", "gpt-5-mini", "gpt-5.2-chat", "Kimi-K2-Thinking"]:
            model_output = generate_response_azure_gpt(args.model, prompt, case)
        elif args.model == "gemini":
            model_output = generate_response_gemini(prompt, case)
        elif args.model in ["claude-opus-4-5", "claude-sonnet-4-5", "claude-haiku-4-5"]:
            print("Running claude")
            model_output = generate_response_azure_claude(args.model, prompt, case)
            print(model_output)
        else:
            print("running", args.model)
            model_output = generate_response_azure(args.model, prompt, case)
            print(model_output)
        
        end_time = time.time()
        duration = end_time - start_time
        processing_times.append(duration)
        print(f"Time taken: {duration:.2f} seconds")

        if isinstance(model_output, list) and len(model_output) > 0:
            case_data = model_output[0]
            all_cases.append(case_data)
        elif isinstance(model_output, dict) and "cases" in model_output and isinstance(model_output["cases"], list) and len(model_output["cases"]) > 0:
            case_data = model_output["cases"][0]
            all_cases.append(case_data)
        else:
            print(f"Warning: Unexpected model output format for {case['case']}. Skipping.")

    print("\n")

    performance_stats = {}
    if len(processing_times) > 1:
        quantiles = statistics.quantiles(processing_times, n=4) # Q1, Median, Q3
        performance_stats = {
            "N (Count)": len(processing_times),
            "Mean": statistics.mean(processing_times),
            "Median": statistics.median(processing_times),
            "Standard Deviation": statistics.stdev(processing_times),
            "Min": min(processing_times),
            "Max": max(processing_times),
            "IQR": quantiles[2] - quantiles[0]
        }
        print("--- Performance Summary ---")
        for key, value in performance_stats.items():
            print(f"  {key:<20} {value:.4f}")
        print("-" * 27)

    final_output = {"cases": all_cases}

    output_filename = os.path.join(output_subdir, f"output_{timestamp}.json")
    report_filename = os.path.join(report_subdir, f"report_{timestamp}.txt")
    save_json_output(output_filename, final_output)

    compare_json_files(
        output_filename,
        "data/training_data/gold_standard.json",
        args.model,
        performance_stats=performance_stats,
        report_filename=report_filename
    )
    
    print("-" * 30)
    print(f"Experiment with {args.model} completed.")

if __name__ == "__main__":
    main()