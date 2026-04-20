# FMDS extraction experiment 


This test is designed to compare LLM against real human FMDS schema


Just run every single model and return the results to me

Make it run 3 times and do the evaluation script on it and append it to a text file

source venv-mistral3/bin/activate

# Inference Script

## Usage

```bash
python src/test1.py <model>
```

## Base Models: works

```bash
python src/test1.py ministral-3b
python src/test1.py ministral-8b
python src/test1.py ministral-14b
python src/test1.py qwen3-4b
python src/test1.py granite-8b
python src/test1.py llama3-8b
python src/test1.py mistral-7b
python src/test1.py phi-4
```

## Fine-tuned (LoRA)

```bash
python src/test1.py ministral-3b-lora #funker ikke
python src/test1.py ministral-8b-lora #Funker ikke 
python src/test1.py ministral-14b-lora #Funker ikke 
python src/test1.py qwen3-4b-lora
python src/test1.py granite-8b-lora
python src/test1.py llama3-8b-lora. # funker ikke
python src/test1.py mistral-7b-lora 
```

## Output

- **JSON:** `output/<model>/output/<timestamp>_output.json`
- **Report:** `output/<model>/report/<timestamp>_report.txt`