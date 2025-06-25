# MedVAL: Expert-level validation of AI-generated medical text

[![arXiv](https://img.shields.io/badge/arXiv-ARXIV_ID-b31b1b.svg?style=for-the-badge)](https://arxiv.org/abs/ARXIV_ID)          [![Hugging Face](https://huggingface.co/datasets/huggingface/badges/resolve/main/model-on-hf-md.svg)](https://huggingface.co/stanfordmimi/MedVAL-4B)          [![Hugging Face](https://huggingface.co/datasets/huggingface/badges/resolve/main/dataset-on-hf-md.svg)](https://huggingface.co/datasets/stanfordmimi/MedVAL-Bench/)          [![License](https://img.shields.io/github/license/stanfordmimi/medvae?style=for-the-badge)](LICENSE)

Official implementation for [Towards expert-level validation of AI-generated medical text with language models](https://arxiv.org/abs/ARXIV_ID).

![Overview](assets/fig.png)

## 🏥 What is MedVAL?

MedVAL is a comprehensive framework for expert-level validation of AI-generated medical text using language models. The system is designed to evaluate the quality, accuracy, and safety of AI-generated medical text across multiple healthcare tasks. The framework supports both model fine-tuning and zero-shot evaluation.

## ⚡️ Installation

### Environment Setup

Create and activate the conda environment:

```bash
conda env create -f env.yml
conda activate medval
```

## 🚀 Usage Instructions

MedVAL supports four main usage scenarios. Each scenario is configured through YAML files and executed via the main `run.py` script.

## Evaluation

```bash
python run.py --config=test
```

### 1. API-based Models

For evaluating API-based models (OpenAI, Anthropic, Gemini, etc.):

**Configuration (`configs/test.yaml`)**:
```yaml
tasks: [dialogue2note, medication2answer, query2question, report2impression]
data: test
method: zero-shot # [zero-shot, finetune]

n_samples: null
debug: False

model: openai/gpt-4o-mini
api_base: null
api_key: ${API_KEY}
local_model_path: null
```

### 2. Local/Huggingface Models

For using local or HuggingFace models:

**Configuration**:
```yaml
tasks: [dialogue2note, medication2answer, query2question, report2impression]
data: test
method: zero-shot # [zero-shot, finetune]

n_samples: null
debug: False

model: local/MODEL_NAME
api_base: null
api_key: null
local_model_path: /path/to/local/model
```

## Fine-Tuning

```bash
python run.py --config=train
```

### 3. API-based Teacher Models

For fine-tuning a local student model using an API-based teacher model:

```bash
python run.py --config=train
```

**Configuration**:
```yaml
tasks: [report2simplified, query2question, report2impression, report2simplified]
data: train
method: finetune

n_samples: null
debug: False
num_threads: 16
num_epochs: 5
threshold: 0.95

model: openai/gpt-4o-mini
api_base: null
api_key: ${API_KEY}

student_model: local/STUDENT_MODEL_NAME
local_model_path: /path/to/student/model
```

### 4. Local/Huggingface Models

For fine-tuning a local student model using a local teacher model:

**Configuration (`configs/train.yaml`)**:
```yaml
tasks: [report2simplified, query2question, report2impression, report2simplified]
data: train
method: finetune

n_samples: null
debug: False
num_threads: 16
num_epochs: 5
threshold: 0.95

model: local/TEACHER_MODEL_NAME
api_base: null
api_key: null

student_model: local/STUDENT_MODEL_NAME
local_model_path: /path/to/student/model
```

## 🔧 API Model Configurations

### OpenAI
```yaml
model: openai/MODEL_NAME
api_base: null
api_key: ${OPENAI_API_KEY}
```

### Gemini
```yaml
model: gemini/MODEL_NAME
api_base: null
api_key: ${GEMINI_API_KEY}
```

### Anthropic
```yaml
model: anthropic/MODEL_NAME
api_base: null
api_key: ${ANTHROPIC_API_KEY}
```

### SGLang
```yaml
model: openai/HUGGINGFACE_MODEL_NAME
api_base: http://SERVER_IP:PORT/v1
api_key: local
```

### Ollama
```yaml
model: ollama_chat/MODEL_NAME
api_base: http://SERVER_IP:PORT
api_key: null
```

## 📊 Dataset

MedVAL uses the MedVAL-Bench dataset, automatically loaded from HuggingFace:

```python
from datasets import load_dataset
dataset = load_dataset("stanfordmimi/MedVAL-Bench")
```

The dataset includes training and test splits with multiple medical text validation tasks.

## 🎯 Configuration Parameters

### Core Parameters
- `tasks`: List of tasks for training/evaluation
- `data`: Dataset split (`train` or `test`)
- `method`: Evaluation method (`zero-shot` or `finetune`)
- `n_samples`: Number of samples to process (null for all)
- `debug`: Enable debug mode for detailed output

### Model Parameters
- `model`: Model identifier (API or local)
- `api_base`: API endpoint URL
- `api_key`: API key (use `${ENV_VAR}` for environment variables)
- `local_model_path`: Path to local model files

### Fine-tuning Parameters
- `student_model`: Student model for fine-tuning
- `num_threads`: Number of threads for training
- `num_epochs`: Training epochs
- `threshold`: Filtering threshold

## 📈 Results

Results are automatically saved to the `results/` directory with the following structure:
```
results/
├── zero-shot/
│   └── model_name.csv
└── finetune/
    └── model_name.csv
```

Each results file contains:
- Input text and outputs
- Model predictions and reasoning
- Error assessments and risk grades
- Physician assessments (ground-truth) for comparison

## 🏗️ Project Structure

```
MedVAL/
├── configs/          # Configuration files
├── medval/           # Core package
│   ├── pipeline.py   # Main MedVAL pipeline
│   ├── generator.py  # Text generation module
│   └── validator.py  # Validation module
├── utils/            # Utility functions and prompts
├── agents/           # Fine-tuned model storage
├── results/          # Evaluation results
└── run.py            # Main execution script
```

## 📎 Citation

If you find this repository useful for your work, please cite the following paper:

```bibtex
@misc{PLACEHOLDER_CITATION_KEY,
      title={MedVAL: Towards Expert-Level Validation of AI-Generated Medical Text with Language Models}, 
      author={PLACEHOLDER_AUTHORS},
      year={PLACEHOLDER_YEAR},
      eprint={PLACEHOLDER_ARXIV_ID},
      archivePrefix={arXiv},
      primaryClass={PLACEHOLDER_PRIMARY_CLASS},
      url={PLACEHOLDER_ARXIV_URL}, 
}
```

## 📄 License

This project is licensed under the terms specified in the [LICENSE](LICENSE) file.

## 🤝 Contributing

We welcome contributions to improve MedVAL! Please feel free to submit issues, feature requests, or pull requests.

## 🙏 Acknowledgments

This repository is built using [DSPy](https://github.com/stanfordnlp/dspy) for language model fine-tuning.
