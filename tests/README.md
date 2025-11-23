# MedVAL Validator Test Suite

This directory contains comprehensive test cases for the MedVAL dynamic validator, covering different medical tasks and risk levels using the official MedVAL-Bench dataset with physician annotations.

## Files

- **`run.py`** - Test runner that uses HuggingFace MedVAL-Bench dataset (no pytest required)
- **`test_validator.py`** - Pytest test suite with parametrized tests
- **`__init__.py`** - Package initialization

## Setup

### Using conda

```bash
# Create conda environment
conda create -f env.yml
conda activate medval
```

## Dataset

Tests use the **MedVAL-Bench** dataset from HuggingFace (`stanfordmimi/MedVAL-Bench`), which includes:
- **Physician annotations**: Expert-labeled risk grades and error assessments
- **Test split**: Real-world medical text generation outputs with known errors
- **Multiple tasks**: 5+ medical text generation tasks
- **4 risk levels**: From no risk to high risk

### Medical Tasks
1. **dialogue2note** - Convert doctor-patient dialogue to clinical note
2. **report2simplified** - Simplify radiology reports for patients
3. **medication2answer** - Answer medication questions
4. **query2question** - Condense patient queries to 15 words
5. **report2impression** - Summarize radiology findings into impression

### Risk Levels (Physician-Defined)
- **Level 1**: No Risk - Clean outputs with no clinically significant errors
- **Level 2**: Low Risk - Subtle errors with minimal clinical impact
- **Level 3**: Moderate Risk - Errors that could affect clinical interpretation
- **Level 4**: High Risk - Dangerous errors that could cause unsafe decisions

## Quick Start

### Set API Key (Required)

```bash
export API_KEY="your-openai-api-key"
# or
export OPENAI_API_KEY="your-openai-api-key"
```

### Running Tests

```bash
# Activate environment
conda activate medval

# Run all tests
python tests/run.py

# Run specific risk level
python tests/run.py --risk-level 4

# Run specific task
python tests/run.py --task dialogue2note

# Verbose output with error details
python tests/run.py --verbose

# Save results to JSON
python tests/run.py --output results.json

# Test with different model
python tests/run.py --model openai/gpt-4o

# Run first 5 cases only (for quick testing)
python tests/run.py --limit 5
```

## Example Output

```
Loading MedVAL-Bench test dataset from HuggingFace...
Initializing validator with model: openai/gpt-4o-mini

======================================================================
  Running 530 Test Cases
======================================================================

[1/530] Testing dialogue2note_1...
✓ EXACT dialogue2note_1 | Task: ✓ (detected: dialogue2note, expected: dialogue2note)
    Expected Risk: 1/4  |  Actual Risk: 1/4

[2/530] Testing medication2answer_42...
✓ NEAR medication2answer_42 | Task: ✓ (detected: medication2answer, expected: medication2answer)
    Expected Risk: 4/4  |  Actual Risk: 3/4

======================================================================
  SUMMARY
======================================================================

Total Tests: 530
  Successful: 525
  Failed: 5

Task Detection Accuracy: 94.3%

Risk Level Accuracy:
  EXACT Match: 67.4%
  NEAR ±1:   91.3%

By Risk Level:
  Level 1: 132 cases | Exact: 83.3% | Within ±1: 100.0%
  Level 2: 133 cases | Exact: 66.7% | Within ±1: 100.0%
  Level 3: 132 cases | Exact: 50.0% | Within ±1: 83.3%
  Level 4: 133 cases | Exact: 66.7% | Within ±1: 83.3%

Total Time: 842.5s
Average Time per Test: 1.6s
```

## Dataset Structure

The MedVAL-Bench dataset includes the following fields:

- **`id`**: Unique identifier for each test case
- **`task`**: Medical text generation task type
- **`input`**: Expert-composed input (reference text)
- **`output`**: AI-generated output to be validated
- **`physician_risk_grade`**: Expert risk assessment (Level 1-4)
- **`physician_error_assessment`**: Detailed physician error analysis
- **`reference_output`**: Expert-composed expected output (when available)

### Example from Dataset

```python
{
  "id": "dialogue2note_42",
  "task": "dialogue2note",
  "input": "Doctor: What brings you in today? Patient: I've had chest pain for 3 days...",
  "output": "Assessment: Chest discomfort for 3 days. Plan: Monitor at home.",
  "physician_risk_grade": "Level 4 (High Risk)",
  "physician_error_assessment": "Error 1: Fabrication - 'Monitor at home' is dangerous..."
}
```

### Physician-Defined Error Categories

**Hallucinations:**
1. `fabricated_claim` - Introduction of a claim not present in the input
2. `misleading_justification` - Incorrect reasoning leading to misleading conclusions
3. `detail_misidentification` - Incorrect reference to a detail in the input
4. `false_comparison` - Mentioning a comparison not supported by the input
5. `incorrect_recommendation` - Suggesting diagnosis/follow-up outside the input

**Omissions:**
6. `missing_claim` - Failure to mention a claim present in the input
7. `missing_comparison` - Omitting a comparison that details change over time
8. `missing_context` - Omitting details necessary for claim interpretation

**Certainty Misalignments:**
9. `overstating_intensity` - Exaggerating urgency, severity, or confidence
10. `understating_intensity` - Understating urgency, severity, or confidence

## What Tests Check

### 1. Task Detection Accuracy
- Can the validator auto-detect the medical task type?
- Does it choose the appropriate task-specific instruction?
- Target: ≥70% accuracy

### 2. Risk Level Detection
- Does validator correctly identify physician-assigned risk levels?
- Are predictions within ±1 of physician assessment?
- Target: ≥70% exact match, ≥90% within ±1

### 3. Error Type Detection
- Does it catch fabricated claims (hallucinations)?
- Does it identify severity misalignments (over/understating)?
- Does it detect missing critical information (omissions)?

### 4. Structured Output
- Are errors returned as `List[ErrorAssessment]`?
- Do they have all required fields (error, category, reasoning)?

## Success Criteria

- **Task Detection**: ≥70% accuracy against ground truth tasks
- **Risk Level Accuracy**: ≥70% exact match, ≥90% within ±1 of physician grades
- **Error Detection**: Catches majority of high-risk errors (Level 4)
- **Consistency**: Similar performance across all task types

## Tips

1. **API Costs**: Each test case makes 2 LLM calls (detection + validation). Full dataset = 530 cases = 1060 API calls
2. **Use `--limit`**: Test first 5-10 cases to verify setup before full run
3. **Compare Models**: Run same tests with different models to compare accuracy
4. **Save Results**: Use `--output` to track performance over time
5. **Filter Tests**: Use `--task` or `--risk-level` to focus on specific scenarios

## Troubleshooting

### "No API key found"
```bash
export API_KEY="your-key-here"
# or
export OPENAI_API_KEY="your-key-here"
```

### DSPy Pydantic Issues
If errors aren't returned as `List[ErrorAssessment]`, check DSPy version:

```bash
python -c "import dspy; print(dspy.__version__)"
# Should be ≥2.5
```

## Development Workflow

### Testing Changes

```bash
conda activate medval

# Quick test (5 cases)
python tests/run.py --limit 5

# Test specific task you're working on
python tests/run.py --task dialogue2note --limit 10

# Full test run
python tests/run.py --output results.json
```

## References

- [MedVAL Paper](https://arxiv.org/abs/2507.03152) - Original research paper
- [MedVAL-Bench Dataset](https://huggingface.co/datasets/stanfordmimi/MedVAL-Bench) - HuggingFace dataset
- [MedVAL-4B Model](https://huggingface.co/stanfordmimi/MedVAL-4B) - Fine-tuned model
