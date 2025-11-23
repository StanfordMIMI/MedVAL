import dspy
from utils.prompts import errors_prompt, risk_levels_prompt, task_keys, instruction_mappings_prompt
from typing import Literal, List
from pydantic import BaseModel, Field

class ErrorAssessment(BaseModel):
    error_occurrence: str = Field(description="The exact snippet of text in the candidate where the error occurred")
    error: str = Field(description="Why this is an error")
    category: str = Field(description="Error category from the 11 categories")
    reasoning: str = Field(description="The reasoning why this part of the candidate is an error")

class DetectTask(dspy.Signature):
    """
    Detect the intended task from the reference text and the generated candidate
    """
    reference: str = dspy.InputField()
    candidate: str = dspy.InputField()
    task: Literal[*task_keys] = dspy.OutputField(description=instruction_mappings_prompt)

class MedVAL_Validator(dspy.Signature):
    """
    Evaluate a candidate in comparison to the reference composed by an expert.
    
    Instructions:
    1. Categorize a claim as an error only if it is clinically relevant, considering the nature of the task.
    2. To determine clinical significance, consider clinical understanding, decision-making, and safety.
    3. Some tasks (e.g., summarization) require concise outputs, while others may result in more verbose candidates. 
       - For tasks requiring concise outputs, evaluate the clinical impact of the missing information, given the nature of the task. 
       - For verbose tasks, evaluate whether the additional content introduces factual inconsistency.
    """
    instruction: str = dspy.InputField()
    reference: str = dspy.InputField()
    candidate: str = dspy.InputField()
    errors: str = dspy.OutputField(description=errors_prompt)
    structured_errors: List[ErrorAssessment] = dspy.OutputField(description=errors_prompt)
    risk_level: Literal[1, 2, 3, 4] = dspy.OutputField(description=risk_levels_prompt)