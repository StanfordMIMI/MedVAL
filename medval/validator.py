import dspy
from utils.prompts import errors_prompt, risk_levels_prompt
from typing import Literal

class MedVAL_Validator(dspy.Signature):
    """
    Evaluate the output in comparison to the input composed by an expert.
    
    Instructions:
    1. Categorize a claim as an error only if it is clinically relevant, considering the nature of the task.
    2. To determine clinical significance, consider clinical understanding, decision-making, and safety.
    3. Some tasks (e.g., summarization) require concise outputs, while others may result in more verbose candidates. 
       - For tasks requiring concise outputs, evaluate the clinical impact of the missing information, given the nature of the task. 
       - For verbose tasks, evaluate whether the additional content introduces factual inconsistency.
    """
    instruction: str = dspy.InputField()
    input: str = dspy.InputField()
    output: str = dspy.InputField()
    errors: str = dspy.OutputField(description=errors_prompt)
    risk_level: Literal[1, 2, 3, 4] = dspy.OutputField(description=risk_levels_prompt)