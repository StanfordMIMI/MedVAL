import dspy

class MedVAL_Generator(dspy.Signature):
    """
    Generate a candidate, given the reference composed by an expert.
    """
    instruction: str = dspy.InputField()
    reference: str = dspy.InputField()
    candidate: str = dspy.OutputField(description="Only respond with the candidate, do not include any additional text or explanation.")