import dspy

class MedVAL_Generator(dspy.Signature):
    """
    Generate an output, given the input composed by an expert.
    """
    instruction: str = dspy.InputField()
    input: str = dspy.InputField()
    output: str = dspy.OutputField(description="Only respond with the output, do not include any additional text or explanation.")