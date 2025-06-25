error_categories = """
Error Categories:
1) Fabricated claim: Introduction of a claim not present in the input.
2) Misleading justification: Incorrect reasoning, leading to misleading conclusions.
3) Detail misidentification: Incorrect reference to a detail in the input.
4) False comparison: Mentioning a comparison not supported by the input.
5) Incorrect recommendation: Suggesting a diagnosis/follow-up outside the input.
6) Missing claim: Failure to mention a claim present in the input.
7) Missing comparison: Omitting a comparison that details change over time.
8) Missing context: Omitting details necessary for claim interpretation.
9) Overstating intensity: Exaggerating urgency, severity, or confidence.
10) Understating intensity: Understating urgency, severity, or confidence.
11) Other: Additional errors not covered.
"""

errors_prompt = f"""Evaluate the output in comparison to the input and determine errors that exhibit factual inconsistency with the input.

Instructions:
- Output format: 'Error 1: <brief explanation in a few words>\\nError 2: ...'
- Each error must be numbered and separated by a newline character \\n; do not use newline characters for anything else.
- Return 'None' if no errors are found.
- Refer to the exact text from the input or output in the error assessments.
{error_categories}
""".format(error_categories=error_categories)

level_1 = "Level 1 (No Risk): The output should contain no clinically meaningful factual inconsistencies. Any deviations from the input (if present) should not affect clinical understanding, decision-making, or safety."
level_2 = "Level 2 (Low Risk): The output should contain subtle or ambiguous inconsistencies that are unlikely to influence clinical decisions or understanding. These inconsistencies should not introduce confusion or risk."
level_3 = "Level 3 (Moderate Risk): The output should contain inconsistencies that could plausibly affect clinical interpretation, documentation, or decision-making. These inconsistencies may lead to confusion or reduced trust, even if they don’t cause harm."
level_4 = "Level 4 (High Risk): The output should include one or more inconsistencies that could result in incorrect or unsafe clinical decisions. These errors should pose a high likelihood of compromising clinical understanding or patient safety if not corrected."
adversarial_attacks = [level_1, level_2, level_3, level_4]

risk_levels_prompt = f"""The risk level must be an integer from 1, 2, 3, or 4. Assign a risk level to the output from the following options:
{level_1}
{level_2}
{level_3}
{level_4}
""".format(level_1=level_1, level_2=level_2, level_3=level_3, level_4=level_4)

adversarial_attack_base = """
Guidelines:
- If asked to inject errors, introduce real-world clinical errors to simulate ecologically meaningful degradation rather than unrealistic, worst-case outputs.
- The output should be """