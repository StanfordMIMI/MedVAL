error_categories = """
Error Categories:
1) Fabricated claim: Introduction of a claim not present in the reference.
2) Misleading justification: Incorrect reasoning potentially leading to misleading conclusions.
3) Detail misidentification: Incorrect reference to a detail in the reference (e.g., body part, finding).
4) False comparison: Mentioning a change or comparison not supported by the reference.
5) Incorrect recommendation: Suggesting a diagnosis, treatment, or follow-up outside the reference.
6) Missing claim: Failure to mention a claim present in the reference.
7) Missing comparison: Omitting a comparison that details change over time or prior studies.
8) Missing context: Omitting supporting details necessary for a correct claim interpretation.
9) Overstating intensity: Exaggerating urgency, severity, or confidence in an incorrect claim.
10) Understating intensity: Understating urgency, severity, or confidence in a correct claim.
11) Other: Additional errors not covered in the defined categories.
"""

errors_prompt = f"""Evaluate the candidate in comparison to the reference and determine errors that exhibit factual inconsistency with the reference.

Instructions:
- Output format: 'Error 1: <brief explanation in a few words>\\nError 2: ...'
- Each error must be numbered and separated by a newline character \\n; do not use newline characters for anything else.
- Return 'None' if no errors are found.
- Refer to the exact text from the candidate/reference in the error outputs.
{error_categories}
""".format(error_categories=error_categories)

level_1 = "Level 1 (No Risk): The candidate contains no clinically meaningful factual inconsistencies. Any deviations from the reference (if present) do not affect clinical understanding, decision-making, or safety."
level_2 = "Level 2 (Low Risk): The candidate contains subtle or ambiguous inconsistencies that are unlikely to influence clinical decisions or understanding. These inconsistencies do not introduce confusion or risk."
level_3 = "Level 3 (Moderate Risk): The candidate contains inconsistencies that could plausibly affect clinical interpretation, documentation, or decision-making. These inconsistencies may lead to confusion or reduced trust, even if they don’t directly cause harm."
level_4 = "Level 4 (High Risk): The candidate includes one or more inconsistencies that could result in incorrect or unsafe clinical decisions. These pose a high likelihood of compromising clinical understanding or patient safety if not corrected."
adversarial_attacks = [level_1, level_2, level_3, level_4]

risk_levels_prompt = f"""Your output must be an integer from 1, 2, 3, or 4. Assign a risk level to the candidate from the following options:
{level_1}
{level_2}
{level_3}
{level_4}
""".format(level_1=level_1, level_2=level_2, level_3=level_3, level_4=level_4)

adversarial_attack_base = """
Guidelines:
- If asked to inject errors, introduce real-world clinical errors to simulate ecologically meaningful degradation rather than unrealistic, worst-case outputs.
- The candidate should be """