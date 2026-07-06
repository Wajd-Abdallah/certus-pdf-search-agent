from evaluation.full_eval import llm_judge_score

question = "What are the challenges in estimating output impedance in inverter-based grids?"
reference = "Estimating output impedance in inverter-based grids is challenging due to dynamic grid conditions, which require real-time estimation."

good_answer = "Estimating output impedance is difficult because grid conditions change dynamically, requiring real-time estimation methods."
bad_answer = "The capital of France is Paris, and bananas are a good source of potassium."

print("Score for GOOD answer (expect high, e.g. 4-5):", llm_judge_score(question, good_answer, reference))
print("Score for BAD/irrelevant answer (expect low, e.g. 1-2):", llm_judge_score(question, bad_answer, reference))
