import evaluation._env  # must be first -- points at the same eval_chroma_db used by index_benchmark_pdfs.py
from app.pipeline import answer_question

questions = [
    "What are the challenges in estimating output impedance in inverter-based grids?",
    "How do traditional feature selection methods for classification work?",
]

for q in questions:
    print("=" * 60)
    print("Question:", q)
    result = answer_question(q)
    print("\nANSWER:")
    print(result.get("answer", ""))
    print("\nCITATIONS:")
    print(result.get("citations", []))
    print()
