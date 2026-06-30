from evaluation.run_retrieval_eval import main as run_retrieval_eval
from evaluation.generate_report import main as generate_report


def main():
    print("Starting full retrieval evaluation...")

    run_retrieval_eval()
    generate_report()

    print("Full retrieval evaluation finished.")


if __name__ == "__main__":
    main()