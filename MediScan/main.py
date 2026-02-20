import argparse
from src.pipelines.pipeline import full_pipeline
from src.utils.test import test_run
from src.app_runner import run_app

def main():
    parser = argparse.ArgumentParser(description="MediScan CLI")

    parser.add_argument(
        "command",
        choices=["train", "test", "app"],
        help="Command to execute"
    )

    args = parser.parse_args()

    if args.command == "train":
        print("Running training pipeline...")
        full_pipeline()

    elif args.command == "test":
        print("Running model tests...")
        test_run()

    elif args.command == "app":
        print("Launching Streamlit app...")
        run_app()

if __name__ == "__main__":
    main()
