import argparse
from pid_tuner.train import train_model
from pid_tuner.predict import diagnose_log

def main():
    parser = argparse.ArgumentParser(description="AI PID Tuning Assistant")
    subparsers = parser.add_subparsers(dest="command")
    
    # Train
    train_parser = subparsers.add_parser("train", help="Train the model")
    train_parser.add_argument("--data", required=True, help="Path to dataset folder")
    train_parser.add_argument("--output", default="models/pid_classifier.pkl", help="Model output path")
    
    # Diagnose
    diag_parser = subparsers.add_parser("diagnose", help="Diagnose a log file")
    diag_parser.add_argument("--log", required=True, help="Path to flight log CSV")
    diag_parser.add_argument("--model", default="models/pid_classifier.pkl", help="Path to trained model")
    
    args = parser.parse_args()
    
    if args.command == "train":
        train_model(args.data, args.output)
        
    elif args.command == "diagnose":
        results = diagnose_log(args.log, args.model)
        print("\n--- PID Diagnosis Report ---")
        for res in results:
            print(f"\nAxis: {res['axis']}")
            print(f"  Diagnosis:   {res['diagnosis']} (Conf: {res['confidence']})")
            print(f"  Oscillation: {'YES' if res['oscillation'] else 'No'}")
            print(f"  Advice:      {res['advice']}")
        print("\n----------------------------")
    else:
        parser.print_help()

if __name__ == "__main__":
    main()