"""
Meta-tuning script that sweeps over sequence lengths.

Dispatches tune.py with different sequence lengths, appending the sequence length
to the study name and passing through all other arguments.
"""

import argparse
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Meta-tuning: sweep sequence lengths and dispatch tune.py",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    # Sequence length sweep arguments
    parser.add_argument(
        "--sequence-lengths",
        type=int,
        nargs="+",
        default=[10, 15, 20, 25, 30, 40],
        help="Sequence lengths to sweep over",
    )
    parser.add_argument(
        "--base-study-name",
        type=str,
        default="cnn_lstm_hyperopt",
        help="Base study name (sequence length will be appended)",
    )
    
    # All other arguments will be passed through to tune.py
    # We'll use parse_known_args to capture only our args and pass the rest through
    
    args, unknown_args = parser.parse_known_args()
    
    # Get the path to tune.py
    script_dir = Path(__file__).parent
    tune_script = script_dir / "tune.py"
    
    if not tune_script.exists():
        print(f"Error: {tune_script} not found", file=sys.stderr)
        sys.exit(1)
    
    print(f"Meta-tuning: Will sweep over sequence lengths: {args.sequence_lengths}")
    print(f"Base study name: {args.base_study_name}")
    print(f"Passing through arguments: {' '.join(unknown_args)}")
    print("=" * 80)
    
    # Run tune.py for each sequence length
    for seq_len in args.sequence_lengths:
        study_name = f"{args.base_study_name}_seq{seq_len}"
        
        print(f"\n{'='*80}")
        print(f"Running tune.py with sequence_length={seq_len}, study_name={study_name}")
        print(f"{'='*80}\n")
        
        # Remove any existing --sequence-length or --study-name from unknown_args
        # (in case user provided them, we want to override)
        filtered_args = []
        skip_next = False
        for arg in unknown_args:
            if skip_next:
                skip_next = False
                continue
            if arg in ["--sequence-length", "--study-name"]:
                skip_next = True
                continue
            filtered_args.append(arg)
        
        # Build command
        cmd = [
            sys.executable,
            "-m", "src.tune",
            "--sequence-length", str(seq_len),
            "--study-name", study_name,
        ] + filtered_args
        
        print(f"Command: {' '.join(cmd)}\n")
        
        # Run the command
        result = subprocess.run(cmd, check=False)
        
        if result.returncode != 0:
            print(
                f"\n{'='*80}",
                f"ERROR: tune.py failed for sequence_length={seq_len}",
                f"Exit code: {result.returncode}",
                f"{'='*80}\n",
                file=sys.stderr,
                sep="\n"
            )
            # Ask user if they want to continue
            response = input(f"Continue with remaining sequence lengths? (y/n): ")
            if response.lower() != 'y':
                print("Aborting meta-tuning.")
                sys.exit(result.returncode)
        else:
            print(f"\n✓ Completed sequence_length={seq_len}\n")
    
    print(f"\n{'='*80}")
    print("Meta-tuning complete!")
    print(f"Completed sequence lengths: {args.sequence_lengths}")
    print(f"\nStudy names created:")
    for seq_len in args.sequence_lengths:
        study_name = f"{args.base_study_name}_seq{seq_len}"
        print(f"  - {study_name}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
