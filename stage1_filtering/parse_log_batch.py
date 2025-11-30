"""
Batch log parser for case_10.log ~ case_20.log
Uses Drain algorithm to parse Android logs
"""
import os
import sys

# Add current directory to path for Drain import
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from Drain import LogParser

# Configuration
input_dir = '/home/bigdata/1113/lorin-proto/data/logs'
output_dir = '/home/bigdata/1113/lorin-proto/data/logs'
dataset = 'Android'

# Android log format
android_format = r'<Date> <Time>  <Pid>  <Tid> <Level> <Component>: <Content>'

# Android regex patterns
Android_regex = [
    r"(/[\w-]+)+",
    r"([\w-]+\.){2,}[\w-]+",
    r"\b(\-?\+?\d+)\b|\b0[Xx][a-fA-F\d]+\b|\b[a-fA-F\d]{4,}\b",
]

threshold = 5
delimeter = [r""]

def parse_case_logs():
    """Parse case_10.log through case_20.log"""

    # Create parser
    parser = LogParser(
        logname=dataset,
        log_format=android_format,
        indir=input_dir,
        outdir=output_dir,
        threshold=threshold,
        delimeter=delimeter,
        rex=Android_regex
    )

    # Process each case file
    for case_num in range(10, 21):
        log_file = f'case_{case_num}.log'
        log_path = os.path.join(input_dir, log_file)

        if os.path.exists(log_path):
            print(f"\n{'='*50}")
            print(f"Processing: {log_file}")
            print(f"{'='*50}")

            try:
                parser.parse(log_file)
                print(f"Successfully parsed: {log_file}")
            except Exception as e:
                print(f"Error parsing {log_file}: {e}")
        else:
            print(f"File not found: {log_path}")

if __name__ == "__main__":
    parse_case_logs()
    print("\n" + "="*50)
    print("Batch parsing completed!")
    print("="*50)
