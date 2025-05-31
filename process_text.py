import pandas as pd
import random
from tqdm import tqdm

def process_file(input_filename="unrelated_text.txt", lines_per_file=30000):
    """
    Shuffles the content of the input text file globally, then splits it into multiple CSV files
    based on the specified number of lines. Adds a 'class' column with all values set to 0.

    Args:
        input_filename (str): The name of the input txt file.
        lines_per_file (int): The number of lines per output CSV file.
    """
    try:
        with open(input_filename, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f"Error: File {input_filename} not found.")
        return
    except Exception as e:
        print(f"Error reading file: {e}")
        return

    # Shuffle lines globally before chunking
    random.shuffle(lines)

    num_lines = len(lines)
    num_output_files = (num_lines + lines_per_file - 1) // lines_per_file

    print(f"Total lines: {num_lines}")
    print(f"Lines per file: {lines_per_file}")
    print(f"Will generate {num_output_files} CSV files.")

    for i in tqdm(range(num_output_files), desc="Processing files progress"):
        start_index = i * lines_per_file
        end_index = min((i + 1) * lines_per_file, num_lines)
        chunk_lines = lines[start_index:end_index]

        # Remove newline characters from the end of lines
        cleaned_chunk_lines = [line.strip() for line in chunk_lines]

        # Create DataFrame
        # Note: .sample(frac=1) is no longer needed here as lines are already globally shuffled
        df = pd.DataFrame(cleaned_chunk_lines, columns=['contect'])
        df['class'] = 0

        output_filename = f"unrelated_text_part_{i+1}.csv"
        try:
            df.to_csv(output_filename, index=False, encoding='utf-8')
        except Exception as e:
            print(f"Error writing file {output_filename}: {e}")

if __name__ == "__main__":
    process_file()
