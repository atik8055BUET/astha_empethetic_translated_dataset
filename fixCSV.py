import csv
import re
import pandas as pd
from io import StringIO

def fix_csv_file(input_file, output_file):
    """
    Fix CSV file with delimiter and formatting issues
    """
    try:
        # Read the raw content first
        with open(input_file, 'r', encoding='utf-8') as file:
            content = file.read()
        
        # Split into lines
        lines = content.strip().split('\n')
        
        # Expected headers based on the first line
        headers = ['conv_id', 'utterance_idx', 'context', 'prompt', 'speaker_idx', 'utterance', 'selfeval', 'tags']
        
        fixed_rows = []
        fixed_rows.append(headers)  # Add headers as first row
        
        # Process each data line (skip header line)
        for line_num, line in enumerate(lines[1:], 2):  # Start from line 2
            if not line.strip():  # Skip empty lines
                continue
                
            try:
                # Do not replace _comma_ with actual commas
                # Parse the line directly
                csv_reader = csv.reader(StringIO(line))
                row = next(csv_reader)
                
                # If we don't have enough columns, pad with empty strings
                while len(row) < len(headers):
                    row.append('')
                
                # If we have too many columns, merge the extra ones into the last column
                if len(row) > len(headers):
                    # Merge extra columns into the tags column (last column)
                    extra_content = ','.join(row[len(headers)-1:])
                    row = row[:len(headers)-1] + [extra_content]
                
                fixed_rows.append(row)
                
            except Exception as e:
                print(f"Warning: Issue processing line {line_num}: {e}")
                print(f"Problematic line: {line}")
                
                # Try manual parsing as fallback
                try:
                    # Simple split and clean up
                    parts = line.split(',')
                    
                    # Try to reconstruct based on expected pattern
                    if len(parts) >= 6:
                        # Basic reconstruction
                        conv_id = parts[0] if parts[0] else ''
                        utterance_idx = parts[1] if len(parts) > 1 else ''
                        context = parts[2] if len(parts) > 2 else ''
                        
                        # The prompt and utterance might contain commas, so we need to be careful
                        # We'll join parts and then try to split them properly
                        remaining = ','.join(parts[3:])
                        
                        # Try to extract the last few fields which have more predictable patterns
                        # selfeval typically looks like "5|5|5_2|2|5" 
                        # speaker_idx is usually a single digit
                        
                        # This is a simplified approach - join everything as utterance for now
                        prompt = ''
                        speaker_idx = ''
                        utterance = remaining
                        selfeval = ''
                        tags = ''
                        
                        # Look for selfeval pattern at the end
                        selfeval_pattern = r'(\d+\|\d+\|\d+_\d+\|\d+\|\d+)'
                        match = re.search(selfeval_pattern, remaining)
                        if match:
                            selfeval = match.group(1)
                            # Remove selfeval from utterance
                            utterance = remaining.replace(selfeval, '').rstrip(',')
                        
                        row = [conv_id, utterance_idx, context, prompt, speaker_idx, utterance, selfeval, tags]
                        fixed_rows.append(row)
                        
                except Exception as e2:
                    print(f"Could not fix line {line_num}, skipping: {e2}")
                    continue
        
        # Write the fixed CSV
        with open(output_file, 'w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file, quoting=csv.QUOTE_ALL)
            writer.writerows(fixed_rows)
        
        print(f"Successfully processed {len(fixed_rows)-1} rows")
        print(f"Fixed CSV saved to: {output_file}")
        
        # Display some statistics
        print(f"\nFile Statistics:")
        print(f"- Total rows (including header): {len(fixed_rows)}")
        print(f"- Data rows: {len(fixed_rows)-1}")
        print(f"- Columns: {len(headers)}")
        
        return True
        
    except FileNotFoundError:
        print(f"Error: Input file '{input_file}' not found.")
        return False
    except Exception as e:
        print(f"An error occurred: {e}")
        return False

def validate_csv(file_path):
    """
    Validate the fixed CSV file
    """
    try:
        df = pd.read_csv(file_path)
        print(f"\nValidation Results:")
        print(f"- Successfully loaded CSV with {len(df)} rows and {len(df.columns)} columns")
        print(f"- Columns: {list(df.columns)}")
        print(f"- No obvious parsing errors detected")
        
        # Show first few rows
        print(f"\nFirst 3 rows preview:")
        print(df.head(3).to_string())
        
        return True
    except Exception as e:
        print(f"Validation failed: {e}")
        return False

# Main execution
if __name__ == "__main__":
    input_file = "empatheticdialogues/train.csv"  # Your input file
    output_file = "dataset.csv"  # Output file
    
    print("Starting CSV fix process...")
    
    if fix_csv_file(input_file, output_file):
        print("\n" + "="*50)
        print("Validating the fixed file...")
        validate_csv(output_file)
        print("\nProcess completed successfully!")
    else:
        print("Process failed!")