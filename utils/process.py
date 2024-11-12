from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
from collections import defaultdict
import os

def remove_duplicate_sequences(file_path, output_path):
    # Dictionary to store unique sequences and their first occurrence
    sequence_map = {}

    # List to store unique SeqRecord objects to write to the output file
    unique_records = []

    # Read file and store the first occurrence of each sequence
    for record in SeqIO.parse(file_path, "fasta"):
        sequence = str(record.seq)
        if sequence not in sequence_map:
            sequence_map[sequence] = record.id  # Store the ID of the first occurrence
            unique_records.append(record)  # Add the first occurrence to the output list

    # Write only unique records to the output file
    with open(output_path, "w") as output_file:
        SeqIO.write(unique_records, output_file, "fasta")

    print(f"Total unique sequences: {len(unique_records)}")
    print(f"File with duplicates removed has been saved as {output_path}")

def rename_files_sequentially(folder_path, start=0):
    for i in range(36):
        file_path = os.path.join(folder_path, str(i))
        if os.path.isfile(file_path):
            os.remove(file_path)
            print(f"Deleted file: {file_path}")

    remaining_files = sorted(os.listdir(folder_path), key=lambda x: int(x))
    
    for new_index, file_name in enumerate(remaining_files):
        old_file_path = os.path.join(folder_path, file_name)
        new_file_path = os.path.join(folder_path, str(new_index))
        os.rename(old_file_path, new_file_path)
        print(f"Renamed {old_file_path} to {new_file_path}")

    print("Renaming complete.")

# Call function with input and output file paths
# remove_duplicate_sequences(
#     "D:/Major/AIProject/ATG/data/fastadata/F_Negative_training.fasta", 
#     "D:/Major/AIProject/ATG/data/fastadata/Negative_training.fasta"
# )

# folder_path = "D:/Major/AIProject/ATG/data/pssmdata/Negative_training/pssm_profile_uniref50"
# rename_files_sequentially(folder_path)