from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
from collections import defaultdict

def check_duplicate_sequences(file_path):
    # Dictionary to store sequences and their corresponding indices and IDs
    sequence_map = defaultdict(list)

    # Read file and store sequences with sequential indices and IDs
    with open(file_path) as file:
        for index, record in enumerate(SeqIO.parse(file, "fasta")):
            sequence = str(record.seq)
            sequence_map[sequence].append((index, record.id))

    # Find duplicates based on sequence content only
    duplicates = {seq: data for seq, data in sequence_map.items() if len(data) > 1}

    # Output results
    if duplicates:
        print(f"Found {len(duplicates)} duplicate sequences:")
        for seq, entries in duplicates.items():
            indices = [entry[0] for entry in entries]
            ids = [entry[1] for entry in entries]
            print(f"Sequence found at indices {indices} with IDs {ids}")
    else:
        print("No duplicate sequences found.")

    # Total unique sequences and records
    print(f"Total unique sequences: {len(sequence_map)}")
    print(f"Total records in file: {sum(len(v) for v in sequence_map.values())}")

# Call function with the path to your FASTA file
check_duplicate_sequences("D:/Major/AIProject/ATG/data/fastadata/Negative_training.fasta")

# (protein) D:\Major\AIProject\ATG>D:/miniconda3/envs/protein/python.exe d:/Major/AIProject/ATG/utils/check.py
# Found 36 duplicate sequences:
# Sequence found at indices [0, 36] with IDs ['tr|D3BTL7|D3BTL7_POLPA', 'tr|D3BTL7|D3BTL7_POLPA']
# Sequence found at indices [1, 37] with IDs ['tr|D3BRE2|D3BRE2_POLPA', 'tr|D3BRE2|D3BRE2_POLPA']
# Sequence found at indices [2, 38] with IDs ['tr|B4KV30|B4KV30_DROMO', 'tr|B4KV30|B4KV30_DROMO']
# Sequence found at indices [3, 39] with IDs ['tr|C0YUP5|C0YUP5_9FLAO', 'tr|C0YUP5|C0YUP5_9FLAO']
# Sequence found at indices [4, 40] with IDs ['tr|C0QKA9|C0QKA9_DESAH', 'tr|C0QKA9|C0QKA9_DESAH']
# Sequence found at indices [5, 41] with IDs ['tr|A4X9A3|A4X9A3_SALTO', 'tr|A4X9A3|A4X9A3_SALTO']
# Sequence found at indices [6, 42] with IDs ['tr|Q01BC6|Q01BC6_OSTTA', 'tr|Q01BC6|Q01BC6_OSTTA']
# Sequence found at indices [7, 43] with IDs ['tr|B4QPW6|B4QPW6_DROSI', 'tr|B4QPW6|B4QPW6_DROSI']
# Sequence found at indices [8, 44] with IDs ['tr|C3KAP5|C3KAP5_PSEFS', 'tr|C3KAP5|C3KAP5_PSEFS']
# Sequence found at indices [9, 45] with IDs ['tr|Q1DA49|Q1DA49_MYXXD', 'tr|Q1DA49|Q1DA49_MYXXD']
# Sequence found at indices [10, 46] with IDs ['tr|Q2QJT5|Q2QJT5_SOLLC', 'tr|Q2QJT5|Q2QJT5_SOLLC']
# Sequence found at indices [11, 47] with IDs ['tr|A1U009|A1U009_MARAV', 'tr|A1U009|A1U009_MARAV']
# Sequence found at indices [12, 48] with IDs ['tr|A8Y4F1|A8Y4F1_CAEBR', 'tr|A8Y4F1|A8Y4F1_CAEBR']
# Sequence found at indices [13, 49] with IDs ['tr|Q0KAS9|Q0KAS9_RALEH', 'tr|Q0KAS9|Q0KAS9_RALEH']
# Sequence found at indices [14, 50] with IDs ['tr|A8I9C7|A8I9C7_CHLRE', 'tr|A8I9C7|A8I9C7_CHLRE']
# Sequence found at indices [15, 51] with IDs ['tr|Q2H4S4|Q2H4S4_CHAGB', 'tr|Q2H4S4|Q2H4S4_CHAGB']
# Sequence found at indices [16, 52] with IDs ['tr|A5KA71|A5KA71_PLAVI', 'tr|A5KA71|A5KA71_PLAVI']
# Sequence found at indices [17, 53] with IDs ['tr|Q9XXH5|Q9XXH5_CAEEL', 'tr|Q9XXH5|Q9XXH5_CAEEL']
# Sequence found at indices [18, 54] with IDs ['tr|A8NCV9|A8NCV9_COPC7', 'tr|A8NCV9|A8NCV9_COPC7']
# Sequence found at indices [19, 55] with IDs ['tr|A4J2C9|A4J2C9_DESRM', 'tr|A4J2C9|A4J2C9_DESRM']
# Sequence found at indices [20, 56] with IDs ['tr|C1N5Z8|C1N5Z8_9CHLO', 'tr|C1N5Z8|C1N5Z8_9CHLO']
# Sequence found at indices [21, 57] with IDs ['tr|Q4PCR5|Q4PCR5_USTMA', 'tr|Q4PCR5|Q4PCR5_USTMA']
# Sequence found at indices [22, 58] with IDs ['tr|Q9QKE6|Q9QKE6_9VIRU', 'tr|Q9QKE6|Q9QKE6_9VIRU']
# Sequence found at indices [23, 59] with IDs ['tr|Q4RVP5|Q4RVP5_TETNG', 'tr|Q4RVP5|Q4RVP5_TETNG']
# Sequence found at indices [24, 60] with IDs ['tr|A8XL03|A8XL03_CAEBR', 'tr|A8XL03|A8XL03_CAEBR']
# Sequence found at indices [25, 61] with IDs ['tr|D3BQP8|D3BQP8_POLPA', 'tr|D3BQP8|D3BQP8_POLPA']
# Sequence found at indices [26, 62] with IDs ['tr|D0NMK0|D0NMK0_PHYIN', 'tr|D0NMK0|D0NMK0_PHYIN']
# Sequence found at indices [27, 63] with IDs ['tr|C7MMK1|C7MMK1_CRYCD', 'tr|C7MMK1|C7MMK1_CRYCD']
# Sequence found at indices [28, 64] with IDs ['tr|C5NBI8|C5NBI8_BURMA', 'tr|C5NBI8|C5NBI8_BURMA']
# Sequence found at indices [29, 65] with IDs ['tr|Q6JGV6|Q6JGV6_9CLOS', 'tr|Q6JGV6|Q6JGV6_9CLOS']
# Sequence found at indices [30, 66] with IDs ['tr|D4GCV4|D4GCV4_PANAM', 'tr|D4GCV4|D4GCV4_PANAM']
# Sequence found at indices [31, 67] with IDs ['tr|Q8J1W4|Q8J1W4_CLAPU', 'tr|Q8J1W4|Q8J1W4_CLAPU']
# Sequence found at indices [32, 68] with IDs ['tr|D2VJV1|D2VJV1_NAEGR', 'tr|D2VJV1|D2VJV1_NAEGR']
# Sequence found at indices [33, 69] with IDs ['tr|Q1D888|Q1D888_MYXXD', 'tr|Q1D888|Q1D888_MYXXD']
# Sequence found at indices [34, 70] with IDs ['tr|Q7VH02|Q7VH02_HELHP', 'tr|Q7VH02|Q7VH02_HELHP']
# Sequence found at indices [35, 71] with IDs ['tr|B3ERL3|B3ERL3_AMOA5', 'tr|B3ERL3|B3ERL3_AMOA5']
# Total unique sequences: 357
# Total records in file: 393