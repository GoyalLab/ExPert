from Bio import SeqIO
import gzip
from tqdm import tqdm
from typing import List
import numpy as np
import scipy.sparse as sp

NUC_MAP = {'-': 0, 'N': 1, 'A': 2, 'C': 3, 'G': 4, 'T': 5}


def read_alignment(align_p: str) -> tuple[List[str], List[str]]:
    headers = []
    seqs = []
    with open(align_p, 'r') as f:
        s = ''
        header = None
        for line in tqdm(f, desc='Reading fasta', unit='lines'):
            if line.startswith('>'):
                h_idx = line[1:].split()[0]
                headers.append(h_idx)
                if header is None:
                    header = h_idx
                else:
                    header = h_idx
                    seqs.append(s)
                    s = ''
            else:
                s += line.rstrip()
        seqs.append(s)
    return headers, seqs


def read_fasta(fasta_p: str, id_key: str = 'gene_symbol', sep: str = ':') -> dict:
    fasta_d = {}
    with gzip.open(fasta_p, 'rt') as handle:
        for record in tqdm(SeqIO.parse(handle, 'fasta'), desc='Reading fasta', unit='entries'):
            gs = [desc.split(sep)[1] for desc in record.description.split() if desc.startswith(id_key)]
            if len(gs) != 0:
                gn = gs[0]
                if gn not in fasta_d.keys() or len(fasta_d[gn].seq) < len(record.seq):
                    record.id = gs[0]
            fasta_d[record.id] = record
    return fasta_d

def subset_fasta(fasta: dict, targets: List[str]) -> dict:
    return {k:v for k,v in fasta.items() if k in targets}

def fasta_to_matrix(sequences: List[str], max_length: int | None = None, as_sparse: bool = True):
    # Find max length if not specified
    if max_length is None:
        max_length = max(len(seq) for seq in sequences)

    # Define base-to-index mapping as a translation table
    base_to_int = str.maketrans("-NACGT", "012345")

    # Create matrix with numeric values for each nucleotide
    vs = []
    for i, seq in enumerate(sequences):
        # Translate and convert to integer array
        encoded = np.fromiter(str(seq).upper().translate(base_to_int), dtype=np.uint8)
        vs.append(encoded)
    padded = np.array([
        np.pad(seq, (0, max_length - len(seq)), constant_values=0)
        for seq in vs
    ])
    if as_sparse:
        padded = sp.csr_matrix(padded)
    return padded

# Encode DNA sequences as one-hot vectors
def one_hot_encode(sequences: List[str], max_length: int | None = None):
    # DNA nucleotide mapping: A, C, G, T
    # N is for unknown nucleotide
    nucleotide_map = NUC_MAP
    
    # Find max length if not specified
    if max_length is None:
        max_length = max(len(seq) for seq in sequences)
    
    # Create one-hot encoding
    encoded = np.zeros((len(sequences), max_length, len(nucleotide_map)), dtype=np.float32)
    for i, seq in enumerate(sequences):
        seq = seq.upper()
        for j, nuc in enumerate(seq[:max_length]):
            if nuc in nucleotide_map:
                encoded[i, j, nucleotide_map[nuc]] = 1.0
            else:
                # Unknown nucleotide
                encoded[i, j, 0] = 1.0
                
    return encoded
