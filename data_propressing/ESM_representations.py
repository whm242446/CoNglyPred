import torch
import esm
import numpy as np
import pandas as pd
from Bio import SeqIO

from tqdm import tqdm

# Using GPU devices
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
name = "/protein_complete.fasta"

# Loading pre-trained models
model, alphabet = esm.pretrained.esm2_t6_8M_UR50D()
model = model.to(device)
batch_converter = alphabet.get_batch_converter()
model.eval()

# Record samples of sequences with errors
error_placeholders = []
protein_data = {}
for seq_record in tqdm(SeqIO.parse(name, "fasta"), desc="Processing Sequences"):
    try:
        placeholder = seq_record.id
        seq = str(seq_record.seq)
        placeholder = placeholder.split("|")[1]

        data = [
            ('placeholder', seq),
        ]

        batch_labels, batch_strs, batch_tokens = batch_converter(data)
        batch_tokens = batch_tokens.to(device)

        with torch.no_grad():
            results = model(batch_tokens, repr_layers=[33], return_contacts=True)
            
        # Taking the representation of the last layer as a protein characterisation
        token_representations = results["representations"][33].cpu().numpy()
        features = token_representations[0][1:-1]

        protein_data[placeholder] = {
            'seq': seq,
            'esm_features': features
        }

    except Exception as e:
        error_placeholders.append(placeholder)
    finally:
        if 'batch_tokens' in locals(): 
            del batch_tokens
        if 'results' in locals():
            del results
        if 'token_representations' in locals():
            del token_representations
        torch.cuda.empty_cache() # Release GPU memory
