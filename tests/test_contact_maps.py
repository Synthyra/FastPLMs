import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
import os
import random
import requests
import tempfile
import sys
from Bio.PDB import PDBParser, PPBuilder
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from modeling_fastesm import FastEsmModel


def download_random_pdb():
    """
    Download a random protein chain PDB file.
    
    Returns:
        str: Path to the downloaded PDB file.
    """
    example_pdbs = ["1AKE"]
    
    # Select a random PDB ID
    pdb_id = random.choice(example_pdbs)
    print(f"Selected random PDB ID: {pdb_id}")
    
    # Create a temporary file to store the PDB
    temp_file = tempfile.NamedTemporaryFile(suffix=".pdb", delete=False)
    temp_file_path = temp_file.name
    temp_file.close()
    
    # Download the PDB file
    url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
    response = requests.get(url)
    
    if response.status_code == 200:
        with open(temp_file_path, 'wb') as f:
            f.write(response.content)
        print(f"Downloaded PDB file to: {temp_file_path}")
        return temp_file_path
    else:
        raise Exception(f"Failed to download PDB file: {response.status_code}")


def parse_pdb(pdb_file):
    """
    Parse a PDB file and extract the protein sequence and CA atom coordinates.
    
    Parameters:
        pdb_file (str): Path to the PDB file.
    
    Returns:
        tuple: (sequence (str), coords (np.ndarray of shape (L, 3)))
    """
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("protein", pdb_file)
    ppb = PPBuilder()
    
    # Assume a single protein chain; take the first polypeptide found.
    for pp in ppb.build_peptides(structure):
        sequence = str(pp.get_sequence())
        coords = []
        for residue in pp:
            # Only add the CA atom if available.
            if 'CA' in residue:
                coords.append(residue['CA'].get_coord())
        if len(coords) == 0:
            raise ValueError("No CA atoms found in the polypeptide.")
        return sequence, np.array(coords)
    
    raise ValueError("No polypeptide chains were found in the PDB file.")


def compute_distance_matrix(coords):
    """
    Compute the pairwise Euclidean distance matrix from a set of coordinates.
    
    Parameters:
        coords (np.ndarray): Array of shape (L, 3) where L is the number of residues.
    
    Returns:
        np.ndarray: A matrix of shape (L, L) containing distances.
    """
    diff = coords[:, None, :] - coords[None, :, :]
    dist_matrix = np.sqrt(np.sum(diff**2, axis=-1))

    return dist_matrix


def get_esm_contact_map(sequence):
    """
    Use the ESM model to predict a contact map for the given protein sequence.
    
    Parameters:
        sequence (str): Amino acid sequence.
    
    Returns:
        np.ndarray: A 2D array (L x L) with contact probabilities.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = "Synthyra/ESM2-650M"
    model = FastEsmModel.from_pretrained(model_path).eval().to(device)
    tokenizer = model.tokenizer
    
    inputs = tokenizer(sequence, return_tensors="pt")
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.no_grad():
        contact_map = model.predict_contacts(inputs["input_ids"], inputs["attention_mask"])
        print(contact_map.shape)
        contact_map = contact_map.squeeze().cpu().numpy()
        print(contact_map.shape)
    return contact_map


def plot_maps(true_contact_map, predicted_contact_map, pdb_file):
    """
    Generate two subplots:
      1. ESM predicted contact map.
      2. True contact map from the PDB (binary, thresholded).
    
    Parameters:
        true_contact_map (np.ndarray): Binary (0/1) contact map from PDB.
        predicted_contact_map (np.ndarray): Predicted contact probabilities from ESM.
        pdb_file (str): Path to the PDB file, used to generate output filename.
    """
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    
    # Plot the ESM-predicted contact map.
    im0 = axs[0].imshow(predicted_contact_map, cmap='RdYlBu_r', aspect='equal')
    axs[0].set_title("Predicted contact probabilities")
    axs[0].set_xlabel("Residue index")
    axs[0].set_ylabel("Residue index")
    fig.colorbar(im0, ax=axs[0], fraction=0.046, pad=0.04)
    
    # Plot the true contact map (binary contacts).
    im1 = axs[1].imshow(true_contact_map, cmap='RdYlBu_r', aspect='equal')
    axs[1].set_title("True contacts (PDB, threshold = 8 Å)")
    axs[1].set_xlabel("Residue index")
    axs[1].set_ylabel("Residue index")
    fig.colorbar(im1, ax=axs[1], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    
    # Generate output filename from PDB filename
    pdb_name = os.path.splitext(os.path.basename(pdb_file))[0]
    output_file = f"contact_maps_{pdb_name}.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()


def main():
    # py tests/test_contact_maps.py
    parser = argparse.ArgumentParser(
        description="Extract protein sequence and compute contact maps from a PDB file using ESM predictions."
    )
    parser.add_argument("--pdb_file", type=str, help="Path to the PDB file of the protein. If not provided, a random PDB will be downloaded.", default=None)
    parser.add_argument(
        "--threshold",
        type=float,
        default=8.0,
        help="Distance threshold (in Å) for defining true contacts (default: 8.0 Å)."
    )
    args = parser.parse_args()
    
    # If no PDB file is provided, download a random one
    if args.pdb_file is None:
        pdb_file = download_random_pdb()
    else:
        pdb_file = args.pdb_file
    
    try:
        # Parse the PDB file.
        sequence, coords = parse_pdb(pdb_file)
        print("Extracted Protein Sequence:")
        print(sequence)
        
        # Compute the pairwise distance matrix.
        dist_matrix = compute_distance_matrix(coords)
        
        # Create a binary contact map from the distance matrix using the threshold.
        true_contact_map = (dist_matrix < args.threshold).astype(float)

        # Get the predicted contact map from the ESM model.
        predicted_contact_map = get_esm_contact_map(sequence)
        
        # Check that the dimensions agree.
        if predicted_contact_map.shape[0] != true_contact_map.shape[0]:
            print("Warning: The predicted contact map and true contact map have different dimensions.")
        
        # Plot the maps.
        plot_maps(true_contact_map, predicted_contact_map, pdb_file)
        
        print(f"Contact maps saved to: contact_maps_{os.path.splitext(os.path.basename(pdb_file))[0]}.png")
    
    finally:
        # Clean up the temporary file if we downloaded a random PDB
        if args.pdb_file is None and os.path.exists(pdb_file):
            os.remove(pdb_file)
            print(f"Removed temporary PDB file: {pdb_file}")


if __name__ == '__main__':
    main()



