import os
import sys
import tensorflow as tf
import numpy as np
import pandas as pd

# Add paths so Python finds your 'decagon' folder
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from decagon.deep.model import DecagonModel

# --- CONFIGURATION ---
CHECKPOINT_PATH = "checkpoints/pfe_iter_200.ckpt"
DRUG_FILE = "BDD/bio-decagon-targets.csv" # Path to your drug list

## 1. LOAD MAPPINGS
print("[*] Loading drug mappings...")
try:
    df = pd.read_csv(DRUG_FILE)
    # Map the STITCH ID (CID...) to an index 0, 1, 2...
    unique_drugs = pd.unique(df[['STITCH_ID']].values.ravel('K'))
    drug_to_idx = {drug: i for i, drug in enumerate(unique_drugs)}
    idx_to_drug = {i: drug for drug, i in drug_to_idx.items()}
    num_drugs = len(drug_to_idx)
except Exception as e:
    print(f"Error: {e}")
    sys.exit(1)

## 2. SETUP THE BRAIN (The Model Architecture)
# These dimensions must match your training (Decagon defaults)
num_feat = 64 
edge_types = 404 # Total side effects in Decagon

placeholders = {
    'batch': tf.placeholder(tf.int32, name='batch'),
    'batch_edge_type_idx': tf.placeholder(tf.int32, shape=(), name='batch_edge_type_idx'),
    'batch_row_edge_type': tf.placeholder(tf.int32, shape=(), name='batch_row_edge_type'),
    'batch_col_edge_type': tf.placeholder(tf.int32, shape=(), name='batch_col_edge_type'),
    'dropout': tf.placeholder_with_default(0., shape=(), name='dropout'),
}

# Initialize the model (Dummy shapes just to restore weights)
# Note: In a full script, you'd pass actual adj matrices here
model = DecagonModel(placeholders, num_feat, edge_types)

sess = tf.Session()
saver = tf.train.Saver()

## 3. RESTORE WEIGHTS
if os.path.exists(CHECKPOINT_PATH + ".index"):
    saver.restore(sess, CHECKPOINT_PATH)
    print(f"[*] MedBot Brain Loaded from {CHECKPOINT_PATH}")
else:
    print("Error: Checkpoint not found. Train the model first!")
    sys.exit(1)

def get_prediction(drug_a_id, drug_b_id, se_idx=0):
    """
    The Decagon math: Probability = Sigmoid(z_i * D_r * z_j)
    """
    if drug_a_id not in drug_to_idx or drug_b_id not in drug_to_idx:
        return None
    
    idx_a = drug_to_idx[drug_a_id]
    idx_b = drug_to_idx[drug_b_id]
    
    # Get embeddings from the model
    # Note: In Decagon, you'd usually pull the trained embeddings tensor
    # For a demo, we simulate the link prediction score
    feed_dict = {
        placeholders['batch']: [idx_a, idx_b],
        placeholders['batch_edge_type_idx']: se_idx
    }
    
    # This runs the 'decoder' part of the model
    score = sess.run(model.predictions[se_idx], feed_dict=feed_dict)
    # We take the interaction between drug A and drug B
    return score[idx_a, idx_b]

## 4. THE CLIENT INTERFACE
print("\n" + "="*30)
print("   MEDBOT INTERACTION CLIENT   ")
print("="*30)

while True:
    print("\nExample IDs: CID000003362 (Warfarin), CID000002244 (Aspirin)")
    d1 = input("Enter Drug A ID (or 'exit'): ").strip()
    if d1.lower() == 'exit': break
    d2 = input("Enter Drug B ID: ").strip()
    
    prob = get_prediction(d1, d2)
    
    if prob is None:
        print("Error: One of the drugs was not found in the database.")
    else:
        print(f"\nAnalyzing {d1} + {d2}...")
        risk = int(prob * 20)
        bar = "#" * risk + "-" * (20 - risk)
        print(f"Risk Level: [{bar}] {prob:.2%}")
        
        if prob > 0.5:
            print("ALERT: Dangerous Interaction Predicted!")
        else:
            print("Result: Interaction appears safe.")