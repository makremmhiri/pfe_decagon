import os
import sys
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior() # This tells TF2 to act exactly like TF1
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
    unique_drugs = pd.unique(df[['STITCH']].values.ravel('K'))
    drug_to_idx = {drug: i for i, drug in enumerate(unique_drugs)}
    idx_to_drug = {i: drug for drug, i in drug_to_idx.items()}
    num_drugs = len(drug_to_idx)
except Exception as e:
    print(f"Error: {e}")
    sys.exit(1)

## 2. SETUP THE BRAIN (The Model Architecture)
num_feat = 64 

# Decagon expects a dictionary of edge types. 
# Node type 0 = Protein, Node type 1 = Drug. 
# (1, 1) represents the Drug-Drug interactions (your 100 side effects!)
edge_types_dict = {
    (0, 0): 2,   # 2 types of Protein-Protein edges
    (0, 1): 1,   # 1 type of Protein-Drug edge
    (1, 0): 1,   # 1 type of Drug-Protein edge
    (1, 1): 100  # 100 types of Drug-Drug edges (Side Effects)
}

# Decagon needs to know which math to use for which relationships
decoders = {
    (0, 0): 'innerproduct',
    (0, 1): 'innerproduct',
    (1, 0): 'innerproduct',
    (1, 1): 'dedicom'       # 'dedicom' is the specific tensor math for side effects
}

# Dummy non-zero features (needed for model initialization)
nonzero_feat = {0: num_feat, 1: num_feat}

placeholders = {
    # Added the missing feature placeholders
    'feat_0': tf.sparse_placeholder(tf.float32, shape=(None, num_feat), name='feat_0'),
    'feat_1': tf.sparse_placeholder(tf.float32, shape=(None, num_feat), name='feat_1'),
    
    # Kept the original placeholders
    'batch': tf.placeholder(tf.int32, name='batch'),
    'batch_edge_type_idx': tf.placeholder(tf.int32, shape=(), name='batch_edge_type_idx'),
    'batch_row_edge_type': tf.placeholder(tf.int32, shape=(), name='batch_row_edge_type'),
    'batch_col_edge_type': tf.placeholder(tf.int32, shape=(), name='batch_col_edge_type'),
    'degrees': tf.placeholder(tf.int32, name='degrees'),
    'dropout': tf.placeholder_with_default(0., shape=(), name='dropout'),
}

# Initialize the model perfectly
model = DecagonModel(
    placeholders=placeholders, 
    num_feat=num_feat, 
    nonzero_feat=nonzero_feat, 
    edge_types=edge_types_dict, 
    decoders=decoders
)

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
    
    # FIX: We must pass a 2D array [[A, B]] and explicitly state they are Drugs (Type 1)
    feed_dict = {
        placeholders['batch']: [[idx_a, idx_b]], 
        placeholders['batch_edge_type_idx']: se_idx,
        placeholders['batch_row_edge_type']: 1,
        placeholders['batch_col_edge_type']: 1,
        placeholders['dropout']: 0.0
    }
    
    # Run the model
    score = sess.run(model.predictions[se_idx], feed_dict=feed_dict)
    
    # FIX: Convert the raw output 'logit' into a probability (0.0 to 1.0)
    prob = 1 / (1 + np.exp(-score[0]))
    return prob

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