from __future__ import division
from __future__ import print_function
from operator import itemgetter
from itertools import combinations
from collections import defaultdict
import time
import os
import sys
import re

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
import networkx as nx
import scipy.sparse as sp
from sklearn import metrics

from decagon.deep.optimizer import DecagonOptimizer
from decagon.deep.model import DecagonModel
from decagon.deep.minibatch import EdgeMinibatchIterator
from decagon.utility import rank_metrics, preprocessing
from polypharmacy import utility

# Train on CPU (hide GPU) due to memory constraints
# os.environ['CUDA_VISIBLE_DEVICES'] = ""

# Train on GPU
os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.allow_soft_placement = True

os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(0)

###########################################################
#
# Functions
#
###########################################################


def get_accuracy_scores(edges_pos, edges_neg, edge_type):
    feed_dict.update({placeholders['dropout']: 0})
    feed_dict.update({placeholders['batch_edge_type_idx']: int(minibatch.edge_type2idx[edge_type])})
    feed_dict.update({placeholders['batch_row_edge_type']: int(edge_type[0])})
    feed_dict.update({placeholders['batch_col_edge_type']: int(edge_type[1])})
    rec = sess.run(opt.predictions, feed_dict=feed_dict)

    def sigmoid(x):
        return 1. / (1 + np.exp(-x))

    # Predict on test set of edges
    preds = []
    actual = []
    predicted = []
    edge_ind = 0
    for u, v in edges_pos[edge_type[:2]][edge_type[2]]:
        score = sigmoid(rec[u, v])
        preds.append(score)
        assert adj_mats_orig[edge_type[:2]][edge_type[2]][u,v] == 1, 'Problem 1'

        actual.append(edge_ind)
        predicted.append((score, edge_ind))
        edge_ind += 1

    preds_neg = []
    for u, v in edges_neg[edge_type[:2]][edge_type[2]]:
        score = sigmoid(rec[u, v])
        preds_neg.append(score)
        assert adj_mats_orig[edge_type[:2]][edge_type[2]][u,v] == 0, 'Problem 0'

        predicted.append((score, edge_ind))
        edge_ind += 1

    preds_all = np.hstack([preds, preds_neg])
    preds_all = np.nan_to_num(preds_all)
    labels_all = np.hstack([np.ones(len(preds)), np.zeros(len(preds_neg))])
    # ADD THIS SAFETY CHECK: If there is no test data for this edge, skip it.
    if len(predicted) == 0:
        return 0.0, 0.0, 0.0  # Return zero scores so it doesn't crash
        
    predicted = list(zip(*sorted(predicted, reverse=True, key=itemgetter(0))))[1]

    roc_sc = metrics.roc_auc_score(labels_all, preds_all)
    aupr_sc = metrics.average_precision_score(labels_all, preds_all)
    apk_sc = rank_metrics.apk(actual, predicted, k=50)

    return roc_sc, aupr_sc, apk_sc


def construct_placeholders(edge_types):
    placeholders = {
        'batch': tf.placeholder(tf.int32, name='batch'),
        'batch_edge_type_idx': tf.placeholder(tf.int32, shape=(), name='batch_edge_type_idx'),
        'batch_row_edge_type': tf.placeholder(tf.int32, shape=(), name='batch_row_edge_type'),
        'batch_col_edge_type': tf.placeholder(tf.int32, shape=(), name='batch_col_edge_type'),
        'degrees': tf.placeholder(tf.int32),
        'dropout': tf.placeholder_with_default(0., shape=()),
    }
    placeholders.update({
        'adj_mats_%d,%d,%d' % (i, j, k): tf.sparse_placeholder(tf.float32)
        for i, j in edge_types for k in range(edge_types[i,j])})
    placeholders.update({
        'feat_%d' % i: tf.sparse_placeholder(tf.float32)
        for i, _ in edge_types})
    return placeholders

###########################################################
#
# Load and preprocess data (This is a dummy toy example!)
#
###########################################################

val_test_size = 0.2

# Define paths for BDD files
ppi_path = os.path.join('BDD', 'bio-decagon-ppi.csv')
drug_drug_path = os.path.join('BDD', 'bio-decagon-combo.csv')
drug_target_path = os.path.join('BDD', 'bio-decagon-targets.csv')

# 1. Load Gene Network (PPI)
gene_net, gene_node2idx = utility.load_ppi(ppi_path)
n_genes = len(gene_node2idx)
gene_adj = nx.adjacency_matrix(gene_net)
gene_degrees = np.array(gene_adj.sum(axis=0)).squeeze()

# 2. Load Drug-Drug Interactions and Side Effects
combo2stitch, combo2se, se2name = utility.load_combo_se(drug_drug_path)

# 3. Load Drug-Target Interactions
stitch2proteins = utility.load_targets(drug_target_path)

# Map drugs to indices
drugs = set()
for combo in combo2stitch:
    drugs.add(combo2stitch[combo][0])
    drugs.add(combo2stitch[combo][1])
for drug in stitch2proteins:
    drugs.add(drug)

drug2idx = {drug: i for i, drug in enumerate(sorted(list(drugs)))}
n_drugs = len(drug2idx)

# Construct Gene-Drug Adjacency Matrix
gene_drug_adj = sp.dok_matrix((n_genes, n_drugs), dtype=float)
for drug, targets in stitch2proteins.items():
    if drug in drug2idx:
        d_idx = drug2idx[drug]
        for gene in targets:
            if gene in gene_node2idx:
                g_idx = gene_node2idx[gene]
                gene_drug_adj[g_idx, d_idx] = 1.

gene_drug_adj = gene_drug_adj.tocsr()
drug_gene_adj = gene_drug_adj.transpose(copy=True)

# Construct Drug-Drug Adjacency Matrices (one per Side Effect)
se_to_combos = defaultdict(list)
for combo, se_set in combo2se.items():
    d1, d2 = combo2stitch[combo]
    if d1 in drug2idx and d2 in drug2idx:
        idx1 = drug2idx[d1]
        idx2 = drug2idx[d2]
        for se in se_set:
            se_to_combos[se].append((idx1, idx2))

drug_drug_adj_list = []
sorted_se = sorted(list(se2name.keys()))
sorted_se = sorted_se[:200]
for se in sorted_se:
    mat = sp.dok_matrix((n_drugs, n_drugs), dtype=float)
    for idx1, idx2 in se_to_combos[se]:
        mat[idx1, idx2] = mat[idx2, idx1] = 1.
    drug_drug_adj_list.append(mat.tocsr())

drug_degrees_list = [np.array(drug_adj.sum(axis=0)).squeeze() for drug_adj in drug_drug_adj_list]


# data representation
adj_mats_orig = {
    (0, 0): [gene_adj, gene_adj.transpose(copy=True)],
    (0, 1): [gene_drug_adj],
    (1, 0): [drug_gene_adj],
    (1, 1): drug_drug_adj_list + [x.transpose(copy=True) for x in drug_drug_adj_list],
}
degrees = {
    0: [gene_degrees, gene_degrees],
    1: drug_degrees_list + drug_degrees_list,
}

# featureless (genes)
gene_feat = sp.identity(n_genes)
gene_nonzero_feat, gene_num_feat = gene_feat.shape
gene_feat = preprocessing.sparse_to_tuple(gene_feat.tocoo())

# features (drugs)
drug_feat = sp.identity(n_drugs)
drug_nonzero_feat, drug_num_feat = drug_feat.shape
drug_feat = preprocessing.sparse_to_tuple(drug_feat.tocoo())

# data representation
num_feat = {
    0: gene_num_feat,
    1: drug_num_feat,
}
nonzero_feat = {
    0: gene_nonzero_feat,
    1: drug_nonzero_feat,
}
feat = {
    0: gene_feat,
    1: drug_feat,
}

edge_type2dim = {k: [adj.shape for adj in adjs] for k, adjs in adj_mats_orig.items()}
edge_type2decoder = {
    (0, 0): 'bilinear',
    (0, 1): 'bilinear',
    (1, 0): 'bilinear',
    (1, 1): 'dedicom',
}

edge_types = {k: len(v) for k, v in adj_mats_orig.items()}
num_edge_types = sum(edge_types.values())
print("Edge types:", "%d" % num_edge_types)

###########################################################
#
# Settings and placeholders
#
###########################################################

flags = tf.flags
FLAGS = flags.FLAGS
flags.DEFINE_integer('neg_sample_size', 1, 'Negative sample size.')
flags.DEFINE_float('learning_rate', 0.001, 'Initial learning rate.')
flags.DEFINE_integer('epochs', 2, 'Number of epochs to train.')
flags.DEFINE_integer('hidden1', 64, 'Number of units in hidden layer 1.')
flags.DEFINE_integer('hidden2', 32, 'Number of units in hidden layer 2.')
flags.DEFINE_float('weight_decay', 0, 'Weight for L2 loss on embedding matrix.')
flags.DEFINE_float('dropout', 0.1, 'Dropout rate (1 - keep probability).')
flags.DEFINE_float('max_margin', 0.1, 'Max margin parameter in hinge loss')
#512
flags.DEFINE_integer('batch_size', 128, 'minibatch size.')
flags.DEFINE_boolean('bias', True, 'Bias term.')
# Important -- Do not evaluate/print validation performance every iteration as it can take
# substantial amount of time
PRINT_PROGRESS_EVERY = 50

print("Defining placeholders")
placeholders = construct_placeholders(edge_types)

###########################################################
#
# Create minibatch iterator, model and optimizer
#
###########################################################

print("Create minibatch iterator")
minibatch = EdgeMinibatchIterator(
    adj_mats=adj_mats_orig,
    feat=feat,
    edge_types=edge_types,
    batch_size=FLAGS.batch_size,
    val_test_size=val_test_size
)

print("Create model")
model = DecagonModel(
    placeholders=placeholders,
    num_feat=num_feat,
    nonzero_feat=nonzero_feat,
    edge_types=edge_types,
    decoders=edge_type2decoder,
)

print("Create optimizer")
with tf.name_scope('optimizer'):
    opt = DecagonOptimizer(
        embeddings=model.embeddings,
        latent_inters=model.latent_inters,
        latent_varies=model.latent_varies,
        degrees=degrees,
        edge_types=edge_types,
        edge_type2dim=edge_type2dim,
        placeholders=placeholders,
        batch_size=FLAGS.batch_size,
        margin=FLAGS.max_margin
    )

saver = tf.train.Saver(max_to_keep=None)

print("Initialize session")
sess = tf.Session(config=config)

# --- SMART LOADING LOGIC ---
# This looks for the most recent checkpoint in your folder
ckpt = tf.train.get_checkpoint_state("checkpoints/")

if ckpt and ckpt.model_checkpoint_path:
    print(f"Restoring model from: {ckpt.model_checkpoint_path}")
    # This loads the weights from your SSD into the GPU
    saver.restore(sess, ckpt.model_checkpoint_path)
else:
    print("No checkpoint found. Initializing new variables...")
    # This starts the model from zero (random weights)
    sess.run(tf.global_variables_initializer())

# ==============================================================
# INTERACTIVE LOOP STARTS HERE (Indented to the left/global level)
# ==============================================================

# =============== START MEDBOT CLIENT ===============

print("\n[*] Initializing MedBot Architecture...")

# 1. Restore the trained weights
CHECKPOINT_PATH = "checkpoints/pfe_iter_60000.ckpt" 

if os.path.exists(CHECKPOINT_PATH + ".index"):
    saver.restore(sess, CHECKPOINT_PATH)
    print(f"[*] MedBot Brain Successfully Loaded from {CHECKPOINT_PATH}")
else:
    print(f"[!] ERROR: Checkpoint not found at {CHECKPOINT_PATH}")
    sys.exit(1)

# ==============================================================
# THE CRITICAL FIX: Reset the data iterator so it isn't empty!
# ==============================================================
minibatch.shuffle() 

# 2. Interactive Loop
# 1. BUILD THE MASTER STRUCTURAL FEED DICT
# This manually feeds the raw adjacency matrices and features into the graph,
# bypassing the exhausted minibatch iterator completely.
master_feed_dict = {}

# Feed Adjacency Matrices
for i, j in edge_types:
    for k in range(edge_types[i,j]):
        # Convert dense matrices to sparse tuples required by TensorFlow
        sparse_mat = preprocessing.sparse_to_tuple(adj_mats_orig[(i, j)][k])
        master_feed_dict[placeholders['adj_mats_%d,%d,%d' % (i, j, k)]] = sparse_mat

# Feed Node Features
for i, _ in edge_types:
    master_feed_dict[placeholders['feat_%d' % i]] = feat[i]

# 2. Interactive Loop
print("\n" + "="*40)
print("       MEDBOT INTERACTION CLIENT       ")
print("="*40)

while True:
    print("\nExample IDs: CID000003362 (Warfarin), CID000002244 (Aspirin)")
    d1 = input("Enter Drug A ID (or 'exit'): ").strip()
    if d1.lower() == 'exit': break
    d2 = input("Enter Drug B ID: ").strip()
    
    if d1 not in drug2idx or d2 not in drug2idx:
        print("Error: One of the drugs was not found in the database.")
        continue
        
    idx_a = drug2idx[d1]
    idx_b = drug2idx[d2]
    
    print(f"\nScanning all side effects for {d1} + {d2}...")
    
    predicted_side_effects = []
    
    try:
        # Loop through all the side effects your model was trained on
        for se_idx in range(len(sorted_se)):
            
            feed_dict = master_feed_dict.copy()
            
            feed_dict.update({
                placeholders['batch']: np.array([[idx_a, idx_b]], dtype=np.int32), 
                placeholders['batch_edge_type_idx']: int(se_idx),
                placeholders['batch_row_edge_type']: 1,  
                placeholders['batch_col_edge_type']: 1,
                placeholders['dropout']: 0.0,
                placeholders['degrees']: np.zeros(1, dtype=np.int32)
            })
            
            score = sess.run(opt.predictions, feed_dict=feed_dict)
            raw_val = float(score[0][0]) if isinstance(score[0], np.ndarray) else float(score[0])
            prob = 1 / (1 + np.exp(-raw_val)) 
            
            # If the probability is greater than 50%, record it
            if prob > 0.50:
                se_id = sorted_se[se_idx]
                se_name = se2name[se_id]
                predicted_side_effects.append((se_name, prob))
        
        # Sort the side effects from highest risk to lowest risk
        predicted_side_effects.sort(key=lambda x: x[1], reverse=True)
        
        if len(predicted_side_effects) == 0:
            print("Result: Interaction appears safe. No high-risk side effects detected.")
        else:
            print(f"ALERT: {len(predicted_side_effects)} potential side effects predicted!")
            print("-" * 50)
            
            # Print the Top 10 most likely side effects
            for name, p in predicted_side_effects[:10]:
                risk_bar = "#" * int(p * 20) + "-" * (20 - int(p * 20))
                print(f"[{risk_bar}] {p:.2%} | {name}")
            print("-" * 50)
                
    except Exception as e:
        print(f"Prediction Error: {e}")

###########################################################
#
# Train model
#
###########################################################

print("Train model")


# --- 1. AUTOMATIC ITERATION DETECTION (Outside the loops) ---
start_itr = 0
if os.path.exists('checkpoints'):
    # This looks for any file with 'iter_' followed by numbers
    ckpt_files = [f for f in os.listdir('checkpoints') if 'iter_' in f and f.endswith('.index')]
    if ckpt_files:
        # Extracts the numbers and finds the highest one
        iters = [int(re.findall(r'iter_(\d+)', f)[0]) for f in ckpt_files]
        start_itr = max(iters)
        print(f"[*] Resuming from iteration: {start_itr}")
    else:
        print("[*] No checkpoints found, starting from 0.")

# --- 2. TRAINING LOOPS ---
# for epoch in range(FLAGS.epochs):
    
#     # IMPORTANT: Shuffle only if you are starting a fresh epoch
#     minibatch.shuffle()
    
#     # Initialize itr with our detected start point
#     itr = start_itr 
    
#     # --- 3. THE DATA FAST-FORWARD ---
#     if itr > 0:
#         print(f"[*] Fast-forwarding minibatch to iteration {itr}...")
#         for _ in range(itr):
#             minibatch.next_minibatch_feed_dict(placeholders=placeholders)

#     try:
#         while not minibatch.end():
#             # Construct feed dictionary
#             feed_dict = minibatch.next_minibatch_feed_dict(placeholders=placeholders)
#             feed_dict = minibatch.update_feed_dict(
#                 feed_dict=feed_dict,
#                 dropout=FLAGS.dropout,
#                 placeholders=placeholders)

#             t = time.time()

#             # Training step
#             outs = sess.run([opt.opt_op, opt.cost, opt.batch_edge_type_idx], feed_dict=feed_dict)
#             train_cost = outs[1]
#             batch_edge_type = outs[2]

#             # Print progress
#             if itr % PRINT_PROGRESS_EVERY == 0:
#                 val_auc, val_auprc, val_apk = get_accuracy_scores(
#                     minibatch.val_edges, minibatch.val_edges_false,
#                     minibatch.idx2edge_type[minibatch.current_edge_type_idx])

#                 print("Epoch:", "%04d" % (epoch + 1), "Iter:", "%04d" % (itr + 1), "Edge:", "%04d" % batch_edge_type,
#                     "train_loss=", "{:.5f}".format(train_cost),
#                     "val_roc=", "{:.5f}".format(val_auc), "val_auprc=", "{:.5f}".format(val_auprc),
#                     "val_apk=", "{:.5f}".format(val_apk), "time=", "{:.5f}".format(time.time() - t))
            
#             itr += 1

#             # --- PERIODIC SAVE ---
#             if itr % 20000 == 0:
#                 checkpoint_path = f"checkpoints/pfe_iter_{itr}.ckpt"
#                 if not os.path.exists('checkpoints'):
#                     os.makedirs('checkpoints')
#                 saver.save(sess, checkpoint_path)
#                 print(f"!!! CHECKPOINT SAVED AT ITERATION {itr} !!!")

#     except KeyboardInterrupt:
#         print("\n[!] Training interrupted by user.")
#         choice = input("Do you want to save a checkpoint before exiting? (1 for Yes / 0 for No): ")
        
#         if choice == '1':
#             # Replace 'saver' and 'sess' with the variable names used in your code
#             import os
#             checkpoint_path = os.path.join(FLAGS.checkpoint_dir, 'manual_interrupt_save.ckpt')
#             saver.save(sess, checkpoint_path)
#             print(f"[*] Checkpoint saved successfully at: {checkpoint_path}")
#         else:
#             print("[*] Exiting without saving.")
        
#         # Optional: Exit the program
#         import sys
#         sys.exit(0)

#     # Reset start_itr to 0 after the first epoch is finished 
#     # so the second epoch starts from the beginning of the data
#     start_itr = 0 
print("\n[*] Skipping Training... Moving directly to Final Evaluation!")


# =============== START MEDBOT CLIENT ===============

print("\n[*] Initializing MedBot Architecture...")

# 1. Restore the trained weights (Make sure this path is correct!)
CHECKPOINT_PATH = "checkpoints/pfe_iter_60000.ckpt" 

if os.path.exists(CHECKPOINT_PATH + ".index"):
    saver.restore(sess, CHECKPOINT_PATH)
    print(f"[*] MedBot Brain Successfully Loaded from {CHECKPOINT_PATH}")
else:
    print(f"[!] ERROR: Checkpoint not found at {CHECKPOINT_PATH}")
    sys.exit(1)

# 2. Interactive Loop
print("\n" + "="*40)
print("       MEDBOT INTERACTION CLIENT       ")
print("="*40)

while True:
    print("\nExample IDs: CID000003362 (Warfarin), CID000002244 (Aspirin)")
    d1 = input("Enter Drug A ID (or 'exit'): ").strip()
    if d1.lower() == 'exit': break
    d2 = input("Enter Drug B ID: ").strip()
    
    if d1 not in drug2idx or d2 not in drug2idx:
        print("Error: One of the drugs was not found in the database.")
        continue
        
    idx_a = drug2idx[d1]
    idx_b = drug2idx[d2]
    
    # We test for Side Effect 0 (You can prompt the user for this later!)
    se_idx = 0 
    
    # Feed the exact format the Decagon optimizer expects
    feed_dict = {
        placeholders['batch']: [[idx_a, idx_b]], 
        placeholders['batch_edge_type_idx']: se_idx,
        placeholders['batch_row_edge_type']: 1,  
        placeholders['batch_col_edge_type']: 1,
        placeholders['dropout']: 0.0
    }
    
    try:
        # Run the prediction through the optimizer's prediction tensor
        score = sess.run(opt.predictions, feed_dict=feed_dict)
        
        # Convert logit to probability
        prob = 1 / (1 + np.exp(-score[0])) 
        
        print(f"\nAnalyzing {d1} + {d2}...")
        risk = int(prob * 20)
        bar = "#" * risk + "-" * (20 - risk)
        print(f"Risk Level: [{bar}] {prob:.2%}")
        
        if prob > 0.5:
            print("ALERT: Dangerous Interaction Predicted!")
        else:
            print("Result: Interaction appears safe.")
            
    except Exception as e:
        print(f"Prediction Error: {e}")

        
# ==============================================================
# THE FIX: Give the model the graph structure before testing
# ==============================================================
# feed_dict = minibatch.next_minibatch_feed_dict(placeholders=placeholders)
# feed_dict = minibatch.update_feed_dict(
#     feed_dict=feed_dict,
#     dropout=0.0,
#     placeholders=placeholders
# )
# print("Optimization finished!")

# for et in range(num_edge_types):
#     roc_score, auprc_score, apk_score = get_accuracy_scores(
#         minibatch.test_edges, minibatch.test_edges_false, minibatch.idx2edge_type[et])
    
#     # Format the array part like [01, 01, 198]
#     edge_array = "[{:02d}, {:02d}, {:02d}]".format(*minibatch.idx2edge_type[et])
    
#     # Print everything side-by-side using unified {} formatting
#     print("Edge: {:04d} {} AUROC: {:.5f} | AUPRC: {:.5f} | AP@k: {:.5f}".format(
#         et, edge_array, roc_score, auprc_score, apk_score))
