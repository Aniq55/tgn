#!/usr/bin/env python3
import math
import logging
import time
import sys
import argparse
import torch
import numpy as np
import pickle
from pathlib import Path

from evaluation.evaluation import eval_edge_prediction
from model.tgn import TGN
from utils.utils import EarlyStopMonitor, RandEdgeSampler, get_neighbor_finder
from utils.data_processing import get_data, compute_time_statistics

from loguru import logger as loguru1

MEMORY_DIM_DICT = {
  "enron": 32,
  "reddit": 172,
  "uci": 100,
  "wikipedia": 172,
  "mooc": 172,
  "lastfm": 172,
  "Contacts": 172
}

# Define custom log levels
INFO1_LEVEL = 25
INFO2_LEVEL = 35

# Add custom levels to loguru
loguru1.level("INFO1", INFO1_LEVEL)
loguru1.level("INFO2", INFO2_LEVEL)

### Argument and global variables
parser = argparse.ArgumentParser('TGN self-supervised training')
parser.add_argument('-d', '--data', type=str, help='Dataset name (eg. wikipedia or reddit)',
                    default='wikipedia')
parser.add_argument('--other_data', type=str, help='Optional second dataset for testing (eg. ctig_dagger). If empty, no second-dataset eval is done.',
                    default='')
parser.add_argument('-x', '--distortion', type=str, help='distortion technique', default='')
parser.add_argument('--modelname', type=str, help='model name', default='TGN')
parser.add_argument('--bs', type=int, default=200, help='Batch_size')
parser.add_argument('--prefix', type=str, default='', help='Prefix to name the checkpoints')
parser.add_argument('--n_degree', type=int, default=10, help='Number of neighbors to sample')
parser.add_argument('--n_head', type=int, default=2, help='Number of heads used in attention layer')
parser.add_argument('--n_epoch', type=int, default=50, help='Number of epochs')
parser.add_argument('--n_layer', type=int, default=1, help='Number of network layers')
parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
parser.add_argument('--patience', type=int, default=5, help='Patience for early stopping')
parser.add_argument('--n_runs', type=int, default=1, help='Number of runs')
parser.add_argument('--drop_out', type=float, default=0.1, help='Dropout probability')
parser.add_argument('--gpu', type=int, default=0, help='Idx for the gpu to use')
parser.add_argument('--node_dim', type=int, default=100, help='Dimensions of the node embedding')
parser.add_argument('--time_dim', type=int, default=100, help='Dimensions of the time embedding')
parser.add_argument('--backprop_every', type=int, default=1, help='Every how many batches to backprop')
parser.add_argument('--use_memory', action='store_true', help='Whether to augment the model with a node memory')
parser.add_argument('--embedding_module', type=str, default="graph_attention", choices=[
  "graph_attention", "graph_sum", "identity", "time"], help='Type of embedding module')
parser.add_argument('--message_function', type=str, default="identity", choices=[
  "mlp", "identity"], help='Type of message function')
parser.add_argument('--memory_updater', type=str, default="gru", choices=[
  "gru", "rnn"], help='Type of memory updater')
parser.add_argument('--aggregator', type=str, default="last", help='Type of message aggregator')
parser.add_argument('--memory_update_at_end', action='store_true', help='Whether to update memory at the end or at the start of the batch')
parser.add_argument('--message_dim', type=int, default=100, help='Dimensions of the messages')
parser.add_argument('--memory_dim', type=int, default=172, help='Dimensions of the memory for each user')
parser.add_argument('--different_new_nodes', action='store_true', help='Whether to use disjoint set of new nodes for train and val')
parser.add_argument('--uniform', action='store_true', help='take uniform sampling from temporal neighbors')
parser.add_argument('--randomize_features', action='store_true', help='Whether to randomize node features')
parser.add_argument('--use_destination_embedding_in_message', action='store_true', help='Whether to use the embedding of the destination node as part of the message')
parser.add_argument('--use_source_embedding_in_message', action='store_true', help='Whether to use the embedding of the source node as part of the message')
parser.add_argument('--dyrep', action='store_true', help='Whether to run the dyrep model')

try:
  args = parser.parse_args()
except:
  parser.print_help()
  sys.exit(0)

BATCH_SIZE = args.bs
NUM_NEIGHBORS = args.n_degree
NUM_NEG = 1
NUM_EPOCH = args.n_epoch
print(NUM_EPOCH)
NUM_HEADS = args.n_head
DROP_OUT = args.drop_out
GPU = args.gpu
DATA = args.data
OTHER_DATA = args.other_data
NUM_LAYER = args.n_layer
LEARNING_RATE = args.lr
NODE_DIM = args.node_dim
TIME_DIM = args.time_dim
USE_MEMORY = args.use_memory
MESSAGE_DIM = args.message_dim
MEMORY_DIM = MEMORY_DIM_DICT.get(DATA, 172)
DISTORTION = args.distortion
MODEL_NAME = args.modelname

LOG_FILE_val = f"/home/chri6578/Documents/aniq/tgn/logs/{DATA}_val.log"
LOG_FILE_test = f"/home/chri6578/Documents/aniq/tgn/logs/{DATA}_test.log"

loguru1.add(LOG_FILE_val, level=INFO1_LEVEL, format="{message}", filter=lambda record: record["level"].name == "INFO1")
loguru1.add(LOG_FILE_test, level=INFO2_LEVEL, format="{message}", filter=lambda record: record["level"].name == "INFO2")

Path("./saved_models/").mkdir(parents=True, exist_ok=True)
Path("./saved_checkpoints/").mkdir(parents=True, exist_ok=True)
MODEL_SAVE_PATH = f'./saved_models/{args.prefix}-{args.data}{args.distortion}.pth'
get_checkpoint_path = lambda epoch: f'./saved_checkpoints/{args.prefix}-{args.data}-{epoch}.pth'

### set up logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
Path("log/").mkdir(parents=True, exist_ok=True)
fh = logging.FileHandler('log/{}.log'.format(str(time.time())))
fh.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.WARN)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
ch.setFormatter(formatter)
logger.addHandler(fh)
logger.addHandler(ch)
logger.info(args)

### Extract data for training, validation and testing for the primary dataset
node_features, edge_features, full_data, train_data, val_data, test_data, new_node_val_data, new_node_test_data = get_data(
    DATA, DISTORTION,
    different_new_nodes_between_val_and_test=args.different_new_nodes,
    randomize_features=args.randomize_features
)

# If a second dataset is provided, we'll load it using the same temporal split time
other_node_features = other_edge_features = None
other_full_data = other_train_data = other_val_data = other_test_data = None
other_new_node_val_data = other_new_node_test_data = None
other_full_ngh_finder = other_test_rand_sampler = other_nn_test_rand_sampler = None

# --- compute t_value (split time) from primary dataset ---
t_value = None
if test_data is not None and len(test_data.timestamps) > 0:
    t_value = float(np.min(test_data.timestamps))
    logger.info(f"Using split_time/t_value={t_value} from primary dataset ({DATA}) for other dataset splits.")
else:
    logger.warning("Could not determine t_value from primary dataset; other dataset will use default splitting.")

# Load other dataset (if provided) and pass split_time so get_data can align it
if OTHER_DATA:
    other_node_features, other_edge_features, other_full_data, other_train_data, other_val_data, other_test_data, \
        other_new_node_val_data, other_new_node_test_data = get_data(
            OTHER_DATA,
            DISTORTION,
            split_time=t_value,
            different_new_nodes_between_val_and_test=args.different_new_nodes,
            randomize_features=args.randomize_features
        )

# Initialize neighbor finders for primary dataset
train_ngh_finder = get_neighbor_finder(train_data, args.uniform)
full_ngh_finder = get_neighbor_finder(full_data, args.uniform)

# If other dataset provided, initialize its neighbor finder(s)
if OTHER_DATA and other_full_data is not None:
    other_full_ngh_finder = get_neighbor_finder(other_full_data, args.uniform)

# Initialize negative samplers for primary dataset
train_rand_sampler = RandEdgeSampler(train_data.sources, train_data.destinations)
val_rand_sampler = RandEdgeSampler(full_data.sources, full_data.destinations, seed=0)
nn_val_rand_sampler = RandEdgeSampler(new_node_val_data.sources, new_node_val_data.destinations, seed=1)
test_rand_sampler = RandEdgeSampler(full_data.sources, full_data.destinations, seed=2)
nn_test_rand_sampler = RandEdgeSampler(new_node_test_data.sources, new_node_test_data.destinations, seed=3)

# Negative samplers for other dataset (if present)
if OTHER_DATA and other_full_data is not None:
    other_test_rand_sampler = RandEdgeSampler(other_full_data.sources, other_full_data.destinations, seed=100)
    other_nn_test_rand_sampler = RandEdgeSampler(other_new_node_test_data.sources, other_new_node_test_data.destinations, seed=101)

# === Device selection (robust) ===
import os
if torch.cuda.is_available():
    gpu_count = torch.cuda.device_count()
    if GPU < 0 or GPU >= gpu_count:
        logger.warning(f"Requested --gpu {GPU} out of range (found {gpu_count} GPUs). Falling back to GPU 0.")
        GPU = 0
    device = torch.device(f"cuda:{GPU}")
    logger.info(f"Using device: {device} (torch.cuda.device_count()={gpu_count})")
else:
    device = torch.device("cpu")
    logger.warning("CUDA not available — using CPU.")
    
print(device)

# Compute time statistics (primary dataset)
mean_time_shift_src, std_time_shift_src, mean_time_shift_dst, std_time_shift_dst = \
  compute_time_statistics(full_data.sources, full_data.destinations, full_data.timestamps)

for i in range(args.n_runs):
  results_path = "results/{}_{}.pkl".format(args.prefix, i) if i > 0 else "results/{}.pkl".format(args.prefix)
  Path("results/").mkdir(parents=True, exist_ok=True)

  # Initialize Model for primary dataset
  tgn = TGN(neighbor_finder=train_ngh_finder, node_features=node_features,
            edge_features=edge_features, device=device,
            n_layers=NUM_LAYER,
            n_heads=NUM_HEADS, dropout=DROP_OUT, use_memory=USE_MEMORY,
            message_dimension=MESSAGE_DIM, memory_dimension=MEMORY_DIM,
            memory_update_at_start=not args.memory_update_at_end,
            embedding_module_type=args.embedding_module,
            message_function=args.message_function,
            aggregator_type=args.aggregator,
            memory_updater_type=args.memory_updater,
            n_neighbors=NUM_NEIGHBORS,
            mean_time_shift_src=mean_time_shift_src, std_time_shift_src=std_time_shift_src,
            mean_time_shift_dst=mean_time_shift_dst, std_time_shift_dst=std_time_shift_dst,
            use_destination_embedding_in_message=args.use_destination_embedding_in_message,
            use_source_embedding_in_message=args.use_source_embedding_in_message,
            dyrep=args.dyrep)
  criterion = torch.nn.BCELoss()
  optimizer = torch.optim.Adam(tgn.parameters(), lr=LEARNING_RATE)
  tgn = tgn.to(device)

  # Diagnostic: confirm device and model placement
  try:
      logger.info(f"Model moved to device: {next(tgn.parameters()).device}")
      logger.info(f"torch.cuda.is_available(): {torch.cuda.is_available()}")
      if torch.cuda.is_available():
          logger.info(f"cuda device count: {torch.cuda.device_count()}, current_device: {torch.cuda.current_device()}, device name: {torch.cuda.get_device_name(torch.cuda.current_device())}")
          logger.info(f"cuda memory allocated (bytes): {torch.cuda.memory_allocated()}")
  except Exception as e:
      logger.warning(f"CUDA diagnostic failed: {e}")

  num_instance = len(train_data.sources)
  num_batch = math.ceil(num_instance / BATCH_SIZE)

  logger.info('num of training instances: {}'.format(num_instance))
  logger.info('num of batches per epoch: {}'.format(num_batch))
  idx_list = np.arange(num_instance)

  new_nodes_val_aps = []
  val_aps = []
  epoch_times = []
  total_epoch_times = []
  train_losses = []

  early_stopper = EarlyStopMonitor(max_round=args.patience)
  for epoch in range(NUM_EPOCH):
    start_epoch = time.time()
    ### Training

    # Reinitialize memory of the model at the start of each epoch
    if USE_MEMORY:
      tgn.memory.__init_memory__()

    # Train using only training graph
    tgn.set_neighbor_finder(train_ngh_finder)
    m_loss = []

    logger.info('start {} epoch'.format(epoch))
    for k in range(0, num_batch, args.backprop_every):
      loss = 0
      optimizer.zero_grad()

      # Custom loop to allow to perform backpropagation only every a certain number of batches
      for j in range(args.backprop_every):
        batch_idx = k + j

        if batch_idx >= num_batch:
          continue

        start_idx = batch_idx * BATCH_SIZE
        end_idx = min(num_instance, start_idx + BATCH_SIZE)
        sources_batch, destinations_batch = train_data.sources[start_idx:end_idx], \
                                            train_data.destinations[start_idx:end_idx]
        edge_idxs_batch = train_data.edge_idxs[start_idx: end_idx]
        timestamps_batch = train_data.timestamps[start_idx:end_idx]

        size = len(sources_batch)
        _, negatives_batch = train_rand_sampler.sample(size)

        with torch.no_grad():
          pos_label = torch.ones(size, dtype=torch.float, device=device)
          neg_label = torch.zeros(size, dtype=torch.float, device=device)

        tgn = tgn.train()
        pos_prob, neg_prob = tgn.compute_edge_probabilities(sources_batch, destinations_batch, negatives_batch,
                                                            timestamps_batch, edge_idxs_batch, NUM_NEIGHBORS)

        loss += criterion(pos_prob.squeeze(), pos_label) + criterion(neg_prob.squeeze(), neg_label)

      loss /= args.backprop_every

      loss.backward()
      optimizer.step()
      m_loss.append(loss.item())

      # Detach memory after 'args.backprop_every' number of batches so we don't backpropagate to
      # the start of time
      if USE_MEMORY:
        tgn.memory.detach_memory()

    epoch_time = time.time() - start_epoch
    epoch_times.append(epoch_time)

    ### Validation (primary)
    tgn.set_neighbor_finder(full_ngh_finder)

    if USE_MEMORY:
      train_memory_backup = tgn.memory.backup_memory()

    _, val_ap, val_auc = eval_edge_prediction(model=tgn,
                                          negative_edge_sampler=val_rand_sampler,
                                          data=val_data,
                                          n_neighbors=NUM_NEIGHBORS)
    
    if USE_MEMORY:
      val_memory_backup = tgn.memory.backup_memory()
      tgn.memory.restore_memory(train_memory_backup)

    # Validate on unseen nodes (primary)
    _, nn_val_ap, nn_val_auc = eval_edge_prediction(model=tgn,
                                                    negative_edge_sampler=val_rand_sampler,
                                                    data=new_node_val_data,
                                                    n_neighbors=NUM_NEIGHBORS)

    if USE_MEMORY:
      tgn.memory.restore_memory(val_memory_backup)

    new_nodes_val_aps.append(nn_val_ap)
    val_aps.append(val_ap)
    train_losses.append(np.mean(m_loss))

    # Save temporary results to disk
    pickle.dump({
      "val_aps": val_aps,
      "new_nodes_val_aps": new_nodes_val_aps,
      "train_losses": train_losses,
      "epoch_times": epoch_times,
      "total_epoch_times": total_epoch_times
    }, open(results_path, "wb"))

    total_epoch_time = time.time() - start_epoch
    total_epoch_times.append(total_epoch_time)

    logger.info('epoch: {} took {:.2f}s'.format(epoch, total_epoch_time))
    logger.info('Epoch mean loss: {}'.format(np.mean(m_loss)))
    logger.info('val auc: {}, new node val auc: {}'.format(val_auc, nn_val_auc))
    logger.info('val ap: {}, new node val ap: {}'.format(val_ap, nn_val_ap))

    # Early stopping
    if early_stopper.early_stop_check(val_ap):
      logger.info('No improvement over {} epochs, stop training'.format(early_stopper.max_round))
      logger.info(f'Loading the best model at epoch {early_stopper.best_epoch}')
      best_model_path = get_checkpoint_path(early_stopper.best_epoch)
      # load with map_location
      tgn.load_state_dict(torch.load(best_model_path, map_location=device))
      logger.info(f'Loaded the best model at epoch {early_stopper.best_epoch} for inference')
      tgn = tgn.to(device)
      tgn.eval()
      break
    else:
      torch.save(tgn.state_dict(), get_checkpoint_path(epoch))

  # Training finished; restore memory to train end for testing
  if USE_MEMORY:
      tgn.memory.restore_memory(train_memory_backup)
  
  ### Final validation (primary)
  val_acc, val_ap, val_auc = eval_edge_prediction(model=tgn,
                                        negative_edge_sampler=val_rand_sampler,
                                        data=val_data,
                                        n_neighbors=NUM_NEIGHBORS)

  if USE_MEMORY:
      tgn.memory.restore_memory(train_memory_backup)
    
  # Validate on unseen nodes (primary)
  nn_val_acc, nn_val_ap, nn_val_auc = eval_edge_prediction(model=tgn,
                                                  negative_edge_sampler=val_rand_sampler,
                                                  data=new_node_val_data,
                                                  n_neighbors=NUM_NEIGHBORS)
  
  if DISTORTION!='':
    DISTORTION_A = DISTORTION.split('_')[0]
    DISTORTION_B = DISTORTION.split('_')[-2]
  else:
    DISTORTION_A, DISTORTION_B = '', ''
  

  if USE_MEMORY:
    val_memory_backup = tgn.memory.backup_memory()
    
  
  ### Test on primary dataset
  tgn.embedding_module.neighbor_finder = full_ngh_finder
  test_acc, test_ap, test_auc = eval_edge_prediction(model=tgn,
                                                    negative_edge_sampler=test_rand_sampler,
                                                    data=test_data,
                                                    n_neighbors=NUM_NEIGHBORS)

  if USE_MEMORY:
    tgn.memory.restore_memory(val_memory_backup)

  # Test on unseen nodes (primary dataset)
  nn_test_acc, nn_test_ap, nn_test_auc = eval_edge_prediction(model=tgn,
                                                                          negative_edge_sampler=nn_test_rand_sampler,
                                                                          data=new_node_test_data,
                                                                          n_neighbors=NUM_NEIGHBORS)
  
  # Log primary dataset test results (include dataset label)
  # info2_message = [MODEL_NAME, DATA, DISTORTION_A, DISTORTION_B, "Test", "tdv",
  #                 f"{test_acc:.4f}", f"{test_auc:.4f}", f"{test_ap:.4f}"]
  # loguru1.log("INFO2", '\t'.join(info2_message))

  # === Other dataset evaluation (use fresh model instance to avoid timeline/memory issues) ===
  if OTHER_DATA and other_full_data is not None:
    logger.info(f"Evaluating trained model on other dataset: {OTHER_DATA}")

    # Build fresh model sized for other dataset
    other_mean_time_shift_src, other_std_time_shift_src, other_mean_time_shift_dst, other_std_time_shift_dst = \
        compute_time_statistics(other_full_data.sources, other_full_data.destinations, other_full_data.timestamps)

    model_other = TGN(neighbor_finder=other_full_ngh_finder,
                      node_features=other_node_features,
                      edge_features=other_edge_features,
                      device=device,
                      n_layers=NUM_LAYER,
                      n_heads=NUM_HEADS,
                      dropout=DROP_OUT,
                      use_memory=USE_MEMORY,
                      message_dimension=MESSAGE_DIM,
                      memory_dimension=MEMORY_DIM,
                      memory_update_at_start=not args.memory_update_at_end,
                      embedding_module_type=args.embedding_module,
                      message_function=args.message_function,
                      aggregator_type=args.aggregator,
                      memory_updater_type=args.memory_updater,
                      n_neighbors=NUM_NEIGHBORS,
                      mean_time_shift_src=other_mean_time_shift_src,
                      std_time_shift_src=other_std_time_shift_src,
                      mean_time_shift_dst=other_mean_time_shift_dst,
                      std_time_shift_dst=other_std_time_shift_dst,
                      use_destination_embedding_in_message=args.use_destination_embedding_in_message,
                      use_source_embedding_in_message=args.use_source_embedding_in_message,
                      dyrep=args.dyrep)

    # load trained weights into model_other (assumes same architecture)
    model_other.load_state_dict(tgn.state_dict())
    model_other = model_other.to(device)
    model_other.eval()

    # Initialize memory for the other dataset so updates are forward-only
    if USE_MEMORY:
        model_other.memory.__init_memory__()

    # ensure neighbor finder for embedding module is set to other dataset
    model_other.embedding_module.neighbor_finder = other_full_ngh_finder

    # Evaluate other dataset old nodes (tdv)
    other_test_acc, other_test_ap, other_test_auc = eval_edge_prediction(
        model=model_other,
        negative_edge_sampler=other_test_rand_sampler,
        data=other_test_data,
        n_neighbors=NUM_NEIGHBORS
    )

    # Reset memory before evaluating unseen/new nodes (idv)
    if USE_MEMORY:
        model_other.memory.__init_memory__()

    # Evaluate other dataset unseen/new nodes (idv) if present
    # other_nn_test_acc = other_nn_test_ap = other_nn_test_auc = None
    # if other_new_node_test_data is not None and len(other_new_node_test_data.timestamps) > 0:
    #     other_nn_test_acc, other_nn_test_ap, other_nn_test_auc = eval_edge_prediction(
    #         model=model_other,
    #         negative_edge_sampler=other_nn_test_rand_sampler,
    #         data=other_new_node_test_data,
    #         n_neighbors=NUM_NEIGHBORS
    #     )

    # Log other dataset test results (include dataset label)
    # info2_other = [MODEL_NAME, OTHER_DATA, DISTORTION_A, DISTORTION_B, "Test", "tdv",
    #                f"{other_test_acc:.4f}", f"{other_test_auc:.4f}", f"{other_test_ap:.4f}"]
    # loguru1.log("INFO2", '\t'.join(map(str, info2_other)))

    # if other_nn_test_acc is not None:
    #     info2_other_new = [MODEL_NAME, OTHER_DATA, DISTORTION_A, DISTORTION_B, "Test", "idv",
    #                        f"{other_nn_test_acc:.4f}", f"{other_nn_test_auc:.4f}", f"{other_nn_test_ap:.4f}"]
    #     loguru1.log("INFO2", '\t'.join(map(str, info2_other_new)))
    
    # DONE: supply the causal model distance as an argument to this file
    # make sure we can import the distances.py from CES
    # import sys
    # from pathlib import Path
    # CES_DIR = Path("/home/chri6578/Documents/aniq/CES").resolve()
    # if str(CES_DIR) not in sys.path:
    #     sys.path.insert(0, str(CES_DIR))
    # from distances import distance_empirical
    
    ANIQ_DIR = Path("/home/chri6578/Documents/aniq").resolve()
    if str(ANIQ_DIR) not in sys.path:
        sys.path.insert(0, str(ANIQ_DIR))

    # then do normal package import
    from CES.distances import distance_empirical, oracle_accuracy_neg_sample

    # MEASURE: \bar{d}){0, dagger} through C_0, C_dagger 
    bar_d_0_dagger = distance_empirical(DATA, OTHER_DATA, T=1000, n_iters=10)
    
    
    # MEASURE: acc_0^0, acc_0^dagger
    acc_oracle = oracle_accuracy_neg_sample(DATA, DATA)
    acc_dagger = oracle_accuracy_neg_sample(DATA, OTHER_DATA)
    
    
    # UNIFIED LOGGING:
    # MODEL_NAME, O_AUC O_AP D_AUC D_AP distance
    info_combined = [MODEL_NAME, "Test", "tdv", 
                     f"{bar_d_0_dagger:.4f}",                     # \bar{d}){0, dagger}
                     f"{acc_oracle:.4f}", f"{acc_dagger:.4f}",    # acc_0^0, acc_0^dagger
                     f"{test_auc:.4f}",f"{other_test_auc:.4f}",   # AUC_star^0, AUC_star^dagger
                     f"{test_ap:.4f}", f"{other_test_ap:.4f}"]    # AP_star^0, AP_star^dagger
    
    # TODO: also include the results of the oracle test, with negative sampling
    
    loguru1.log("INFO2", '\t'.join(map(str, info_combined)))

  # Save model and final logs
  logger.info('Saving TGN model')
  if USE_MEMORY:
    tgn.memory.restore_memory(val_memory_backup)
  torch.save(tgn.state_dict(), MODEL_SAVE_PATH)
  logger.info('TGN model saved')
