import math

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score


def _get_affinity_module(model):
  """
  Try common attribute names used in TGN implementations.
  """
  for name in ["affinity_score", "merge_layer", "decoder", "edge_predictor", "link_predictor"]:
    if hasattr(model, name):
      return getattr(model, name)
  raise AttributeError(
    "Could not find an affinity/decoder module on the model. "
    "Expected one of: affinity_score, merge_layer, decoder, edge_predictor, link_predictor."
  )


def _edge_scores_no_side_effects(model, sources, destinations, timestamps, edge_idxs, n_neighbors, use_memory=True):
  """
  Compute sigmoid scores for a batch of edges, while ensuring memory is NOT permanently updated.
  We do this by backing up and restoring memory around the forward pass.
  """
  mem = getattr(model, "memory", None)
  if use_memory and mem is not None:
    mem_backup = mem.backup_memory()
  else:
    mem_backup = None

  try:
    src_emb, dst_emb, _ = model.compute_temporal_embeddings(
      sources, destinations, destinations, timestamps, edge_idxs, n_neighbors
    )
    affinity = _get_affinity_module(model)
    prob = affinity(src_emb, dst_emb).sigmoid()
  finally:
    if mem_backup is not None:
      mem.restore_memory(mem_backup)

  return prob


def _edge_scores_update_memory(model, sources, destinations, timestamps, edge_idxs, n_neighbors):
  """
  Compute sigmoid scores for a batch of POSITIVE edges, allowing the model to update memory
  in the usual way (if enabled). This mirrors standard TGN behavior where memory updates are driven
  by observed interactions.
  """
  src_emb, dst_emb, _ = model.compute_temporal_embeddings(
    sources, destinations, destinations, timestamps, edge_idxs, n_neighbors
  )
  affinity = _get_affinity_module(model)
  prob = affinity(src_emb, dst_emb).sigmoid()
  return prob


def eval_edge_prediction(model, pos_data, neg_data, n_neighbors, batch_size=200):
  """
  Evaluate link prediction using EXPLICIT negatives.

  Semantics:
  - We process POSITIVE interactions in temporal order (batching), which updates memory as usual.
  - We also score NEGATIVE edges (the complement set) up to the same time horizon, but we do NOT
    permanently update memory on them (backup/restore around scoring).
  - This yields scores for ALL provided positives and ALL provided negatives.

  Returns: (acc, ap, auc)
  """
  val_scores = []
  val_labels = []

  with torch.no_grad():
    model = model.eval()

    TEST_BATCH_SIZE = batch_size

    pos_n = len(pos_data.sources)
    neg_n = len(neg_data.sources)

    pos_batches = math.ceil(pos_n / TEST_BATCH_SIZE) if pos_n > 0 else 0

    neg_ptr = 0
    use_memory = (getattr(model, "memory", None) is not None)

    for k in range(pos_batches):
      s_idx = k * TEST_BATCH_SIZE
      e_idx = min(pos_n, s_idx + TEST_BATCH_SIZE)

      # --- positives ---
      p_src = pos_data.sources[s_idx:e_idx]
      p_dst = pos_data.destinations[s_idx:e_idx]
      p_ts = pos_data.timestamps[s_idx:e_idx]
      p_ei = pos_data.edge_idxs[s_idx:e_idx]

      if len(p_src) > 0:
        p_prob = _edge_scores_update_memory(model, p_src, p_dst, p_ts, p_ei, n_neighbors)
        val_scores.append(p_prob.squeeze().cpu().numpy())
        val_labels.append(np.ones(len(p_src), dtype=np.float32))

      # --- negatives up to the current positive time horizon ---
      if len(p_ts) > 0:
        t_horizon = float(np.max(p_ts))
      else:
        t_horizon = None

      # Consume negatives with timestamp <= t_horizon (in chunks)
      while t_horizon is not None and neg_ptr < neg_n and float(neg_data.timestamps[neg_ptr]) <= t_horizon:
        neg_end = min(neg_n, neg_ptr + TEST_BATCH_SIZE)
        # ensure we don't step past horizon inside this chunk; shrink if needed
        while neg_end > neg_ptr and float(neg_data.timestamps[neg_end - 1]) > t_horizon:
          neg_end -= 1
        if neg_end == neg_ptr:
          break

        n_src = neg_data.sources[neg_ptr:neg_end]
        n_dst = neg_data.destinations[neg_ptr:neg_end]
        n_ts = neg_data.timestamps[neg_ptr:neg_end]
        n_ei = neg_data.edge_idxs[neg_ptr:neg_end]

        n_prob = _edge_scores_no_side_effects(model, n_src, n_dst, n_ts, n_ei, n_neighbors, use_memory=use_memory)
        val_scores.append(n_prob.squeeze().cpu().numpy())
        val_labels.append(np.zeros(len(n_src), dtype=np.float32))

        neg_ptr = neg_end

    # If there are remaining negatives after the last positive timestamp, score them too.
    while neg_ptr < neg_n:
      neg_end = min(neg_n, neg_ptr + TEST_BATCH_SIZE)
      n_src = neg_data.sources[neg_ptr:neg_end]
      n_dst = neg_data.destinations[neg_ptr:neg_end]
      n_ts = neg_data.timestamps[neg_ptr:neg_end]
      n_ei = neg_data.edge_idxs[neg_ptr:neg_end]

      n_prob = _edge_scores_no_side_effects(model, n_src, n_dst, n_ts, n_ei, n_neighbors, use_memory=use_memory)
      val_scores.append(n_prob.squeeze().cpu().numpy())
      val_labels.append(np.zeros(len(n_src), dtype=np.float32))
      neg_ptr = neg_end

  if len(val_scores) == 0:
    return float("nan"), float("nan"), float("nan")

  pred_score = np.concatenate([np.atleast_1d(x) for x in val_scores])
  true_label = np.concatenate([np.atleast_1d(x) for x in val_labels])

  pred_label = (pred_score > 0.5).astype(np.float32)
  acc = float((pred_label == true_label).mean())

  ap = float("nan")
  auc = float("nan")
  try:
    ap = float(average_precision_score(true_label, pred_score))
  except Exception:
    pass
  try:
    auc = float(roc_auc_score(true_label, pred_score))
  except Exception:
    pass

  return acc, ap, auc


def eval_node_classification(tgn, decoder, data, edge_idxs, batch_size, n_neighbors):
  pred_prob = np.zeros(len(data.sources))
  num_instance = len(data.sources)
  num_batch = math.ceil(num_instance / batch_size)

  with torch.no_grad():
    decoder.eval()
    tgn.eval()
    for k in range(num_batch):
      s_idx = k * batch_size
      e_idx = min(num_instance, s_idx + batch_size)

      sources_batch = data.sources[s_idx: e_idx]
      destinations_batch = data.destinations[s_idx: e_idx]
      timestamps_batch = data.timestamps[s_idx:e_idx]
      edge_idxs_batch = edge_idxs[s_idx: e_idx]

      source_embedding, destination_embedding, _ = tgn.compute_temporal_embeddings(
        sources_batch,
        destinations_batch,
        destinations_batch,
        timestamps_batch,
        edge_idxs_batch,
        n_neighbors
      )
      pred_prob_batch = decoder(source_embedding).sigmoid()
      pred_prob[s_idx: e_idx] = pred_prob_batch.cpu().numpy()

  auc_roc = roc_auc_score(data.labels, pred_prob)
  return auc_roc
