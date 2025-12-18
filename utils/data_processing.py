import numpy as np
import random
import pandas as pd

split_times = {
    "reddit": (1882469.265, 2261813.658),
    "Contacts": (1625100, 2047800),
    "wikipedia": (1862652.1, 2218288.6),
    "uci": (3834800.6, 6714558.3),
    "SocialEvo": (16988541, 18711359),
    "mooc": (1917235, 2250151.6),
    "lastfm": (103764807.2, 120235473),
    "enron": (83843725.6, 93431801),
    "Flights": (90, 106),
    "UNvote": (1830297600, 2019686400),
    "CanParl": (283996800, 347155200),
    "USLegis": (8, 10),
    "UNtrade": (757382400, 883612800)
}


class Data:
  def __init__(self, sources, destinations, timestamps, edge_idxs, labels):
    self.sources = sources
    self.destinations = destinations
    self.timestamps = timestamps
    self.edge_idxs = edge_idxs
    self.labels = labels
    self.n_interactions = len(sources)
    self.unique_nodes = set(sources) | set(destinations)
    self.n_unique_nodes = len(self.unique_nodes)


def get_data_node_classification(dataset_name, use_validation=False):
  ### Load data and train val test split
  graph_df = pd.read_csv('./data/fake3_ml_{}.csv'.format(dataset_name))
  edge_features = np.load('./data/fake3_ml_{}.npy'.format(dataset_name))
  node_features = np.load('./data/ml_{}_node.npy'.format(dataset_name))

  # val_time, test_time = list(np.quantile(graph_df.ts, [0.70, 0.85]))
  val_time , test_time = 1862652.1, 2218288.6 # wikipedia

  sources = graph_df.u.values
  destinations = graph_df.i.values
  edge_idxs = graph_df.idx.values
  labels = graph_df.label.values
  timestamps = graph_df.ts.values

  random.seed(2020)

  train_mask = timestamps <= val_time if use_validation else timestamps <= test_time
  test_mask = timestamps > test_time
  val_mask = np.logical_and(timestamps <= test_time, timestamps > val_time) if use_validation else test_mask

  full_data = Data(sources, destinations, timestamps, edge_idxs, labels)

  train_data = Data(sources[train_mask], destinations[train_mask], timestamps[train_mask],
                    edge_idxs[train_mask], labels[train_mask])

  val_data = Data(sources[val_mask], destinations[val_mask], timestamps[val_mask],
                  edge_idxs[val_mask], labels[val_mask])

  test_data = Data(sources[test_mask], destinations[test_mask], timestamps[test_mask],
                  edge_idxs[test_mask], labels[test_mask])

  return full_data, node_features, edge_features, train_data, val_data, test_data


def get_data(dataset_name,
             distortion='',
             split_time=None,
             different_new_nodes_between_val_and_test=False,
             randomize_features=False,
             all_samples_dataset_name=None):
  """
  Loads a dataset of POSITIVE interactions from:
      /home/chri6578/Documents/aniq/CES/datasets/{distortion}ml_{dataset_name}.csv

  And (optionally) an ALL-SAMPLES dataset (superset) from:
      /home/chri6578/Documents/aniq/CES/datasets/ml_{all_samples_dataset_name}.csv

  Explicit negatives are computed as:
      negatives = all_samples \ positives
  where edge identity is defined by (u, i, ts).

  Returns:
    node_features, edge_features,
    full_data, train_data, val_data, test_data,
    new_node_val_data, new_node_test_data,
    neg_full_data, neg_train_data, neg_val_data, neg_test_data,
    neg_new_node_val_data, neg_new_node_test_data
  """
  graph_df = pd.read_csv(f'/home/chri6578/Documents/aniq/CES/datasets/{distortion}ml_{dataset_name}.csv')
  edge_features = np.load(f'/home/chri6578/Documents/aniq/CES/datasets/{distortion}ml_{dataset_name}.npy')
  node_features = np.load(f'/home/chri6578/Documents/aniq/CES/datasets/ml_{dataset_name}_node.npy')

  if split_time is None:
    val_time, test_time = list(np.quantile(graph_df.ts, [0.70, 0.85]))
  else:
    val_time = 0.5 * split_time
    test_time = split_time

  if randomize_features:
    node_features = np.random.rand(node_features.shape[0], node_features.shape[1])

  # --- positives ---
  sources = graph_df.u.values
  destinations = graph_df.i.values
  edge_idxs = graph_df.idx.values
  labels = graph_df.label.values
  timestamps = graph_df.ts.values

  full_data = Data(sources, destinations, timestamps, edge_idxs, labels)

  # --- explicit negatives ---
  if all_samples_dataset_name is None or str(all_samples_dataset_name).strip() == "":
    raise ValueError("all_samples_dataset_name must be provided to compute explicit negatives.")

  all_df = pd.read_csv(f'/home/chri6578/Documents/aniq/CES/datasets/ml_{all_samples_dataset_name}.csv')
  pos_df = graph_df.copy()

  EDGE_KEYS = ['u', 'i', 'ts']
  neg_df = (
    all_df
      .merge(pos_df[EDGE_KEYS], on=EDGE_KEYS, how='left', indicator=True)
      .query('_merge == "left_only"')
      .drop(columns=['_merge'])
  )
  neg_df['label'] = 0

  neg_sources = neg_df.u.values
  neg_destinations = neg_df.i.values
  neg_timestamps = neg_df.ts.values
  neg_edge_idxs = neg_df.idx.values
  neg_labels = neg_df.label.values

  neg_full_data = Data(neg_sources, neg_destinations, neg_timestamps, neg_edge_idxs, neg_labels)

  random.seed(2020)

  node_set = set(sources) | set(destinations)
  n_total_unique_nodes = len(node_set)

  # Compute nodes which appear at test time (based on positives)
  test_node_set = set(sources[timestamps > val_time]).union(
    set(destinations[timestamps > val_time]))
  new_test_node_set = set(random.sample(list(test_node_set), int(0.1 * n_total_unique_nodes)))

  # Masks for positives (new test nodes computed from positives)
  new_test_source_mask = graph_df.u.map(lambda x: x in new_test_node_set).values
  new_test_destination_mask = graph_df.i.map(lambda x: x in new_test_node_set).values
  observed_edges_mask = np.logical_and(~new_test_source_mask, ~new_test_destination_mask)

  # Train positives: before val_time and not involving any new test node
  train_mask = np.logical_and(timestamps <= val_time, observed_edges_mask)
  train_data = Data(sources[train_mask], destinations[train_mask], timestamps[train_mask],
                    edge_idxs[train_mask], labels[train_mask])

  # New node sets (based on positive train)
  train_node_set = set(train_data.sources).union(train_data.destinations)
  assert len(train_node_set & new_test_node_set) == 0
  new_node_set = node_set - train_node_set

  # Val/test masks for positives
  val_mask = np.logical_and(timestamps <= test_time, timestamps > val_time)
  test_mask = timestamps > test_time

  # Pos validation/test
  val_data = Data(sources[val_mask], destinations[val_mask], timestamps[val_mask],
                  edge_idxs[val_mask], labels[val_mask])
  test_data = Data(sources[test_mask], destinations[test_mask], timestamps[test_mask],
                   edge_idxs[test_mask], labels[test_mask])

  # New-node splits for positives
  if different_new_nodes_between_val_and_test:
    n_new_nodes = len(new_test_node_set) // 2
    val_new_node_set = set(list(new_test_node_set)[:n_new_nodes])
    test_new_node_set = set(list(new_test_node_set)[n_new_nodes:])

    edge_contains_new_val_node_mask = np.array(
      [(a in val_new_node_set or b in val_new_node_set) for a, b in zip(sources, destinations)])
    edge_contains_new_test_node_mask = np.array(
      [(a in test_new_node_set or b in test_new_node_set) for a, b in zip(sources, destinations)])

    new_node_val_mask = np.logical_and(val_mask, edge_contains_new_val_node_mask)
    new_node_test_mask = np.logical_and(test_mask, edge_contains_new_test_node_mask)
  else:
    edge_contains_new_node_mask = np.array(
      [(a in new_node_set or b in new_node_set) for a, b in zip(sources, destinations)])
    new_node_val_mask = np.logical_and(val_mask, edge_contains_new_node_mask)
    new_node_test_mask = np.logical_and(test_mask, edge_contains_new_node_mask)

  new_node_val_data = Data(sources[new_node_val_mask], destinations[new_node_val_mask],
                           timestamps[new_node_val_mask],
                           edge_idxs[new_node_val_mask], labels[new_node_val_mask])
  new_node_test_data = Data(sources[new_node_test_mask], destinations[new_node_test_mask],
                            timestamps[new_node_test_mask], edge_idxs[new_node_test_mask],
                            labels[new_node_test_mask])

  # --- Negative splits use SAME val/test times.
  # Note: for negatives, we do not remove "new_test_node_set" edges in train; they are simply non-interactions.
  # But for inductive eval (new-node), we mirror the positive notion: negatives whose endpoints include new nodes.
  neg_train_mask = neg_timestamps <= val_time
  neg_val_mask = np.logical_and(neg_timestamps <= test_time, neg_timestamps > val_time)
  neg_test_mask = neg_timestamps > test_time

  neg_train_data = Data(neg_sources[neg_train_mask], neg_destinations[neg_train_mask],
                        neg_timestamps[neg_train_mask], neg_edge_idxs[neg_train_mask],
                        neg_labels[neg_train_mask])

  neg_val_data = Data(neg_sources[neg_val_mask], neg_destinations[neg_val_mask],
                      neg_timestamps[neg_val_mask], neg_edge_idxs[neg_val_mask],
                      neg_labels[neg_val_mask])

  neg_test_data = Data(neg_sources[neg_test_mask], neg_destinations[neg_test_mask],
                       neg_timestamps[neg_test_mask], neg_edge_idxs[neg_test_mask],
                       neg_labels[neg_test_mask])

  # New-node splits for negatives
  if different_new_nodes_between_val_and_test:
    # reuse val_new_node_set / test_new_node_set
    neg_edge_contains_new_val_node_mask = np.array(
      [(a in val_new_node_set or b in val_new_node_set) for a, b in zip(neg_sources, neg_destinations)])
    neg_edge_contains_new_test_node_mask = np.array(
      [(a in test_new_node_set or b in test_new_node_set) for a, b in zip(neg_sources, neg_destinations)])

    neg_new_node_val_mask = np.logical_and(neg_val_mask, neg_edge_contains_new_val_node_mask)
    neg_new_node_test_mask = np.logical_and(neg_test_mask, neg_edge_contains_new_test_node_mask)
  else:
    neg_edge_contains_new_node_mask = np.array(
      [(a in new_node_set or b in new_node_set) for a, b in zip(neg_sources, neg_destinations)])
    neg_new_node_val_mask = np.logical_and(neg_val_mask, neg_edge_contains_new_node_mask)
    neg_new_node_test_mask = np.logical_and(neg_test_mask, neg_edge_contains_new_node_mask)

  neg_new_node_val_data = Data(neg_sources[neg_new_node_val_mask], neg_destinations[neg_new_node_val_mask],
                               neg_timestamps[neg_new_node_val_mask],
                               neg_edge_idxs[neg_new_node_val_mask],
                               neg_labels[neg_new_node_val_mask])

  neg_new_node_test_data = Data(neg_sources[neg_new_node_test_mask], neg_destinations[neg_new_node_test_mask],
                                neg_timestamps[neg_new_node_test_mask],
                                neg_edge_idxs[neg_new_node_test_mask],
                                neg_labels[neg_new_node_test_mask])

  print("The dataset has {} interactions, involving {} different nodes".format(full_data.n_interactions,
                                                                      full_data.n_unique_nodes))
  print("The training dataset has {} interactions, involving {} different nodes".format(
    train_data.n_interactions, train_data.n_unique_nodes))
  print("The validation dataset has {} interactions, involving {} different nodes".format(
    val_data.n_interactions, val_data.n_unique_nodes))
  print("The test dataset has {} interactions, involving {} different nodes".format(
    test_data.n_interactions, test_data.n_unique_nodes))
  print("The new node validation dataset has {} interactions, involving {} different nodes".format(
    new_node_val_data.n_interactions, new_node_val_data.n_unique_nodes))
  print("The new node test dataset has {} interactions, involving {} different nodes".format(
    new_node_test_data.n_interactions, new_node_test_data.n_unique_nodes))
  print("{} nodes were used for the inductive testing, i.e. are never seen during training".format(
    len(new_test_node_set)))

  print("The explicit negative dataset has {} interactions, involving {} different nodes".format(
    neg_full_data.n_interactions, neg_full_data.n_unique_nodes))
  print("The negative training dataset has {} interactions".format(neg_train_data.n_interactions))
  print("The negative validation dataset has {} interactions".format(neg_val_data.n_interactions))
  print("The negative test dataset has {} interactions".format(neg_test_data.n_interactions))

  return (node_features, edge_features,
          full_data, train_data, val_data, test_data,
          new_node_val_data, new_node_test_data,
          neg_full_data, neg_train_data, neg_val_data, neg_test_data,
          neg_new_node_val_data, neg_new_node_test_data)


def compute_time_statistics(sources, destinations, timestamps):
  last_timestamp_sources = dict()
  last_timestamp_dst = dict()
  all_timediffs_src = []
  all_timediffs_dst = []
  for k in range(len(sources)):
    source_id = sources[k]
    dest_id = destinations[k]
    c_timestamp = timestamps[k]
    if source_id not in last_timestamp_sources.keys():
      last_timestamp_sources[source_id] = 0
    if dest_id not in last_timestamp_dst.keys():
      last_timestamp_dst[dest_id] = 0
    all_timediffs_src.append(c_timestamp - last_timestamp_sources[source_id])
    all_timediffs_dst.append(c_timestamp - last_timestamp_dst[dest_id])
    last_timestamp_sources[source_id] = c_timestamp
    last_timestamp_dst[dest_id] = c_timestamp
  assert len(all_timediffs_src) == len(sources)
  assert len(all_timediffs_dst) == len(sources)
  mean_time_shift_src = np.mean(all_timediffs_src)
  std_time_shift_src = np.std(all_timediffs_src)
  mean_time_shift_dst = np.mean(all_timediffs_dst)
  std_time_shift_dst = np.std(all_timediffs_dst)

  return mean_time_shift_src, std_time_shift_src, mean_time_shift_dst, std_time_shift_dst
