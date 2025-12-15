#!/bin/bash

# datasets=("uci")
# datasets=("lastfm" "mooc")
datasets=("ctig_0")

for dataset in "${datasets[@]}" 
do
    echo "Running on dataset: $dataset"

    # real data
    for sample in {1...2}; do
        python train_self_supervised.py --n_epoch 5  --prefix tgn-attn --data "$dataset" --n_runs 1 --modelname TGN --patience 10
    done

    # for sample in {1..10}; do
    #     python train_self_supervised.py --n_epoch 100 --data "$dataset" --n_runs 1 --use_memory --memory_updater rnn --embedding_module time --prefix jodie_rnn --modelname JODIE --patience 10
    # done

    # for iter in {1..10}; do

    # for sample in {1..10}; do
    #     distort="shuffle_${sample}_"
    #     python train_self_supervised.py --n_epoch 100  --distortion "$distort" --prefix tgn-attn --data "$dataset" --n_runs 1 --modelname TGN --patience 10
    # done

    # for sample in {1..10}; do
    #     distort="shuffle_${sample}_"
    #     python train_self_supervised.py --n_epoch 100 --distortion "$distort" --data "$dataset" --n_runs 1  --use_memory --memory_updater rnn --embedding_module time --prefix jodie_rnn --modelname JODIE --patience 10
    # done

    # done

    # # distorted data: all samples
    # for sample in {1..2}; do
    #     distort="intense_5_${sample}_"
    #     python train_self_supervised.py --distortion "$distort" --prefix tgn-attn --data "$dataset" --n_runs 1 --modelname TGN
    #     python train_self_supervised.py --distortion "$distort" --data "$dataset" --n_runs 1  --use_memory --memory_updater rnn --embedding_module time --prefix jodie_rnn --modelname JODIE --n_epoch 4
    # done

done



# dataset           "enron" "reddit"    "uci"   "wikipedia" "lastfm"    "mooc"  "Contacts"
# --memory_dim      32      172         100     172         172         172     172

