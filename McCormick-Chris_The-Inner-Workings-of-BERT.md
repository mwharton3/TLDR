"The Inner Workings of BERT" is a short eBook by Chris McCormick that provides a concise overview of the BERT model architecture. BERT is a general purpose sequence-to-sequence model used in many Natural Language Processing tasks.

## Introduction
BERT is good at some things, not so great at others. Part 1 covers what these things are.

Part 2 covers details inside the "black box". This is where the meat of the "Inner Workings" are unveiled.

There have been several subtle BERT improvements (e.g. RoBERTa, XLNet, ALBERT). They're all the same basic architecture, but the tasks have been modified somewhat to pre-train BERT. Part 3 covers how these pre-training tasks differ somewhat from the original BERT task.

## Pre-requisites
* Basics of neural networks
* Concept of "word embedding"
* You don't need to know about recurrent architectures (e.g. RNN, LSTM), as BERT is not one. "Attention is All You Need" was the title of the original BERT paper, this is what they refer to.

## Transfer Learning
Pre-training and fine-tuning together make transfer learning

### Pre-training
BERT's main contributions are the Masked Language Model (MLM) and Next Sentence Prediction (NSP) tasks.
* MLM - predict missig word
* NSP - binary prediction of whether or not a given sentence follows another

These tasks are only for pre-training. Part of the model is discarded once these objectives are trained on.

Important: no need for labeled data. Sequential text is enough (e.g. wikipedia).

Google trained this model using ~$10k's to ~$100k's worth of TPU compute time.

### Fine-tuning
You can hook up whatever small neural network architecture you want to the final embedding layer (where the MLM and NSP pieces used to be). Training these weights is "fine-tuning".

### Transfer learning
Pre-training and fine-tuning allows us to train highly performant language models more quickly, on our own (~small) data sets, with high performance.


