"The Inner Workings of BERT" is a short eBook by Chris McCormick that provides a concise overview of the BERT model architecture. BERT is a general purpose sequence-to-sequence model used in many Natural Language Processing tasks.

## Introduction
* BERT is good at some things, not so great at others. Part 1 covers what these things are.
* Part 2 covers details inside the "black box". This is where the meat of the "Inner Workings" are unveiled.
* There have been several subtle BERT improvements (e.g. RoBERTa, XLNet, ALBERT). They're all the same basic architecture, but the tasks have been modified somewhat to pre-train BERT. Part 3 covers how these pre-training tasks differ somewhat from the original BERT task.

Prerequisites:
* Basics of neural networks
* Concept of "word embedding"
* You don't need to know about recurrent architectures (e.g. RNN, LSTM), as BERT is not one. "Attention is All You Need" was the title of the original BERT paper, this is what they refer to.

BERT's main contributions are the Masked Language Model (MLM) and Next Sentence Prediction (NSP) tasks.
* MLM - predict missing word
* NSP - binary prediction of whether or not a given sentence follows another

These tasks are only for pre-training. Part of the model is discarded once these objectives are trained on.

Important: no need for labeled data. Sequential text is enough (e.g. wikipedia).

Google trained this model using ~$10k's to ~$100k's worth of TPU compute time.

You can hook up whatever small neural network architecture you want to the final embedding layer (where the MLM and NSP pieces used to be). Training these weights is "fine-tuning".

Pre-training and fine-tuning (which together make transfer learning) allow us to train highly performant language models more quickly, on our own (~small) natural language data sets, with high performance.

# Part 1 - BERT Basics & Applications
You have to use BERT's tokens/embeddings, you can't plug in your own. This is okay, as BERT uses sub-word tokenization (only 60-80% of the 30,000 token BERT vocabulary are whole words). i.e. if the word "embedding" wasn't in its vocabulary, it still has sub-word tokens for "em", "bed", and "ding". In the off chance you can't reduce a word to subwords (seems pretty rare), there are tokens for individual characters (even weird/symbolic ones).

Example: "kroxldyphivc" -> "k", "ro", "x", "ld", "yp", "hi", "vc"

From a "black box" standpoint, BERT takes in a bunch of word embeddings (simultaneously, not in sequence) and transforms them to "enhanced" embeddings. Output dimensions are the same as the input dimensions (768 features each). The whole BERT model runs this "enhancement" 12x in sequence (i.e. 12 layers). Max 512 input features.

Each layer of BERT is a bunch of self-attention encoders. Each one takes a single input token, attends to all other tokens, and produces an enhanced embedding. This is an O(N^2) operation (quadratic dependence) on the input sequence length.

**Positional encoding:** Because BERT processes inputs in parallel, there's no notion of sequence. We fix this by adding positional embeddings to the input sequence. These positional embeddings are constant and unique to each input in the sequence.

**Special tokens:**
* `[CLS]` - for sentence-level classification tasks
* `[SEP]` - this and *Segment Embeddings* handle two-sentence tasks

### Applications
All BERT applications fall into three categories:
* Token classification - (e.g. named entity recognition (NER), question answering (find span where answer occurs in a passage of text)
* Text classification - classify a block of text, or perform a regression (e.g. sentiment analysis)
* Text-pair classification - compare two blocks of text (e.g. similarity)

Notes:
* The first token for an input sequence is always `[CLS]` (no matter the task). The corresponding output "enhanced embedding" can be used for text classification (ignoring the others).
* `[SEP]` is used to separate input sentences in the input for simlultaneous processing.

### What BERT can't do
* Text generation - BERT is only an encoder (not decoder). Each component is only half of a transformer, so we can't generate text with it (e.g. machine translation).
* Real time sequence - since **BERT** is **B**i-directional, it requires all input at once. Not great for things like auto-complete or speech recognition.
* Be sample-efficient. Language Modeling (i.e. next word prediction) gets a training sample for every word, but the MLM task reduces sample efficiency because there are only so many permutations of masking an input sequence.

# Part 2 - BERT Architecture
