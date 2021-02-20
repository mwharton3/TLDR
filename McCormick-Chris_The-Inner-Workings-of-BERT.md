"The Inner Workings of BERT" is a short eBook by Chris McCormick that provides a concise overview of the BERT model architecture. BERT is a general purpose language encoding model used in many Natural Language Processing tasks.

*Read time: 6-7min*

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
Note: this all happens after positional encodings are added to input embeddings.

### Single-headed attention
**Self-attention:** to produce an "enhanced embedding" of an input (word-level) embedding, we do the following:
* Compute dot product between every input embedding and the others (quadratic dependence operation)
* Apply softmax to compute weights across all dot product results (sums to 1)
* Compute weighted average between all of these dot products to get an enhanced embedding

One important missing deatil: each embedding, before being used in the above sequence, is projected into a new vector space using a projection matrix (which is composed of tunable weights). There are three projection matrices:
* Query - projection applied to input word
* Key - projection applied to context words
* Value - projection applied to outputs before computing weighted average

All 3 of these matrices are 768x768, because they are transforming each (length 768) input embedding to a new space.

**Feed-forward network:** after computing enhanced embeddings, each weighted average is sent through a simple feed-forward neural network to create another (length 768) vector, at which point we've now encoded our input sequence. Sizes are:
* input - 768 neurons
* hidden - 4x 768 = 3072 neurons
* output - 768 neurons.

**Parameter counts:**
* 3 * 768 * 768 = 1,769,472 weights for self-attention
* 2 * 768 * 3072 = 4,718,592 weights for feed-forward neural network
TOTAL PARAMETERS PER LAYER: ~6.5M
TOTAL ENCODER PARAMETERS IN ALL 12x LAYERS = ~78M

### Multi-headed attention
The above covers single-head attention. Multi-head attention simply generates multiple "enhanced" embeddings for each input embedding, and then aggregates them. To cut down on parameter bloat, the **query**, **key**, and **value** projection matrices reduce inputs down to length 64 (instead of 768) and *then* they're aggregated. This gets us to an output analagous to the single head attention output embedding (still length 768).

**Parameter counts:**
Two primary versions exist of BERT:
* BERT base – 12 layers (transformer blocks), 12 attention heads, and **110M params**.
* BERT Large – 24 layers, 16 attention heads and, **340M params**.

# Part 3 - Pre-Training Tasks
BERT was trained on two data sets:
* BookCorpus (from University of Toronto) - 800M words
* Wikipedia - 2,500M words

Tasks as mentioned above are MLM and NSP.
### Masked Language Model (MLM)
* 12% of tokens randomly withheld from input sentences using a `[MASK]` token
* Replace 1.5% of tokens with a random token
* Mark another 1.5% of token for predicton, but don't change them
* Leave 85% of the tokens untouched
BERT's authors landed on these numbers empirically.


### Next Sentence Prediction (NSP)
A "sentence" in the BERT paradigm is just a randomly sampled set of tokens of length ≤512. It has nothing to do with actual sentences.

For a "positive" label, stick a `[SEP]` somewhere in the middle of a long span. For negative labels, separate two unrelated spans with a `[SEP]` token.

### Concurrent training
To train BERT, these tasks are simultaneously trained. One output goes to NSP prediction and another goes to a variable number of MLM predictions. Loss is backpropagated simultaneously.

### Training Task Improvements
Goals to improve are generally trying to:
* Reduce size of the model
* Improve support for non-English languages
* Reduce the training cost
* Outperform BERT in NLP benchmarks

**ALBERT**
Instead of masking subwords, ALBERT would mask whole words (which might be multiple tokens) and sometimes multiple words. This is "N-gram masking". Outperforms BERT, but only at the `xxlarge` size.

ALBERT's authors also got rid of NSP and used Sentence Order Prediction (SOP), which predicts whether two sentences are in order or are swapped (requires deeper understanding).

**RoBERTa** and **XLNet**
Remove NSP altogether, among other minor tweaks.

**ELECTRA**
New from Google, used a generator network to replace words with plausible substitutions. Then ask BERT to predict whether a word was swapped out or not. Generally expected to perform better than BERT.
