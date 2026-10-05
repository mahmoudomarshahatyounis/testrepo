# Understanding the Skip-Gram Model (Word2Vec)

## 1. Overview
The **Skip-Gram model** is an unsupervised shallow neural network architecture introduced by Tomas Mikolov et al. at Google in 2013 as part of the **Word2Vec** toolkit. Its primary purpose is to **learn distributed, continuous vector representations (word embeddings) of words** from unstructured text corpora.

In these vector spaces, words that share common semantic and syntactic contexts are mapped to coordinates with high geometric proximity (e.g., measured via cosine similarity).

---

## 2. Core Architecture & Objective

### Architecture Comparison
* **Continuous Bag-of-Words (CBOW):** Uses surrounding context words to predict the target center word.
* **Skip-Gram:** Reverses the CBOW objective—it uses a single **target center word** to predict its surrounding **context words** within a sliding window of size $C$.

```
       [Context w_{t-2}]
      ^
     /
    /  [Context w_{t-1}]
   /  ^
  /  /
[Target Word w_t] 
  \  \
   \  v
    \  [Context w_{t+1}]
     v
       [Context w_{t+2}]
```

### Mathematical Objective
Given a sequence of training words $w_1, w_2, \dots, w_T$, the training objective is to maximize the average log probability across the corpus:

$$\mathcal{L} = \frac{1}{T} \sum_{t=1}^{T} \sum_{-c \le j \le c, \, j \ne 0} \log P(w_{t+j} \mid w_t)$$

where:
* $T$ is the total number of words in the text corpus.
* $c$ is the size of the training context window.
* $w_t$ is the center word at position $t$.
* $w_{t+j}$ is a context word.

---

## 3. Training Formulations

### Standard Softmax Formulation
The conditional probability $P(w_O \mid w_I)$ of predicting context word $w_O$ given center word $w_I$ is defined as:

$$P(w_O \mid w_I) = \frac{\exp({v'_{w_O}}^\top v_{w_I})}{\sum_{w=1}^{|V|} \exp({v'_w}^\top v_{w_I})}$$

where:
* $v_w$ is the input vector representation of word $w$.
* $v'_w$ is the output vector representation of word $w$.
* $|V|$ is the size of the vocabulary.

*Bottleneck:* Computing the denominator requires summing over all words in $|V|$, which becomes computationally intractable for vocabularies containing hundreds of thousands or millions of tokens ($O(|V|)$ per step).

### Optimization Techniques

1. **Skip-Gram with Negative Sampling (SGNS):**
   * Transforms the multiclass classification problem into binary classification (distinguishing the true context word from $k$ randomly drawn "noise" or negative words).
   * Objective function for a single pair $(w_I, w_O)$:

   $$\log \sigma({v'_{w_O}}^\top v_{w_I}) + \sum_{i=1}^{k} \mathbb{E}_{w_i \sim P_n(w)} \left[ \log \sigma(-{v'_{w_i}}^\top v_{w_I}) \right]$$

   * Reduces computational complexity from $O(|V|)$ to $O(k)$ per training step, where $k$ is typically $5\text{--}20$ for small datasets and $2\text{--}5$ for large datasets.

2. **Hierarchical Softmax:**
   * Replaces the flat softmax layer with a balanced binary tree (often a Huffman tree based on word frequencies).
   * Reduces the computation per word to $O(\log_2 |V|)$.

---

## 4. Key Strengths of Skip-Gram

* **Superior Performance on Rare Words:** Because context words are evaluated individually against the target center word rather than being averaged into a single representation (as in CBOW), infrequent words receive dedicated gradient updates and avoid being washed out by high-frequency tokens.
* **Semantic Vector Arithmetic:** Captures linear algebraic relationships between concepts:
  $$\vec{v}_{\text{king}} - \vec{v}_{\text{man}} + \vec{v}_{\text{woman}} \approx \vec{v}_{\text{queen}}$$
* **Scalability:** Scales efficiently over billions of words when paired with Negative Sampling and subsampling of frequent words.

---

## 5. Downstream NLP & Machine Learning Applications

1. **Information Retrieval & Semantic Search:**
   * Encoding documents and queries into vector space to compute dense semantic relevance rather than relying strictly on lexical matches (BM25/TF-IDF).
2. **Text Classification & Sentiment Analysis:**
   * Providing pretrained static embedding matrices for recurrent architectures, CNNs, or tabular feature representations.
3. **Recommender Systems:**
   * Item2Vec: Treating sequences of user interactions (e.g., clicks, purchases) as "sentences" and items as "words" to compute latent product similarities.
4. **Graph Representation Learning:**
   * Models like **DeepWalk** and **Node2Vec** run simulated random walks over graph edges, treating the generated node sequences as sentences and training Skip-Gram to derive node embeddings.