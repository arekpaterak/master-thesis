# Attention across problem sizes: insights from other domains

Question: what do other fields (language, vision, graph transformers, physical fields) do about variable-size
inputs?

They deal with the same two problems: running variable-size inputs efficiently, and making one model work
across sizes. Points marked *inference* are our own reasoning for NDS, not claims from the papers.

## 1. Mixed-size batches without padding waste ("packing")

- **Language models.** Efficient Sequence Packing without Cross-contamination: Accelerating Large Language
  Models without Impacting Performance (Krell et al., 2021) found that with naive padding, "padding tokens
  represent 50% of all tokens of the Wikipedia pre-training dataset at sequence length 512". Their fix
  concatenates several sequences into one row and adds an attention mask so they can't see each other
  ("cross-contamination"). That gave a 2x speed-up for BERT phase-2 pre-training.
- **Vision.** Patch n' Pack: NaViT, a Vision Transformer for any Aspect Ratio and Resolution (Dehghani et al.,
  2023) packs images of different resolutions into one sequence. It needs two changes: "Masked self attention
  and masked pooling. To prevent examples attending to each other, additional self-attention masks are
  introduced. Similarly, masked pooling … pool[s] the token representations within each example".
- **Kernel cost of the mask.** FlexAttention: A Programming Model for Generating Optimized Attention Kernels
  (Dong et al., 2024) shows that a dense B×N×N boolean "document mask" in standard SDPA costs real throughput:
  torchtune lost 25% as sequence length grew from 2k. FlexAttention's block-sparse document mask avoids that.

**For NDS:** mixing sizes in one batch needs NaViT's checklist:

- a block-diagonal attention mask;
- masked pooling;
- the one NaViT doesn't need: a masked instance norm, because NDS normalises over nodes.

Use FlexAttention or jagged nested tensors rather than a dense mask.

## 2. Graph transformers

- Recipe for a General, Powerful, Scalable Graph Transformer (Rampášek et al., 2022), GraphGPS, stacks local
  message passing on the real edges with global attention in every layer, "decoupling the local real-edge
  aggregation" from attention. Swapping in linear attention (Performer, BigBird) brings the cost to O(N + E)
  and "scales to graphs with several thousand nodes".
- **For NDS (inference):** NDS already has this shape. The neighbours and route layers are the local part,
  and the attention layers are the global part. GraphGPS gives a published name for the design. It also
  suggests a direct experiment for large N: replace the global attention with a linear variant and keep the
  local layers as they are.

## 3. Physical fields: neural operators

These models handle the same physical field sampled at different resolutions, the closest analogue to "the
same distribution at a different N".

- Fourier Neural Operator for Parametric Partial Differential Equations (Li et al., 2021) uses the same
  learned parameters "irrespective of the discretization", and does "zero-shot super-resolution: trained on a
  lower resolution directly evaluated on a higher resolution".
- Universal Physics Transformers: A Framework For Efficiently Scaling Neural Operators (Alkin et al., 2024):
  - **Fixed latent size.** Message passing pools any number of points into a fixed set of supernodes (2048),
    and a perceiver pools those into 512 latent tokens. The transformer only ever sees the fixed set.
  - **Variable training sizes.** Models were "trained on inputs between 8K and 24K points" and kept "stable
    performance across different number of input/outputs even though it has never seen that number … during
    training".
  - **Batching.** They "keep the number of mesh points constant between batches to avoid memory fluctuations,
    which allows using a larger batchsize".
- Transolver: A Fast Transformer Solver for PDEs on General Geometries (Wu et al., 2024) softly assigns each
  of N mesh points to M learned "slices" via a softmax, runs attention among the M slice tokens, then maps the
  result back to the points. This is linear in N, and the number of tokens attention sees doesn't depend
  on N.

**For NDS (inference):**

- **A fixed-size latent bottleneck** (UPT or Transolver style) is the cleanest way to make the attention
  pattern independent of N. That connects to section 4.
- **The catch** is that destroy decisions are made per node and are fine-grained, while these models target
  smooth fields. A bottleneck could only be one context path alongside per-node embeddings, not a replacement
  for them.
- **The analogy is imperfect.** Larger TSP instances in the unit square are denser, so typical distances
  shrink roughly like 1/√N. This is less like "finer resolution of the same field" and more like a
  distribution shift in local distances. Normalising distance-based features by a size-aware scale is worth
  trying.
- **UPT's constant-points-per-batch trick** maps directly onto per-batch size sampling: scale the batch size
  so B·N stays roughly constant.

## 4. Why attention degrades across sizes, and a cheap fix

- From Local Structures to Size Generalization in Graph Neural Networks (Yehudai et al., 2021) shows that
  GNNs fail to generalise by size when the distribution of local structures ("d-patterns", a generalisation
  of node degree) differs between small and large graphs.
  - **For NDS:** the neighbours and route encoding is local and has fixed degree (previous and next node),
    which works in our favour. Global attention has no such guarantee.
- Overcoming a Theoretical Limitation of Self-Attention (Chiang & Cholak, 2022) shows the softmax gets more
  diffuse as length n grows. Their fix is "multiplying attention logits by log n", which also improved length
  generalisation in machine translation.
- Improving Generalization of Neural Vehicle Routing Problem Solvers Through the Lens of Model Architecture
  (Xiao et al., 2024) carries this into VRP solvers. They show that attention entropy is at least on the
  order of ln n, and add an Entropy-based Scaling Factor log_{n_train}(n_test) to the attention logits. It
  equals 1 at the training size, so it doesn't change behaviour there.
- **For NDS:** it applies in two places:
  - the encoder's self-attention;
  - the decoder's softmax over N nodes when choosing which node to remove (inference: not covered by the
    papers).

### How the scaling works

Each attention layer computes `softmax(q·kᵀ / √d)` over the N nodes. The logits come from per-node linear
projections, so their range is roughly the same at any instance size. For a fixed logit range Δ, the attention
entropy is at least ln N − Δ. As N grows, attention spreads over more nodes and each node gets weight on the
order of 1/N; the layer's output drifts toward the average of all nodes. A model trained on CVRP100 and run on
CVRP1000 sees flatter attention patterns than it ever saw in training. That is one plausible cause of the
size-generalisation drop, not the only one.

Multiplying the logits by s(N) before the softmax fixes this. If one node's logit beats the rest by a margin δ,
its relative weight becomes N^δ, which grows with N and offsets the larger number of competing nodes. The
amount of attention going to the few most relevant nodes stays roughly constant across sizes.

| | Factor | At the training size | Retraining needed? |
|---|---|---|---|
| Chiang & Cholak | s = ln N | Changes behaviour (about ×4.6 at N=100) | Yes, train with it from the start |
| Xiao et al. (ESF) | s = ln N / ln N_train | Exactly 1 | No, it can be added to an existing checkpoint at inference |

For example, ESF going from size 100 to size 1000 gives s = ln 1000 / ln 100 = 1.5.

Where it applies in NDS:

- **Encoder self-attention:** all attention layers of the joint and split encoders.
- **Not the neighbours and route layers.** The neighbours layer always looks at exactly two nodes, and the
  route layer aggregates within a route, whose length depends mostly on vehicle capacity, not on N. Neither
  has a softmax over N.
- **The decoder (inference).** The destroy decoder attends over N nodes and then takes a softmax over N nodes
  to choose which node to remove. The compatibility scorer clips logits with C·tanh, so they are bounded to
  [−C, C], which is exactly the bounded-range assumption. NDS removes 15 nodes at every size, so at large N
  those choices would become close to random.

How it relates to the rest of the model:

- **Instance norm** keeps the embedding statistics stable across sizes. That is a different source of drift
  from attention flattening; the two are complementary.
- **Multi-size training.** With ln N scaling, training on mixed sizes is natural because the factor is
  computed per batch from its N. With ESF, N_train isn't well defined when training covers a range, so a
  reference size is needed. Xiao et al. use a size baseline n_b (set to 50) during training.

Caveats:

- The evidence comes from machine translation (Chiang & Cholak) and constructive POMO-style VRP solvers
  (Xiao et al.), not from destroy or improvement models like NDS.
- The entropy result is a lower bound. Flatter attention isn't proven to be harmful, and a large instance
  might genuinely need broader attention.
- Inference: in routing, the relevant neighbourhood of a node is roughly its k nearest nodes whatever N is,
  which matches the "attend to a constant number of nodes" behaviour the scaling restores.
- The factor doesn't touch the other size-dependent shift: as N grows in the unit square, typical distances
  shrink, so coordinate-based features change too.

In this repo the scaling is the optional `attention_scaling` field on the encoder and decoder configs
(`core/models/layers/attention_scaling.py`), off by default. The cheapest experiment: evaluate a CVRP100
checkpoint on larger instances with `reference_size: 101`, encoder-only versus encoder plus decoder.
