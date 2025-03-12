# Hopfield network crate

## Purpose

This repository provides a set of functions and structures for building and 
training Hopfield-like, artificial neural networks.

## Status

This repository implements three different kinds of hopfield networks:

1. The original Hopfield network with both Hebbian learning and Storkey learning; and,
2. The "Modern Hopefield Network" described in eg Millidge's 2022 paper; and,
3. An associative memory based on Predictive Coding Networks.

Each of these models have their benefits and drawbacks. With this repository it
is easy to compare their performance.

## Running the example

To run the attractor network example based on Szilágyi's paper from 2017, run
the example like so:

```bash
python3 scripts/multiple_patterns.py \
  | cargo run --release --example attractor_network \
  |  gnuplot -e "plot '-' w lp; pause 99"
```

More examples to follow.

## References

- "Universal Hopfield Networks: A general framework for single-shot associative 
  memory models", B. Millidge et al, 2022
- "Hopfield Networks is all you need", H. Ramsauer et al, 2021
- "Breeding novel solutions in the brain: a model of Darwinian neurodynamics", v2,
  A. Szilágyi, 2017
- "Palimpsest memories: a new high-capacity forgetful learning rule for Hopﬁeld
  networks", A. Storkey, 2000
- "Associative Memories via Predictive Coding", T. Salvatori et al, 2021
