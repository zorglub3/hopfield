# Hopfield network crate

## Purpose

This repository provides a set of functions and structures for building and 
training Hopfield-like, artificial neural networks.

## Status

## Running the example

### Attractor network

```bash
python3 scripts/multiple_patterns.py \
  | cargo run --release --example attractor_network \
  |  gnuplot -e "plot '-' w lp; pause 99"
```

## References

