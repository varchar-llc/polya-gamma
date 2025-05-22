# Polya-Gamma Sampler for Rust

[![Crates.io](https://img.shields.io/crates/v/polya-gamma.svg)](https://crates.io/crates/polya-gamma)
[![Documentation](https://docs.rs/polya-gamma/badge.svg)](https://docs.rs/polya-gamma)

A high-performance sampler for Polya-Gamma random variates.

## Installation

Run `cargo add polya-gamma` or add this to your `Cargo.toml`:

```toml
[dependencies]
polya-gamma = "0.5"
```

Regression features require additional dependencies:

```toml
[dependencies]
polya-gamma = { version = "0.5", features = ["regression"] }
```


## Quick Start

```rust
use polya_gamma::PolyaGamma;
use rand::{SeedableRng, rngs::StdRng};

fn main() {
    // Create a new Polya-Gamma sampler
    let mut pg = PolyaGamma::new();
    
    // Create a random number generator
    let mut rng = StdRng::seed_from_u64(42);
    
    // Draw a sample from PG(1.0, 0.5)
    let sample = pg.draw(1.0, 0.5, &mut rng);
    println!("Sample from PG(1.0, 0.5): {}", sample);
    
    // Draw multiple samples in parallel (requires rayon feature)
    let samples = pg.draw_vec_par(1.0, 0.5, 1000);
    println!("Drew {} samples in parallel", samples.len());
}
```
 
## Features

- **Exact Sampling**: Implements Devroye's algorithm for exact sampling from PG(1, c)
- **Parallel Processing**: Optional Rayon support for generating multiple samples in parallel
- **Bayesian Logistic Regression**: Built-in Gibbs sampler for Bayesian logistic regression
- **Documentation**: Comprehensive API documentation with examples

## Usage Examples

### Basic Sampling

```rust
use polya_gamma::PolyaGamma;
use rand::thread_rng;

let mut pg = PolyaGamma::new();
let mut rng = thread_rng();

// Draw a single sample
let sample = pg.draw(1.0, 0.5, &mut rng);

// Draw multiple samples
let samples = pg.draw_vec(1.0, 0.5, 100, &mut rng);
```

### Bayesian Logistic Regression

```rust
use polya_gamma::logistic::GibbsLogReg;
use nalgebra::{DMatrix, DVector};

// Create some example data
let x = DMatrix::from_row_slice(3, 2, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
let y = DVector::from_vec(vec![0.0, 1.0, 0.0]);

// Create and run the Gibbs sampler
let model = GibbsLogReg::new(x, y);
let samples = model.sample(1000, 100, 10); // 1000 samples, 100 burn-in, thinning=10
```

## Performance

The implementation is optimized for both single-threaded and parallel workloads. For large-scale sampling, the `rayon` feature provides significant speedups on multi-core systems. 
Even in the single-threaded case, this crate has far better performance than `BayesLogit`:
- If $b < 1.0$, this crate is about 2.5x faster.
- If $b == 1.0$, this crate is about 1.3x faster.
- If $b > 1.0$, this crate can be up to 75x faster(!)
If multithreaded operation is enabled, performance also scales with the number of cores available. See the [benchmarks](benches/comparison/plots/summary.txt) for details.

## Documentation

Full API documentation is available on [docs.rs](https://docs.rs/polya-gamma).

## References

1. Polson, N.G., Scott, J.G., & Windle, J. (2013). Bayesian Inference for Logistic Models Using Polya-Gamma Latent Variables. *Journal of the American Statistical Association*, 108(504): 1339–1349.
2. Windle, J., Polson, N.G., & Scott, J.G. (2014). Sampling Pólya-Gamma random variates: alternate and approximate techniques. arXiv:1405.0506.

## License

This project is licensed under the terms of the MIT OR Apache-2.0 license. See the [LICENSE-MIT](LICENSE-MIT) and [LICENSE-APACHE](LICENSE-APACHE) files for details.
