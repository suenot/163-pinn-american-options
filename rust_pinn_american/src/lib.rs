//! # PINN American Options
//!
//! Physics-Informed Neural Network for pricing American options.
//!
//! This crate implements a simple feedforward neural network that learns to solve
//! the Black-Scholes PDE with the American option free boundary condition using
//! the penalty method.
//!
//! ## Key Components
//!
//! - `NeuralNetwork` — feedforward network with backpropagation
//! - `AmericanPINNPricer` — PINN pricer with PDE residual + penalty loss
//! - `LsmBenchmark` — Longstaff-Schwartz Monte Carlo for comparison
//! - `BybitClient` — fetch crypto market data from Bybit
//! - `GreeksComputer` — compute option Greeks via finite differences

pub mod network;
pub mod pricer;
pub mod lsm;
pub mod greeks;
pub mod data;
pub mod backtest;

pub use network::NeuralNetwork;
pub use pricer::AmericanPINNPricer;
pub use lsm::lsm_american_option;
pub use greeks::GreeksComputer;
pub use data::BybitClient;
pub use backtest::run_backtest;

/// Option type: put or call.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum OptionType {
    Put,
    Call,
}

/// Option parameters.
#[derive(Debug, Clone)]
pub struct OptionParams {
    pub strike: f64,
    pub risk_free_rate: f64,
    pub volatility: f64,
    pub maturity: f64,
    pub option_type: OptionType,
    pub s_max: f64,
}

impl Default for OptionParams {
    fn default() -> Self {
        Self {
            strike: 100.0,
            risk_free_rate: 0.05,
            volatility: 0.2,
            maturity: 1.0,
            option_type: OptionType::Put,
            s_max: 300.0,
        }
    }
}

impl OptionParams {
    pub fn payoff(&self, s: f64) -> f64 {
        match self.option_type {
            OptionType::Put => (self.strike - s).max(0.0),
            OptionType::Call => (s - self.strike).max(0.0),
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Neural Network Module
// ═══════════════════════════════════════════════════════════════════════════════

pub mod network {
    use ndarray::{Array1, Array2};
    use rand::Rng;
    use rand_distr::Normal;

    /// Activation function types.
    #[derive(Debug, Clone, Copy)]
    pub enum Activation {
        Tanh,
        ReLU,
        Softplus,
    }

    /// Simple feedforward neural network with backpropagation.
    ///
    /// Architecture: input(2) -> hidden layers -> output(1)
    /// Input: (S, t), Output: V(S, t)
    #[derive(Debug, Clone)]
    pub struct NeuralNetwork {
        pub weights: Vec<Array2<f64>>,
        pub biases: Vec<Array1<f64>>,
        pub activation: Activation,
        layer_sizes: Vec<usize>,
    }

    impl NeuralNetwork {
        /// Create a new network with Xavier initialization.
        pub fn new(layer_sizes: &[usize], activation: Activation) -> Self {
            let mut rng = rand::thread_rng();
            let mut weights = Vec::new();
            let mut biases = Vec::new();

            for i in 0..layer_sizes.len() - 1 {
                let fan_in = layer_sizes[i];
                let fan_out = layer_sizes[i + 1];
                let std = (2.0 / (fan_in + fan_out) as f64).sqrt();
                let normal = Normal::new(0.0, std).unwrap();

                let w = Array2::from_shape_fn((fan_out, fan_in), |_| {
                    rng.sample(normal)
                });
                let b = Array1::zeros(fan_out);

                weights.push(w);
                biases.push(b);
            }

            Self {
                weights,
                biases,
                activation,
                layer_sizes: layer_sizes.to_vec(),
            }
        }

        /// Forward pass: compute V(S, t).
        pub fn forward(&self, input: &Array1<f64>) -> f64 {
            let mut x = input.clone();

            for i in 0..self.weights.len() - 1 {
                x = self.weights[i].dot(&x) + &self.biases[i];
                x = self.activate(&x);
            }

            // Output layer with softplus for non-negativity
            let last = self.weights.len() - 1;
            x = self.weights[last].dot(&x) + &self.biases[last];
            softplus(x[0])
        }

        /// Forward pass returning all intermediate activations for backprop.
        pub fn forward_with_cache(
            &self,
            input: &Array1<f64>,
        ) -> (f64, Vec<Array1<f64>>, Vec<Array1<f64>>) {
            let mut activations = vec![input.clone()];
            let mut pre_activations = Vec::new();
            let mut x = input.clone();

            for i in 0..self.weights.len() - 1 {
                let z = self.weights[i].dot(&x) + &self.biases[i];
                pre_activations.push(z.clone());
                x = self.activate(&z);
                activations.push(x.clone());
            }

            let last = self.weights.len() - 1;
            let z = self.weights[last].dot(&x) + &self.biases[last];
            pre_activations.push(z.clone());

            let output = softplus(z[0]);

            (output, activations, pre_activations)
        }

        /// Backward pass: compute gradients of loss w.r.t. parameters.
        pub fn backward(
            &self,
            d_output: f64,
            activations: &[Array1<f64>],
            pre_activations: &[Array1<f64>],
        ) -> (Vec<Array2<f64>>, Vec<Array1<f64>>) {
            let n = self.weights.len();
            let mut d_weights = Vec::with_capacity(n);
            let mut d_biases = Vec::with_capacity(n);

            // Output layer gradient (through softplus)
            let last_z = pre_activations[n - 1][0];
            let d_softplus = sigmoid(last_z);
            let mut delta = Array1::from_elem(1, d_output * d_softplus);

            for i in (0..n).rev() {
                // Gradient for weights and biases
                let a = &activations[i];
                let dw = outer(&delta, a);
                let db = delta.clone();

                d_weights.push(dw);
                d_biases.push(db);

                if i > 0 {
                    // Propagate gradient
                    let grad_through = self.weights[i].t().dot(&delta);
                    let da = self.activate_derivative(&pre_activations[i - 1]);
                    delta = &grad_through * &da;
                }
            }

            d_weights.reverse();
            d_biases.reverse();

            (d_weights, d_biases)
        }

        /// Update parameters with gradient descent.
        pub fn update(
            &mut self,
            d_weights: &[Array2<f64>],
            d_biases: &[Array1<f64>],
            learning_rate: f64,
        ) {
            for i in 0..self.weights.len() {
                self.weights[i] = &self.weights[i] - &(learning_rate * &d_weights[i]);
                self.biases[i] = &self.biases[i] - &(learning_rate * &d_biases[i]);
            }
        }

        /// Compute dV/dS via finite differences.
        pub fn dv_ds(&self, s: f64, t: f64, h: f64) -> f64 {
            let v_plus = self.forward(&Array1::from_vec(vec![s + h, t]));
            let v_minus = self.forward(&Array1::from_vec(vec![s - h, t]));
            (v_plus - v_minus) / (2.0 * h)
        }

        /// Compute d2V/dS2 via finite differences.
        pub fn d2v_ds2(&self, s: f64, t: f64, h: f64) -> f64 {
            let v_plus = self.forward(&Array1::from_vec(vec![s + h, t]));
            let v_center = self.forward(&Array1::from_vec(vec![s, t]));
            let v_minus = self.forward(&Array1::from_vec(vec![s - h, t]));
            (v_plus - 2.0 * v_center + v_minus) / (h * h)
        }

        /// Compute dV/dt via finite differences.
        pub fn dv_dt(&self, s: f64, t: f64, h: f64) -> f64 {
            let v_plus = self.forward(&Array1::from_vec(vec![s, t + h]));
            let v_minus = self.forward(&Array1::from_vec(vec![s, t - h]));
            (v_plus - v_minus) / (2.0 * h)
        }

        fn activate(&self, x: &Array1<f64>) -> Array1<f64> {
            match self.activation {
                Activation::Tanh => x.mapv(|v| v.tanh()),
                Activation::ReLU => x.mapv(|v| v.max(0.0)),
                Activation::Softplus => x.mapv(softplus),
            }
        }

        fn activate_derivative(&self, pre_activation: &Array1<f64>) -> Array1<f64> {
            match self.activation {
                Activation::Tanh => {
                    pre_activation.mapv(|v| 1.0 - v.tanh().powi(2))
                }
                Activation::ReLU => {
                    pre_activation.mapv(|v| if v > 0.0 { 1.0 } else { 0.0 })
                }
                Activation::Softplus => pre_activation.mapv(sigmoid),
            }
        }

        /// Number of trainable parameters.
        pub fn num_parameters(&self) -> usize {
            self.weights.iter().map(|w| w.len()).sum::<usize>()
                + self.biases.iter().map(|b| b.len()).sum::<usize>()
        }
    }

    fn softplus(x: f64) -> f64 {
        if x > 20.0 {
            x
        } else {
            (1.0 + x.exp()).ln()
        }
    }

    fn sigmoid(x: f64) -> f64 {
        1.0 / (1.0 + (-x).exp())
    }

    fn outer(a: &Array1<f64>, b: &Array1<f64>) -> Array2<f64> {
        let n = a.len();
        let m = b.len();
        Array2::from_shape_fn((n, m), |(i, j)| a[i] * b[j])
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn test_network_forward() {
            let net = NeuralNetwork::new(&[2, 16, 16, 1], Activation::Tanh);
            let input = Array1::from_vec(vec![100.0, 0.5]);
            let output = net.forward(&input);
            assert!(output >= 0.0, "Output should be non-negative (softplus)");
        }

        #[test]
        fn test_network_num_parameters() {
            let net = NeuralNetwork::new(&[2, 32, 32, 1], Activation::Tanh);
            let n = net.num_parameters();
            // 2*32 + 32 + 32*32 + 32 + 32*1 + 1 = 64+32+1024+32+32+1 = 1185
            assert_eq!(n, 1185);
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// PINN Pricer Module
// ═══════════════════════════════════════════════════════════════════════════════

pub mod pricer {
    use crate::network::{Activation, NeuralNetwork};
    use crate::{OptionParams, OptionType};
    use ndarray::Array1;
    use rand::Rng;

    /// PINN pricer for American options.
    pub struct AmericanPINNPricer {
        pub network: NeuralNetwork,
        pub params: OptionParams,
        pub penalty_lambda: f64,
    }

    impl AmericanPINNPricer {
        /// Create a new PINN pricer.
        pub fn new(
            params: OptionParams,
            hidden_layers: &[usize],
            penalty_lambda: f64,
        ) -> Self {
            let mut layer_sizes = vec![2]; // input: (S, t)
            layer_sizes.extend_from_slice(hidden_layers);
            layer_sizes.push(1); // output: V

            let network = NeuralNetwork::new(&layer_sizes, Activation::Tanh);

            Self {
                network,
                params,
                penalty_lambda,
            }
        }

        /// Price an option at (S, t).
        pub fn price(&self, s: f64, t: f64) -> f64 {
            let input = Array1::from_vec(vec![s / self.params.s_max, t / self.params.maturity]);
            self.network.forward(&input) * self.params.strike
        }

        /// Compute the PDE residual at point (S, t).
        ///
        /// residual = dV/dt + 0.5*sig^2*S^2*V_SS + r*S*V_S - r*V
        pub fn pde_residual(&self, s: f64, t: f64) -> f64 {
            let h_s = s * 0.001 + 0.01;
            let h_t = 0.001;

            let dv_dt = self.network.dv_dt(
                s / self.params.s_max,
                t / self.params.maturity,
                h_t / self.params.maturity,
            ) * self.params.strike / self.params.maturity;

            let dv_ds = self.network.dv_ds(
                s / self.params.s_max,
                t / self.params.maturity,
                h_s / self.params.s_max,
            ) * self.params.strike / self.params.s_max;

            let d2v_ds2 = self.network.d2v_ds2(
                s / self.params.s_max,
                t / self.params.maturity,
                h_s / self.params.s_max,
            ) * self.params.strike / (self.params.s_max * self.params.s_max);

            let v = self.price(s, t);
            let r = self.params.risk_free_rate;
            let sigma = self.params.volatility;

            dv_dt + 0.5 * sigma * sigma * s * s * d2v_ds2 + r * s * dv_ds - r * v
        }

        /// Compute total loss for a batch of collocation points.
        pub fn compute_loss(
            &self,
            s_interior: &[f64],
            t_interior: &[f64],
            s_terminal: &[f64],
            t_boundary: &[f64],
        ) -> (f64, f64, f64, f64) {
            let n_int = s_interior.len();
            let n_term = s_terminal.len();
            let n_bnd = t_boundary.len();

            // PDE residual loss
            let mut loss_pde = 0.0;
            for i in 0..n_int {
                let res = self.pde_residual(s_interior[i], t_interior[i]);
                loss_pde += res * res;
            }
            loss_pde /= n_int as f64;

            // Terminal condition loss: V(S, T) = payoff(S)
            let mut loss_terminal = 0.0;
            for i in 0..n_term {
                let v = self.price(s_terminal[i], self.params.maturity);
                let payoff = self.params.payoff(s_terminal[i]);
                let diff = v - payoff;
                loss_terminal += diff * diff;
            }
            loss_terminal /= n_term as f64;

            // Boundary condition loss
            let mut loss_boundary = 0.0;
            for i in 0..n_bnd {
                let t = t_boundary[i];
                match self.params.option_type {
                    OptionType::Put => {
                        // V(0, t) = K * exp(-r*(T-t))
                        let v_low = self.price(0.0, t);
                        let tau = self.params.maturity - t;
                        let target = self.params.strike
                            * (-self.params.risk_free_rate * tau).exp();
                        loss_boundary += (v_low - target).powi(2);

                        // V(S_max, t) -> 0
                        let v_high = self.price(self.params.s_max, t);
                        loss_boundary += v_high * v_high;
                    }
                    OptionType::Call => {
                        // V(0, t) = 0
                        let v_low = self.price(0.0, t);
                        loss_boundary += v_low * v_low;

                        // V(S_max, t) ~ S_max - K*exp(-r*(T-t))
                        let v_high = self.price(self.params.s_max, t);
                        let tau = self.params.maturity - t;
                        let target = self.params.s_max
                            - self.params.strike
                                * (-self.params.risk_free_rate * tau).exp();
                        loss_boundary += (v_high - target).powi(2);
                    }
                }
            }
            loss_boundary /= (2 * n_bnd) as f64;

            // Early exercise penalty
            let mut loss_penalty = 0.0;
            for i in 0..n_int {
                let v = self.price(s_interior[i], t_interior[i]);
                let payoff = self.params.payoff(s_interior[i]);
                let violation = (payoff - v).max(0.0);
                loss_penalty += violation * violation;
            }
            loss_penalty *= self.penalty_lambda / n_int as f64;

            (loss_pde, loss_terminal, loss_boundary, loss_penalty)
        }

        /// Train the PINN using stochastic gradient descent.
        pub fn train(
            &mut self,
            n_epochs: usize,
            learning_rate: f64,
            n_interior: usize,
            n_boundary: usize,
            n_terminal: usize,
            verbose: bool,
        ) -> Vec<f64> {
            let mut rng = rand::thread_rng();
            let mut history = Vec::with_capacity(n_epochs);

            let w_pde = 1.0;
            let w_terminal = 10.0;
            let w_boundary = 5.0;
            let w_penalty = 1.0;

            for epoch in 0..n_epochs {
                // Update penalty schedule
                self.penalty_lambda = match epoch {
                    0..=499 => 100.0,
                    500..=1499 => 500.0,
                    1500..=2999 => 1000.0,
                    _ => 5000.0,
                };

                // Generate random collocation points
                let s_int: Vec<f64> = (0..n_interior)
                    .map(|_| rng.gen::<f64>() * self.params.s_max)
                    .collect();
                let t_int: Vec<f64> = (0..n_interior)
                    .map(|_| rng.gen::<f64>() * self.params.maturity)
                    .collect();
                let s_term: Vec<f64> = (0..n_terminal)
                    .map(|_| rng.gen::<f64>() * self.params.s_max)
                    .collect();
                let t_bnd: Vec<f64> = (0..n_boundary)
                    .map(|_| rng.gen::<f64>() * self.params.maturity)
                    .collect();

                // Compute loss
                let (l_pde, l_term, l_bnd, l_pen) =
                    self.compute_loss(&s_int, &t_int, &s_term, &t_bnd);

                let total_loss =
                    w_pde * l_pde + w_terminal * l_term + w_boundary * l_bnd + w_penalty * l_pen;

                history.push(total_loss);

                // Numerical gradient and update
                // Using simple parameter perturbation for gradient estimation
                let lr = learning_rate / (1.0 + epoch as f64 * 0.0001);
                self.numerical_gradient_step(&s_int, &t_int, &s_term, &t_bnd, lr);

                if verbose && epoch % 500 == 0 {
                    println!(
                        "Epoch {:5} | Loss: {:.6} | PDE: {:.6} | Term: {:.6} | BC: {:.6} | Pen: {:.6}",
                        epoch, total_loss, l_pde, l_term, l_bnd, l_pen
                    );
                }
            }

            history
        }

        /// Single gradient step using parameter perturbation.
        fn numerical_gradient_step(
            &mut self,
            s_int: &[f64],
            t_int: &[f64],
            s_term: &[f64],
            t_bnd: &[f64],
            lr: f64,
        ) {
            let epsilon = 1e-4;
            let base_loss = self.total_loss(s_int, t_int, s_term, t_bnd);

            // Update weights
            for layer in 0..self.network.weights.len() {
                let (rows, cols) = self.network.weights[layer].dim();
                for r in 0..rows {
                    for c in 0..cols {
                        self.network.weights[layer][[r, c]] += epsilon;
                        let loss_plus = self.total_loss(s_int, t_int, s_term, t_bnd);
                        self.network.weights[layer][[r, c]] -= epsilon;

                        let grad = (loss_plus - base_loss) / epsilon;
                        self.network.weights[layer][[r, c]] -= lr * grad;
                    }
                }

                // Update biases
                let n = self.network.biases[layer].len();
                for j in 0..n {
                    self.network.biases[layer][j] += epsilon;
                    let loss_plus = self.total_loss(s_int, t_int, s_term, t_bnd);
                    self.network.biases[layer][j] -= epsilon;

                    let grad = (loss_plus - base_loss) / epsilon;
                    self.network.biases[layer][j] -= lr * grad;
                }
            }
        }

        fn total_loss(
            &self,
            s_int: &[f64],
            t_int: &[f64],
            s_term: &[f64],
            t_bnd: &[f64],
        ) -> f64 {
            let (l_pde, l_term, l_bnd, l_pen) =
                self.compute_loss(s_int, t_int, s_term, t_bnd);
            l_pde + 10.0 * l_term + 5.0 * l_bnd + l_pen
        }

        /// Find the exercise boundary S*(t).
        pub fn find_exercise_boundary(
            &self,
            t_values: &[f64],
            n_s_points: usize,
        ) -> Vec<f64> {
            let mut boundary = Vec::with_capacity(t_values.len());

            for &t in t_values {
                let mut best_s = self.params.strike;
                let mut best_diff = f64::MAX;

                for i in 0..n_s_points {
                    let s = (i as f64 / n_s_points as f64) * self.params.s_max;
                    let v = self.price(s, t);
                    let payoff = self.params.payoff(s);

                    if payoff > 0.01 {
                        let diff = (v - payoff).abs();
                        if diff < best_diff {
                            best_diff = diff;
                            best_s = s;
                        }
                    }
                }

                boundary.push(best_s);
            }

            boundary
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn test_pricer_creation() {
            let params = OptionParams::default();
            let pricer = AmericanPINNPricer::new(params, &[16, 16], 1000.0);
            let v = pricer.price(100.0, 0.5);
            assert!(v >= 0.0, "Price should be non-negative");
        }

        #[test]
        fn test_payoff_constraint() {
            let params = OptionParams {
                strike: 100.0,
                option_type: OptionType::Put,
                ..Default::default()
            };
            let payoff = params.payoff(80.0);
            assert!((payoff - 20.0).abs() < 1e-10);

            let payoff_otm = params.payoff(120.0);
            assert!(payoff_otm.abs() < 1e-10);
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Longstaff-Schwartz Module
// ═══════════════════════════════════════════════════════════════════════════════

pub mod lsm {
    use crate::{OptionParams, OptionType};
    use rand::Rng;
    use rand_distr::StandardNormal;

    /// Result from LSM pricing.
    #[derive(Debug, Clone)]
    pub struct LsmResult {
        pub price: f64,
        pub std_error: f64,
        pub exercise_boundary: Vec<f64>,
    }

    /// Price an American option using Longstaff-Schwartz Monte Carlo.
    pub fn lsm_american_option(
        s0: f64,
        params: &OptionParams,
        n_steps: usize,
        n_paths: usize,
    ) -> LsmResult {
        let mut rng = rand::thread_rng();
        let dt = params.maturity / n_steps as f64;
        let discount = (-params.risk_free_rate * dt).exp();

        // Simulate GBM paths
        let mut paths = vec![vec![0.0f64; n_paths]; n_steps + 1];
        for j in 0..n_paths {
            paths[0][j] = s0;
        }

        for i in 1..=n_steps {
            for j in 0..n_paths {
                let z: f64 = rng.sample(StandardNormal);
                paths[i][j] = paths[i - 1][j]
                    * ((params.risk_free_rate - 0.5 * params.volatility.powi(2)) * dt
                        + params.volatility * dt.sqrt() * z)
                        .exp();
            }
        }

        // Payoff function
        let payoff = |s: f64| -> f64 { params.payoff(s) };

        // Initialize cash flows at maturity
        let mut cashflows: Vec<f64> = paths[n_steps].iter().map(|&s| payoff(s)).collect();
        let mut exercise_time: Vec<usize> = vec![n_steps; n_paths];
        let mut boundary = vec![f64::NAN; n_steps];

        // Backward induction
        for t in (1..n_steps).rev() {
            let s_t: Vec<f64> = paths[t].clone();
            let intrinsic: Vec<f64> = s_t.iter().map(|&s| payoff(s)).collect();

            // In-the-money indices
            let itm_indices: Vec<usize> = (0..n_paths)
                .filter(|&j| intrinsic[j] > 0.0)
                .collect();

            if itm_indices.len() < 4 {
                continue;
            }

            // Compute continuation values for ITM paths
            let s_itm: Vec<f64> = itm_indices.iter().map(|&j| s_t[j]).collect();
            let y_itm: Vec<f64> = itm_indices
                .iter()
                .map(|&j| {
                    let steps_ahead = exercise_time[j] - t;
                    cashflows[j] * discount.powi(steps_ahead as i32)
                })
                .collect();

            // Polynomial regression (degree 3)
            if let Some(coeffs) = polynomial_regression(&s_itm, &y_itm, 3) {
                let mut max_exercised = f64::NEG_INFINITY;
                let mut min_exercised = f64::INFINITY;

                for &j in &itm_indices {
                    let continuation = eval_polynomial(s_t[j], &coeffs);
                    if intrinsic[j] > continuation {
                        cashflows[j] = intrinsic[j];
                        exercise_time[j] = t;

                        match params.option_type {
                            OptionType::Put => {
                                if s_t[j] > max_exercised {
                                    max_exercised = s_t[j];
                                }
                            }
                            OptionType::Call => {
                                if s_t[j] < min_exercised {
                                    min_exercised = s_t[j];
                                }
                            }
                        }
                    }
                }

                boundary[t] = match params.option_type {
                    OptionType::Put => {
                        if max_exercised.is_finite() {
                            max_exercised
                        } else {
                            f64::NAN
                        }
                    }
                    OptionType::Call => {
                        if min_exercised.is_finite() {
                            min_exercised
                        } else {
                            f64::NAN
                        }
                    }
                };
            }
        }

        // Discount to time 0
        let option_values: Vec<f64> = (0..n_paths)
            .map(|j| cashflows[j] * discount.powi(exercise_time[j] as i32))
            .collect();

        let mean = option_values.iter().sum::<f64>() / n_paths as f64;
        let variance = option_values
            .iter()
            .map(|&v| (v - mean).powi(2))
            .sum::<f64>()
            / (n_paths - 1) as f64;
        let std_error = (variance / n_paths as f64).sqrt();

        // Compare with immediate exercise
        let price = mean.max(payoff(s0));

        LsmResult {
            price,
            std_error,
            exercise_boundary: boundary,
        }
    }

    /// Simple polynomial regression.
    fn polynomial_regression(x: &[f64], y: &[f64], degree: usize) -> Option<Vec<f64>> {
        let n = x.len();
        if n < degree + 1 {
            return None;
        }

        // Build Vandermonde matrix
        let cols = degree + 1;
        let mut xtx = vec![0.0f64; cols * cols];
        let mut xty = vec![0.0f64; cols];

        for i in 0..n {
            let mut xi_powers = vec![1.0f64; cols];
            for k in 1..cols {
                xi_powers[k] = xi_powers[k - 1] * x[i];
            }

            for r in 0..cols {
                for c in 0..cols {
                    xtx[r * cols + c] += xi_powers[r] * xi_powers[c];
                }
                xty[r] += xi_powers[r] * y[i];
            }
        }

        // Solve using Gaussian elimination
        solve_linear_system(&xtx, &xty, cols)
    }

    fn eval_polynomial(x: f64, coeffs: &[f64]) -> f64 {
        let mut result = 0.0;
        let mut xi = 1.0;
        for &c in coeffs {
            result += c * xi;
            xi *= x;
        }
        result
    }

    fn solve_linear_system(a: &[f64], b: &[f64], n: usize) -> Option<Vec<f64>> {
        let mut aug = vec![0.0f64; n * (n + 1)];
        for i in 0..n {
            for j in 0..n {
                aug[i * (n + 1) + j] = a[i * n + j];
            }
            aug[i * (n + 1) + n] = b[i];
        }

        // Forward elimination
        for col in 0..n {
            let mut max_row = col;
            let mut max_val = aug[col * (n + 1) + col].abs();
            for row in (col + 1)..n {
                let val = aug[row * (n + 1) + col].abs();
                if val > max_val {
                    max_val = val;
                    max_row = row;
                }
            }

            if max_val < 1e-12 {
                return None;
            }

            // Swap rows
            if max_row != col {
                for j in 0..=n {
                    let tmp = aug[col * (n + 1) + j];
                    aug[col * (n + 1) + j] = aug[max_row * (n + 1) + j];
                    aug[max_row * (n + 1) + j] = tmp;
                }
            }

            let pivot = aug[col * (n + 1) + col];
            for row in (col + 1)..n {
                let factor = aug[row * (n + 1) + col] / pivot;
                for j in col..=n {
                    aug[row * (n + 1) + j] -= factor * aug[col * (n + 1) + j];
                }
            }
        }

        // Back substitution
        let mut x = vec![0.0; n];
        for i in (0..n).rev() {
            let mut sum = aug[i * (n + 1) + n];
            for j in (i + 1)..n {
                sum -= aug[i * (n + 1) + j] * x[j];
            }
            x[i] = sum / aug[i * (n + 1) + i];
        }

        Some(x)
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn test_lsm_put() {
            let params = OptionParams {
                strike: 100.0,
                risk_free_rate: 0.05,
                volatility: 0.2,
                maturity: 1.0,
                option_type: OptionType::Put,
                s_max: 300.0,
            };

            let result = lsm_american_option(100.0, &params, 50, 10000);
            // American put with these params should be around 6-8
            assert!(result.price > 3.0 && result.price < 15.0,
                    "LSM put price {} out of expected range", result.price);
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Greeks Module
// ═══════════════════════════════════════════════════════════════════════════════

pub mod greeks {
    use crate::pricer::AmericanPINNPricer;

    /// Greeks values at a point.
    #[derive(Debug, Clone)]
    pub struct Greeks {
        pub delta: f64,
        pub gamma: f64,
        pub theta: f64,
        pub vega: f64,
        pub rho: f64,
    }

    /// Compute option Greeks via finite differences.
    pub struct GreeksComputer;

    impl GreeksComputer {
        /// Compute all Greeks at (S, t).
        pub fn compute(pricer: &AmericanPINNPricer, s: f64, t: f64) -> Greeks {
            let h_s = s * 0.01 + 0.01;
            let h_t = 0.001;

            // Delta = dV/dS
            let v_up = pricer.price(s + h_s, t);
            let v_down = pricer.price(s - h_s, t);
            let delta = (v_up - v_down) / (2.0 * h_s);

            // Gamma = d2V/dS2
            let v_center = pricer.price(s, t);
            let gamma = (v_up - 2.0 * v_center + v_down) / (h_s * h_s);

            // Theta = dV/dt
            let v_t_up = pricer.price(s, (t + h_t).min(pricer.params.maturity));
            let v_t_down = pricer.price(s, (t - h_t).max(0.0));
            let theta = (v_t_up - v_t_down) / (2.0 * h_t);

            // Vega and Rho are approximations (would need retraining for exact)
            let tau = pricer.params.maturity - t;
            let d1 = if tau > 0.0 {
                ((s / pricer.params.strike).ln()
                    + (pricer.params.risk_free_rate
                        + 0.5 * pricer.params.volatility.powi(2))
                        * tau)
                    / (pricer.params.volatility * tau.sqrt())
            } else {
                0.0
            };

            let phi_d1 = (-0.5 * d1 * d1).exp() / (2.0 * std::f64::consts::PI).sqrt();
            let vega = s * tau.sqrt() * phi_d1 * 0.01; // per 1% vol move

            let d2 = d1 - pricer.params.volatility * tau.sqrt();
            let n_d2 = 0.5 * (1.0 + erf(d2 / std::f64::consts::SQRT_2));
            let rho = match pricer.params.option_type {
                crate::OptionType::Put => {
                    -pricer.params.strike * tau
                        * (-pricer.params.risk_free_rate * tau).exp()
                        * (1.0 - n_d2)
                        * 0.01
                }
                crate::OptionType::Call => {
                    pricer.params.strike * tau
                        * (-pricer.params.risk_free_rate * tau).exp()
                        * n_d2
                        * 0.01
                }
            };

            Greeks {
                delta,
                gamma,
                theta,
                vega,
                rho,
            }
        }
    }

    fn erf(x: f64) -> f64 {
        // Abramowitz and Stegun approximation
        let a1 = 0.254829592;
        let a2 = -0.284496736;
        let a3 = 1.421413741;
        let a4 = -1.453152027;
        let a5 = 1.061405429;
        let p = 0.3275911;

        let sign = if x >= 0.0 { 1.0 } else { -1.0 };
        let x = x.abs();
        let t = 1.0 / (1.0 + p * x);
        let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-x * x).exp();

        sign * y
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        use crate::pricer::AmericanPINNPricer;
        use crate::OptionParams;

        #[test]
        fn test_greeks_computation() {
            let params = OptionParams::default();
            let pricer = AmericanPINNPricer::new(params, &[16, 16], 1000.0);

            let greeks = GreeksComputer::compute(&pricer, 100.0, 0.0);

            // Delta for put should be negative
            // (untrained network may not have correct sign, just check it is finite)
            assert!(greeks.delta.is_finite(), "Delta should be finite");
            assert!(greeks.gamma.is_finite(), "Gamma should be finite");
            assert!(greeks.theta.is_finite(), "Theta should be finite");
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Data Module (Bybit API)
// ═══════════════════════════════════════════════════════════════════════════════

pub mod data {
    use serde::Deserialize;

    /// Bybit API client for fetching crypto market data.
    pub struct BybitClient {
        base_url: String,
    }

    #[derive(Debug, Deserialize)]
    struct BybitKlineResponse {
        #[serde(rename = "retCode")]
        ret_code: i32,
        result: BybitKlineResult,
    }

    #[derive(Debug, Deserialize)]
    struct BybitKlineResult {
        list: Vec<Vec<String>>,
    }

    /// OHLCV candle data.
    #[derive(Debug, Clone)]
    pub struct Candle {
        pub timestamp: u64,
        pub open: f64,
        pub high: f64,
        pub low: f64,
        pub close: f64,
        pub volume: f64,
    }

    /// Market data summary.
    #[derive(Debug, Clone)]
    pub struct MarketData {
        pub symbol: String,
        pub candles: Vec<Candle>,
        pub current_price: f64,
        pub volatility: f64,
    }

    impl BybitClient {
        pub fn new() -> Self {
            Self {
                base_url: "https://api.bybit.com".to_string(),
            }
        }

        /// Fetch historical kline data from Bybit.
        pub async fn fetch_klines(
            &self,
            symbol: &str,
            interval: &str,
            limit: usize,
        ) -> anyhow::Result<Vec<Candle>> {
            let url = format!(
                "{}/v5/market/kline?category=linear&symbol={}&interval={}&limit={}",
                self.base_url, symbol, interval, limit
            );

            let client = reqwest::Client::new();
            let response: BybitKlineResponse = client
                .get(&url)
                .send()
                .await?
                .json()
                .await?;

            if response.ret_code != 0 {
                anyhow::bail!("Bybit API error code: {}", response.ret_code);
            }

            let candles = response
                .result
                .list
                .iter()
                .filter_map(|row| {
                    if row.len() >= 6 {
                        Some(Candle {
                            timestamp: row[0].parse().unwrap_or(0),
                            open: row[1].parse().unwrap_or(0.0),
                            high: row[2].parse().unwrap_or(0.0),
                            low: row[3].parse().unwrap_or(0.0),
                            close: row[4].parse().unwrap_or(0.0),
                            volume: row[5].parse().unwrap_or(0.0),
                        })
                    } else {
                        None
                    }
                })
                .collect();

            Ok(candles)
        }

        /// Fetch market data with computed statistics.
        pub async fn fetch_market_data(
            &self,
            symbol: &str,
            interval: &str,
            limit: usize,
        ) -> anyhow::Result<MarketData> {
            let mut candles = self.fetch_klines(symbol, interval, limit).await?;

            if candles.is_empty() {
                anyhow::bail!("No data received for {}", symbol);
            }

            // Sort by timestamp ascending
            candles.sort_by_key(|c| c.timestamp);

            let current_price = candles.last().unwrap().close;

            // Compute historical volatility
            let log_returns: Vec<f64> = candles
                .windows(2)
                .map(|w| (w[1].close / w[0].close).ln())
                .collect();

            let mean_return = log_returns.iter().sum::<f64>() / log_returns.len() as f64;
            let variance = log_returns
                .iter()
                .map(|r| (r - mean_return).powi(2))
                .sum::<f64>()
                / (log_returns.len() - 1) as f64;
            let daily_vol = variance.sqrt();

            // Annualize (365 for crypto)
            let volatility = daily_vol * (365.0_f64).sqrt();

            Ok(MarketData {
                symbol: symbol.to_string(),
                candles,
                current_price,
                volatility,
            })
        }
    }

    impl Default for BybitClient {
        fn default() -> Self {
            Self::new()
        }
    }

    /// Generate synthetic GBM data for testing.
    pub fn generate_synthetic_data(
        s0: f64,
        mu: f64,
        sigma: f64,
        n_days: usize,
    ) -> Vec<f64> {
        use rand::Rng;
        use rand_distr::StandardNormal;

        let mut rng = rand::thread_rng();
        let dt = 1.0 / 252.0;
        let mut prices = vec![s0];

        for _ in 1..n_days {
            let z: f64 = rng.sample(StandardNormal);
            let prev = *prices.last().unwrap();
            let next = prev * ((mu - 0.5 * sigma * sigma) * dt + sigma * dt.sqrt() * z).exp();
            prices.push(next);
        }

        prices
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Backtest Module
// ═══════════════════════════════════════════════════════════════════════════════

pub mod backtest {
    use crate::pricer::AmericanPINNPricer;

    /// Single trade record.
    #[derive(Debug, Clone)]
    pub struct Trade {
        pub entry_idx: usize,
        pub exit_idx: usize,
        pub entry_spot: f64,
        pub exit_spot: f64,
        pub direction: String,
        pub pnl: f64,
    }

    /// Backtest results.
    #[derive(Debug, Clone)]
    pub struct BacktestResult {
        pub total_pnl: f64,
        pub n_trades: usize,
        pub win_rate: f64,
        pub sharpe_ratio: f64,
        pub max_drawdown: f64,
        pub trades: Vec<Trade>,
    }

    /// Run a mispricing-based backtest.
    pub fn run_backtest(
        pricer: &AmericanPINNPricer,
        spot_prices: &[f64],
        entry_threshold: f64,
        exit_threshold: f64,
        max_holding: usize,
    ) -> BacktestResult {
        let n = spot_prices.len();
        let maturity = pricer.params.maturity;
        let mut trades = Vec::new();

        // Compute PINN prices
        let time_vals: Vec<f64> = (0..n)
            .map(|i| (i as f64 / n as f64) * maturity * 0.9)
            .collect();

        let pinn_prices: Vec<f64> = (0..n)
            .map(|i| pricer.price(spot_prices[i], time_vals[i]))
            .collect();

        // Simulate noisy market prices
        let mut rng = rand::thread_rng();
        use rand::Rng;
        let market_prices: Vec<f64> = pinn_prices
            .iter()
            .map(|&p| (p * (1.0 + rng.gen_range(-0.03..0.03))).max(0.0))
            .collect();

        let mut active_trade: Option<(usize, f64, f64, String)> = None;

        for i in 0..n {
            let pinn_v = pinn_prices[i];
            let mkt_v = market_prices[i];

            if pinn_v < 1e-6 && mkt_v < 1e-6 {
                continue;
            }

            let mispricing = (pinn_v - mkt_v) / pinn_v.max(1e-8);

            match &active_trade {
                None => {
                    if mispricing.abs() > entry_threshold {
                        let direction = if mispricing > 0.0 {
                            "long".to_string()
                        } else {
                            "short".to_string()
                        };
                        active_trade = Some((i, spot_prices[i], mkt_v, direction));
                    }
                }
                Some((entry_idx, entry_spot, entry_price, direction)) => {
                    let holding = i - entry_idx;
                    let should_exit =
                        mispricing.abs() < exit_threshold || holding >= max_holding || i == n - 1;

                    if should_exit {
                        let pnl = if direction == "long" {
                            mkt_v - entry_price
                        } else {
                            entry_price - mkt_v
                        };

                        trades.push(Trade {
                            entry_idx: *entry_idx,
                            exit_idx: i,
                            entry_spot: *entry_spot,
                            exit_spot: spot_prices[i],
                            direction: direction.clone(),
                            pnl,
                        });
                        active_trade = None;
                    }
                }
            }
        }

        // Statistics
        let pnls: Vec<f64> = trades.iter().map(|t| t.pnl).collect();
        let total_pnl: f64 = pnls.iter().sum();
        let n_trades = trades.len();
        let win_rate = if n_trades > 0 {
            pnls.iter().filter(|&&p| p > 0.0).count() as f64 / n_trades as f64
        } else {
            0.0
        };

        let sharpe = if n_trades > 1 {
            let mean = total_pnl / n_trades as f64;
            let var = pnls.iter().map(|p| (p - mean).powi(2)).sum::<f64>() / (n_trades - 1) as f64;
            if var > 0.0 {
                mean / var.sqrt() * (252.0_f64).sqrt()
            } else {
                0.0
            }
        } else {
            0.0
        };

        let max_dd = if !pnls.is_empty() {
            let cum: Vec<f64> = pnls
                .iter()
                .scan(0.0, |state, &p| {
                    *state += p;
                    Some(*state)
                })
                .collect();
            let mut peak = f64::NEG_INFINITY;
            let mut max_drawdown = 0.0f64;
            for &c in &cum {
                peak = peak.max(c);
                max_drawdown = max_drawdown.max(peak - c);
            }
            max_drawdown
        } else {
            0.0
        };

        BacktestResult {
            total_pnl,
            n_trades,
            win_rate,
            sharpe_ratio: sharpe,
            max_drawdown: max_dd,
            trades,
        }
    }
}
