# Research Proposal & Novelty Blueprint: ST-AGBDT
## Self-Throttling Asymmetric Gradient Boosted Decision Trees for Carbon-Aware Proactive Edge Orchestration

---

## 1. Executive Summary & Literature Landscape (2024–2026)

### 1.1 The State of the Art (SOTA)
In contemporary edge computing and sustainable AI research (*IEEE SEC, ACM IoTDI, IEEE GreenCom, MLSys*), researchers are addressing the trade-off between **Service Level Objectives (SLOs)** and **energy/carbon efficiency**:
1. **Heavy Deep Learning Schedulers (e.g., CarbonLiteFormer, DRL/DQN, CNN-LSTM):**  
   Most modern proactive autoscalers use deep neural networks. While they achieve high predictive accuracy, they suffer from the **"Edge AI Paradox"**: loading PyTorch/TensorFlow runtimes (200MB–1GB RAM) and executing multi-layer tensors consumes massive CPU/GPU wattage and causes non-deterministic inference latency, defeating the purpose of energy-saving on resource-constrained micro-edge nodes.
2. **Cloud-Based Asymmetric Estimators (e.g., XGBoost with Custom Objectives):**  
   Some cloud architectures use asymmetric loss functions (asymmetric least squares / expectile loss or quantile loss) to avoid Out-Of-Memory (OOM) errors in massive data centers. However, these are static, cloud-tethered models without real-time edge awareness, without runtime self-throttling, and dependent on heavy C++/Python packages.
3. **Reactive Edge Autoscaling (e.g., Kubernetes HPA, Static Thresholds):**  
   Traditional reactive edge controllers wait until an SLO threshold is breached before spinning up cores/workers, causing severe cold-start latency spikes during burst traffic.

---

## 2. The Research Gap: The Unsolved Edge Paradox

| Existing Approach | Proactive Lead Time | Edge RAM Footprint | Inference Energy | Risk/SLO Awareness | Hardware Self-Throttling |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Reactive HPA** | 0s (Too late) | Low (< 5MB) | Zero | None (Symmetric) | None |
| **Deep Learning (LSTM/Transformer)** | 30s | Very High (> 300MB) | High (100–300 mJ) | Moderate | None |
| **Standard Cloud GBDT (XGBoost/LightGBM)**| 30s | Moderate (~50MB) | Moderate (15–30 mJ) | SOTA (Loss tuning) | None (Evaluates 100% trees) |
| **PERM (Proposed ST-AGBDT)** | **TARGET - NOT MEASURED** *(measured: -7.8s to -31.1s in `step2_summary.json`)* | **145.7 KB lean / 326.6 KB full** *(measured in `forecaster_eval_results.json`)* | **TARGET - NOT MEASURED** *(target 0.8–2.5 mJ; hardware power not instrumented)* | **Asymmetric Least Squares (Expectile, $\alpha=0.85$)** | **TARGET - NOT MEASURED** *(proposal targeted 80% early exit; measured at W=8: 18–56% exit, saving 11–26% trees at 1–5% extra misses in `forecaster_eval_results.json`)* |

---

## 3. What is the Proposed Approach? (ST-AGBDT)

The proposed approach is a **co-designed hardware-software TinyML framework** composed of four tightly integrated, mathematically novel mechanisms:

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                                 PERM-Green Architecture                                │
│                                                                                        │
│  1. Telemetry Collector (RAPL / Energy Tracking / psutil)                              │
│     └── Samples: [CPU%, Memory%, Latency, Request Rate, Joules Consumed]               │
│                                │                                                       │
│                                ▼                                                       │
│  2. ST-AGBDT Inference Engine (Self-Throttling Asymmetric GBDT)                        │
│     ├── Stage A: Fast Guard (Trees 1..K_0) ──> Safe? ──> Early Exit                    │
│     │   (TARGET - NOT MEASURED: 82% joules target; measured 11–26% trees at W=8)       │
│     └── Stage B: Deep Boost (Trees K_0..M) ──> Volatile? ──> Asymmetric Directional   │
│                                              Least Squares (Expectile) Split           │
│                                │                                                       │
│                                ▼                                                       │
│  3. Closed-Loop Pareto Energy-SLO Governor                                             │
│     └── Solves: min_k [ λ · (SLO_Breach_Penalty) + (1-λ) · Hardware_Power(k) ]         │
│                                │                                                       │
│                                ▼                                                       │
│  4. Dynamic Actuator (Worker Core Provisioning & Frequency Scaling)                    │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 4. Detailed Mathematical & Algorithmic Formulation

### 4.1 Novelty 1: Directional Asymmetric Least Squares (Expectile) Split Criterion
Standard decision tree regressors minimize symmetric mean squared error:
$$\mathcal{L}_{\text{MSE}}(y, \hat{y}) = (y - \hat{y})^2$$

In edge computing systems, **under-predicting workload ($y > \hat{y}$)** causes severe SLO breaches and degraded user experience, whereas **over-predicting ($y < \hat{y}$)** only causes a minor, temporary allocation of edge compute. 

We derive an exact **Asymmetric Least Squares (Expectile) Loss**:
$$\mathcal{L}_{\alpha}(y, \hat{y}) = \begin{cases} 
\alpha \cdot (y - \hat{y})^2 & \text{if } y \ge \hat{y} \quad (\text{Under-prediction: Heavy penalty, } \alpha = 0.85) \\ 
(1-\alpha) \cdot (y - \hat{y})^2 & \text{if } y < \hat{y} \quad (\text{Over-prediction: Low penalty, } 1-\alpha = 0.15) 
\end{cases}$$

For any tree split candidate, we analytically derive the 1st-order gradient $g_i$ and 2nd-order hessian $h_i$:
$$g_i = \frac{\partial \mathcal{L}_{\alpha}}{\partial \hat{y}_i} = \begin{cases} -2\alpha(y_i - \hat{y}_i) & \text{if } y_i \ge \hat{y}_i \\ -2(1-\alpha)(y_i - \hat{y}_i) & \text{if } y_i < \hat{y}_i \end{cases}$$
$$h_i = \frac{\partial^2 \mathcal{L}_{\alpha}}{\partial \hat{y}_i^2} = \begin{cases} 2\alpha & \text{if } y_i \ge \hat{y}_i \\ 2(1-\alpha) & \text{if } y_i < \hat{y}_i \end{cases}$$

The optimal leaf weight $w^*$ and split gain are calculated in closed form:
$$w^* = -\frac{\sum_{i \in \text{Leaf}} g_i}{\sum_{i \in \text{Leaf}} h_i + \lambda_{\text{reg}}}$$
$$\text{Gain} = \frac{1}{2} \left[ \frac{(\sum_{i \in L} g_i)^2}{\sum_{i \in L} h_i + \lambda} + \frac{(\sum_{i \in R} g_i)^2}{\sum_{i \in R} h_i + \lambda} - \frac{(\sum_{i \in \text{Parent}} g_i)^2}{\sum_{i \in \text{Parent}} h_i + \lambda} \right]$$

---

### 4.2 Novelty 2: Hardware-Aware Dynamic Early-Exit Tree Inference
Traditional tree ensembles evaluate all $M$ trees (e.g., $M = 50$) for every inference tick.

We formulate an **Adaptive Energy Gating Rule**:
After evaluating the base guard ensemble of $K_0$ trees (e.g., $K_0 \in \{10, 20, 30\}$):
1. **Safety Margin Check:** The intermediate prediction is below the critical threshold with confidence:
   $$\hat{y}_{K_0} + \text{margin} < \text{threshold}$$
2. **Convergence Stability Check:** The moving absolute change across the last 5 trees satisfies:
   $$\sum_{k=K_0-4}^{K_0} |\Delta \hat{y}_k| < \epsilon$$

If both conditions hold, inference **exits immediately**, skipping remaining trees.
* **Status:** **TARGET - NOT MEASURED**: Proposal targeted 80–85% cycle saving. Offline benchmark on burst-heavy demand traces measured mean evaluation of 37–45 of 50 trees at $W=8$ with 1–5% extra misses (`forecaster_eval_results.json`). At $W=2$, demand is predominantly above the 11.11 req/s threshold so 0% exit occurred.

---

### 4.3 Novelty 3: Zero-Allocation 1D Flat-Buffer Execution
Instead of recursive Python object traversal (`node.left`, `node.right`), trees can be compiled into 1D contiguous flat arrays or stacked 2D arrays:
$$\text{Child Index} = 2 \cdot i + 1 + \mathbb{I}(X[f_i] > \theta_i)$$
* **Status:** Measured offline in `forecaster_eval_results.json`: pure recursive node traversal runs in 67.8 $\mu$s per sample in Python; stacked 2D array evaluation runs in 117.6 $\mu$s (outperforming sequential flat arrays at 253.1 $\mu$s). Vectorized batch prediction reaches 90,159 samples/sec.

---

### 4.4 Novelty 4: Closed-Loop Multi-Objective Pareto Actuator
When an alert is issued with forecast $\hat{y}_{t+30}$, the remediation service selects the optimal core count $k^* \in [k_{\min}, k_{\max}]$ by solving:
$$k^* = \arg\min_{k} \left[ \omega \cdot \max(0, \hat{y}_{t+30}(k) - \text{SLO}_{\text{target}})^2 + (1-\omega) \cdot \mathcal{P}_{\text{hardware}}(k) \right]$$
where $\mathcal{P}_{\text{hardware}}(k)$ is the calibrated empirical power curve of the edge processor.
* **Status:** **TARGET - NOT MEASURED**: Power curve calibration requires physical energy sensors (RAPL/wattmeter) not yet instrumented on host testbed.

---

## 5. Summary of Publication Artifacts & Target Venues

### Target Venues:
- **Conferences:** IEEE SEC (Symposium on Edge Computing), ACM IoTDI, IEEE GreenCom, ACM/IEEE SEC, IEEE Cloud.
- **Journals:** IEEE Transactions on Sustainable Computing (TSUSC), IEEE Transactions on Green Communications and Networking (TGCN), Elsevier Sustainable Computing (SUSCOM).

### Key Empirical Figures for the Paper:
1. **Figure 1:** Asymmetric Least Squares (Expectile) vs. Symmetric MSE Pareto frontier (SLO Violations vs. Over-allocation Extra Workers).
2. **Figure 2:** Dynamic Early-Exit Breakdown (Trees evaluated and miss rate trade-offs under steady vs burst phases).
3. **Figure 3:** End-to-end Proactive Lead-Time vs. Reactive Cold-Start Timeline under Locust surge traffic.
4. **Table 1:** Comprehensive Comparison against Sklearn, PyTorch LSTM, and Reactive Kubernetes HPA.
