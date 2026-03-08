---
layout: post
title: "Delta Hedging European Short Call Option"
date: 2026-03-05 
categories: [Random posts on mathematical finance]
---

>In this post, let's replicate and hedge short call option when interest rate is assumed to be 0.

Let $$S_t$$ follow Black-Scholes with zero rate $$r=0$$ as

$$dS_t=\mu S_t\,dt+\sigma S_t\,dW_t,$$

Let $$C(t,S)$$ be price of European call option with payoff $$(S_T-K)^+$$. 

With $$r=0$$, Black-Scholes PDE is:

$$C_t+\displaystyle\frac{1}{2}\sigma^2 S^2 C_{SS}=0,\qquad C(T,S)=(S-K)^+.$$

You are a short one call, so your option position value is $$-C(t,S_t)$$

Hold $$\Delta_t$$ shares of stock. Define the self-financing hedged portfolio:

$$
dB_t = 0 \quad \text{(cash earns nothing, absent rebalancing transfers)}.
$$

Let examine the portfolio:

$$
V_t = \Delta_t S_t + B_t - C(t, S_t).
$$

At $$t=0$$, the portfolio costs 0

$$
V_0 = \Delta_0 S_0+B_0-C(0,S_0) = 0
$$

so 

$$
B_0 = C(0,S_0) - \Delta_0 S_0
$$

A strategy is self-financing if after time 0, no external money is injected or withdrawn, with $$r=0$$ we have

$$
d(\Delta_t S_t+B_t)=\Delta_t\,dS_t.
$$

$$
dV_t=d(\Delta_t S_t+B_t)-dC_t=\Delta_t\,dS_t-dC_t.
$$

## Continuous setting
Choose

$$
\Delta_t=C_S(t,S_t).
$$

Apply Ito to $$C(t,S_t)$$

$$
dC_t=C_t\,dt+C_S\,dS_t+\frac12 C_{SS}(dS_t)^2.
$$

we get

$$
dC_t=\left(C_t+\frac12 \sigma^2 S_t^2 C_{SS}\right)dt+C_S\,dS_t.
$$

then 

$$
dV_t=-\left(C_t+\frac12 \sigma^2 S_t^2 C_{SS}\right)dt.
$$

the BS PDE in this case is:

$$
C_t+\frac12 \sigma^2 S^2 C_{SS}=0.
$$

$$
dV_t=0.
$$

so 

$$
V_t\equiv V_0.
$$

As we impose $$V_0=0$$ then

$$
V_t\equiv 0 \quad \text{for all } t.
$$

At maturity $$T$$, we have

$$
V_T=\Delta_T S_T+B_T-(S_T-K)^+=0.
$$

so 

$$
\Delta_T S_T+B_T=(S_T-K)^+.
$$


## Discrete hedging analysis

let times be

$$
0=t_0 \lt;t_1 \lt;\cdots\lt;t_N=T,
$$

with

$$
\Delta t_i=t_{i+1}-t_i,\qquad \Delta S_i=S_{t_{i+1}}-S_{t_i}.
$$

On interval $$[t_i,t_{i+1})$$, hold the stock position fixed at

$$
\Delta_i := C_S(t_i,S_{t_i}).
$$

Let the portfolio value after rehedging at time $$t_i$$ be

$$
V_{t_i}=\Delta_i S_{t_i}+B_i-C(t_i,S_{t_i}).
$$

Impose

$$
V_{t_0}=0,
$$

$$
B_0=C(0,S_0)-\Delta_0 S_0.
$$

$$
V_{t_{i+1}}-V_{t_i}
=\Delta_i (S_{t_{i+1}}-S_{t_i})-\big(C(t_{i+1},S_{t_{i+1}})-C(t_i,S_{t_i})\big).
$$

$$
\boxed{
\Delta V_i
= \Delta_i \Delta S_i-\Delta C_i
}
$$

where

$$
\Delta C_i:=C(t_{i+1},S_{t_{i+1}})-C(t_i,S_{t_i}).
$$

Summing over all intervals we have:

$$
V_T = V_0 + \sum_{i=0}^{N-1}\Delta V_i
=
\sum_{i=0}^{N-1}\left[\Delta_i\Delta S_i-\Delta C_i\right].
$$

since $$V_0=0$$ the terminal PNL is:

$$
\boxed{
\text{PNL}_T
=
\sum_{i=0}^{N-1}\left[\Delta_i\Delta S_i-\Delta C_i\right].
}
$$

We can expand $$\Delta C_i$$ by Taylor around $$(t_i, S_{t_i})$$:

$$
\Delta C_i
\approx
C_t\,\Delta t_i+C_S\,\Delta S_i+\frac12 C_{SS}(\Delta S_i)^2.
$$

All the derivatives (1st and 2nd order) are evaluated at $$(t_i,S_{t_i})$$

we have $$\Delta_i=C_S(t_i,S_{t_i})$$, the above equation can be rearranged:

$$
\Delta_i\Delta S_i-\Delta C_i
\approx
-C_t\,\Delta t_i-\frac12 \Gamma_i(\Delta S_i)^2.
$$

Using Black-Scholes PDE

$$
C_t=-\frac12 \sigma^2 S^2 \Gamma,
$$

we get 

$$
-C_t\,\Delta t_i
=
\frac12 \sigma^2 S_i^2 \Gamma_i\,\Delta t_i.
$$

Hence

$$
\boxed{
\Delta V_i
\approx
\frac12 \Gamma_i\left(\sigma^2 S_i^2 \Delta t_i-(\Delta S_i)^2\right).
}
$$

For vanilla call,

$$
\Gamma_i \gt 0.
$$

Particularly, for short call delta hedge:
- if realized squared $$(\Delta S_i)^2$$ is larger then model-predicted variance term $$\sigma^2 S_i^2 \Delta t_i$$ then

$$
\Delta V_i \lt 0
$$

so you lose

Effectively, short call option meaning short gamma. Big moves will hurt your portfolio value, even if you hedge.


```python
import numpy as np
import matplotlib.pyplot as plt
from math import log, sqrt, exp, erf, pi


# =========================
# Black-Scholes ingredients
# =========================

def norm_cdf(x):
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))


def norm_pdf(x):
    return (1.0 / sqrt(2.0 * pi)) * np.exp(-0.5 * x * x)


def bs_call_price_delta_gamma(S, K, tau, sigma):
    """
    Black-Scholes call price, delta, gamma with r=0.
    tau = time to maturity
    Works for scalar or numpy array S.
    """
    S = np.asarray(S, dtype=float)

    price = np.zeros_like(S)
    delta = np.zeros_like(S)
    gamma = np.zeros_like(S)

    # At maturity
    mask0 = (tau <= 0)
    if np.any(mask0):
        price[mask0] = np.maximum(S[mask0] - K, 0.0)
        delta[mask0] = (S[mask0] > K).astype(float)
        gamma[mask0] = 0.0

    # Before maturity
    mask = ~mask0
    if np.any(mask):
        Sm = S[mask]
        d1 = (np.log(Sm / K) + 0.5 * sigma**2 * tau) / (sigma * np.sqrt(tau))
        d2 = d1 - sigma * np.sqrt(tau)

        price[mask] = Sm * np.vectorize(norm_cdf)(d1) - K * np.vectorize(norm_cdf)(d2)
        delta[mask] = np.vectorize(norm_cdf)(d1)
        gamma[mask] = np.vectorize(norm_pdf)(d1) / (Sm * sigma * np.sqrt(tau))

    return price, delta, gamma


# =========================
# Path simulation
# =========================

def simulate_delta_hedged_short_call_pnl(
    S0=100.0,
    K=100.0,
    sigma=0.2,
    mu=0.0,
    T=1.0,
    n_steps=252,
    n_paths=50000,
    seed=42
):
    """
    Simulate terminal P&L V_T of a delta-hedged short call with r=0.

    Portfolio:
        V_t = Delta_t * S_t + B_t - C(t,S_t)

    Initial condition:
        V_0 = 0
        B_0 = C_0 - Delta_0 * S_0

    Rebalancing:
        Delta_i = BS delta at time t_i, held fixed over [t_i, t_{i+1})
        Cash updated self-financing at each rebalance.

    Returns:
        dict with pnl array and some diagnostics
    """
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    sqrt_dt = np.sqrt(dt)

    # Initial option quantities
    C0, Delta0, _ = bs_call_price_delta_gamma(
        np.array([S0]), K, T, sigma
    )
    C0 = C0[0]
    Delta0 = Delta0[0]

    # Initial cash chosen so V0 = 0
    B = np.full(n_paths, C0 - Delta0 * S0, dtype=float)
    Delta = np.full(n_paths, Delta0, dtype=float)
    S = np.full(n_paths, S0, dtype=float)

    # Track approximate gamma-based PnL too, for comparison
    gamma_pnl = np.zeros(n_paths, dtype=float)

    for i in range(n_steps):
        t_i = i * dt
        tau_i = T - t_i

        # Current gamma for approximation
        _, _, Gamma_i = bs_call_price_delta_gamma(S, K, tau_i, sigma)

        # Simulate stock move under GBM
        Z = rng.standard_normal(n_paths)
        S_next = S * np.exp((mu - 0.5 * sigma**2) * dt + sigma * sqrt_dt * Z)

        # Gamma approximation of one-step hedging error:
        # dV_i ≈ -1/2 * Gamma_i * ((ΔS)^2 - sigma^2 * S_i^2 * dt)
        dS = S_next - S
        gamma_pnl += -0.5 * Gamma_i * (dS**2 - sigma**2 * S**2 * dt)

        # Move to next time
        S = S_next

        # Rehedge at t_{i+1}, except maturity payoff happens at T
        t_next = (i + 1) * dt
        tau_next = max(T - t_next, 0.0)

        if tau_next > 0:
            _, Delta_new, _ = bs_call_price_delta_gamma(S, K, tau_next, sigma)

            # Self-financing rebalance:
            # Delta_new * S + B_new = Delta_old * S + B_old
            # so B_new = B_old - (Delta_new - Delta_old) * S
            B = B - (Delta_new - Delta) * S
            Delta = Delta_new

    # Terminal option payoff
    payoff = np.maximum(S - K, 0.0)

    # Terminal portfolio value
    pnl = Delta * S + B - payoff

    return {
        "pnl": pnl,
        "gamma_pnl_approx": gamma_pnl,
        "mean": np.mean(pnl),
        "std": np.std(pnl),
        "q01": np.quantile(pnl, 0.01),
        "q05": np.quantile(pnl, 0.05),
        "median": np.quantile(pnl, 0.50),
        "q95": np.quantile(pnl, 0.95),
        "q99": np.quantile(pnl, 0.99),
        "C0": C0,
        "Delta0": Delta0,
    }


# =========================
# Run example
# =========================

if __name__ == "__main__":
    result = simulate_delta_hedged_short_call_pnl(
        S0=100.0,
        K=100.0,
        sigma=0.2,
        mu=0.0,        # drift does not matter much for hedging-error structure
        T=1.0,
        n_steps=52,    # weekly rehedging
        n_paths=50000,
        seed=123
    )

    pnl = result["pnl"]
    gamma_pnl = result["gamma_pnl_approx"]

    print("Initial call price C0      :", result["C0"])
    print("Initial delta Delta0       :", result["Delta0"])
    print("Mean terminal P&L          :", result["mean"])
    print("Std terminal P&L           :", result["std"])
    print("1% quantile               :", result["q01"])
    print("5% quantile               :", result["q05"])
    print("Median                    :", result["median"])
    print("95% quantile              :", result["q95"])
    print("99% quantile              :", result["q99"])
    print("Corr(exact P&L, gamma approx):", np.corrcoef(pnl, gamma_pnl)[0, 1])

    # Histogram of exact terminal P&L
    plt.figure(figsize=(8, 5))
    plt.hist(pnl, bins=120, density=True)
    plt.axvline(np.mean(pnl), linestyle="--", label=f"mean = {np.mean(pnl):.4f}")
    plt.title("Terminal P&L Distribution: Delta-Hedged Short Call (r=0)")
    plt.xlabel("Terminal P&L")
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Compare exact P&L to gamma approximation
    plt.figure(figsize=(8, 5))
    plt.scatter(gamma_pnl[:5000], pnl[:5000], s=5, alpha=0.4)
    plt.title("Exact Terminal P&L vs Gamma Approximation")
    plt.xlabel("Gamma Approximation")
    plt.ylabel("Exact Terminal P&L")
    plt.tight_layout()
    plt.show()
```

















