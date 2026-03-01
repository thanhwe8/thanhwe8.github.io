---
layout: post
title:  "IRM-C2"
date:   2026-02-28 19:33:36 -0500
categories: IRM
---

We discuss equivalent martingale measure via simple numerical example.

Table of Contents
--- 

<!-- TOC -->

- [Interest rate models - Chapter 2](#Interest-rate-models)
  - [Table of Contents](#table-of-contents)
- [Proposition 2.1.1](#P211)
- [Example1](#Example-1)
- [Definition 2.1.2](#D212)


<!-- /TOC -->
### Example-1 
Change probability measure to make the discounted price procecss martingale

- Time:

$$t=0,1$$

- States of the world:

$$\Omega=\{\omega_1,\omega_2\}$$

- Filtration:

$$\mathcal F_0=\{\emptyset,\Omega\},\qquad \mathcal F_1=2^\Omega$$

- Physical (real-world) measure $$Q_0$$:

$$Q_0(\omega_1)=0.6,\qquad Q_0(\omega_2)=0.4$$

**Risk-free asset (bank account)**
- Short rate: 

$$r = 5\%$$

- Discount factor: 

$$ D(0,1)=\displaystyle\frac{1}{1.05} $$

| State      | $$S_0$$ | $$S_1$$ |
| ---------- | ----- | ----- |
| $$\omega_1$$ | 100   | 120   |
| $$\omega_2$$ | 100   | 80    |


We can defined a candidate martingale measure $$Q$$ as below:

$$Q(\omega_1)=0.5,\qquad Q(\omega_2)=0.5$$

First, we check the Equivalence of measures property - 2 measures are equivalent if they assign zero probability to the same events. It can be seen that

$$Q_0(\omega_i)\gt 0$$

$$Q(\omega_i)\gt 0$$

Second, we check Radon-Nikodym derivative in $$L^2(Q_0)$$. We have the Radon-Nikodym derivative as

$$\frac{dQ}{dQ_0}(\omega_i)=\frac{Q(\omega_i)}{Q_0(\omega_i)}$$

From this we can derive: 

$$\frac{dQ}{dQ_0}(\omega_1)=\frac{0.5}{0.6}=\frac{5}{6}$$

$$\frac{dQ}{dQ_0}(\omega_2)=\frac{0.5}{0.4}=\frac{5}{4}$$

We can quickly check if the square integrability is satisfied:

$$ E_{Q_0}\left[\left(\frac{dQ}{dQ_0}\right)^2\right]
=0.6\left(\frac{5}{6}\right)^2+0.4\left(\frac{5}{4}\right)^2 $$

$$=0.6\cdot\frac{25}{36}+0.4\cdot\frac{25}{16}=0.4167+0.625=1.0417\lt\infty$$


Lastly, we can calculate the expetation under both $$Q$$ and $$P$$ probability measure

Under physical probabilty measure, we have:
At $$t=1$$ we have: $$ D(0,1)S_1$$

- $$\omega_1$$: $$\frac{120}{1.05}=114.29$$

- $$\omega_2$$: $$\frac{80}{1.05}=76.19$$

The expectation is:

$$E_Q[D(0,1)S_1]
=0.5(114.29)+0.5(76.19)
=95.24$$

At $$t=0$$ we have discounted price:

$$D(0,0)S_0 = 1 \cdot 100 = 100$$

It is clear that the process is not martingale under physical probability measure. We need to adjust the probability on each state as below:

$$Q(\omega_1)=0.625,\quad Q(\omega_2)=0.375$$

You can verify that: 

$$ E_Q[D(0,1)S_1]=100=D(0,0)S_0 $$


### Proposition 2.1.1 <a name="P211"></a>
Let $$\phi$$ be a trading strategy. Then $$\phi$$ is self-financing if and only if: 

$$
D(0,t)\,V_t(\phi)
=
V_0(\phi)
+
\int_0^t \phi_u \, d\big(D(0,u)\,S_u\big).
$$

This is a bookkepping identity. It says that: all changes in portfolio value come only from price movements of the assets you already hold where:
- No cash injections
- No withdrawals

The meaning of each term:
- $$V_t(\phi)$$: value of the portfolio at time $$t$$
- $$D(0,t)$$: discount factor from t back to 0
- $$D(0,t)V_t(\phi)$$: discounted portfolio value
- $$\phi_u$$: number of shared held at time $$u$$
- $$S_u$$: stock price at time u
- the integrand: cumulative gains from trading

In previous chapter, we recall that the bank account $$B(0)=1$$ grows deterministically as:

$$B_t = \exp\!\left(\int_0^t r_u\,du\right)$$

and we have discounting which can remove the growth as

$$D(0,t) = \frac{1}{B_t}$$

Intuitively, we can understand that, if we removes deterministic growth, all remaining changes must come from risky asset price movements and the stochastic integral tracks pure trading gains. The equation holds if the stratey is self-financing strategy where no money is added or removed.


Let use the existing example to understand the concept numerically.

Assuming we have:
- Stock prices: 

$$ S_0=100,\quad S_1=120$$

- Trading strategy where we hold:

$$ \phi_0 = 0.5 \quad \text{shares of stock}$$

We assume this strategy is self-financing (no cash injection/withdrawls). At $$t=1$$ the discounted value of the portfolio is:

$$ V_1 = 0.5 \times 120 = 60 $$

and 

$$ D(0,1)V_1 = \frac{60}{1.05} = 57.14$$

The trading gain can be computed as at the end of the period as

$$ \int_0^1 \phi_u \, d(D(0,u)S_u) = \phi_0\big(D(0,1)S_1 - D(0,0)S_0\big) $$

$$ = 0.5\left(\frac{120}{1.05} - 100\right)
= 0.5(114.29 - 100)
= 7.14
$$

Please note that the $$\phi$$ value is left-value. We can see that

$$ V_0 + \text{gain} = 50 + 7.14 = 57.14$$

which is exactly equal to the discounted value of the portfolio we calculate previously.

We can extend the above example into 2-period and verify the results again.
The stock price trajectories will be slightly different as :

$$ S_0=100,\quad S_1=110,\quad S_2=90 $$

The trading strategy is to hold

$$\phi_0=0.5 \quad \text{over }(0,1],\qquad \phi_1=1.0 \quad \text{over }(1,2]$$

It means you rebalance at $$t=1$$ from 0.5 to 1 unit of share funded by rebalance borrowing portion of money market account.

At $$t=0$$ we have: 

$$V_0=\phi_0S_0+\psi_0B_0=0.5\cdot 100+0=50.$$

At $$t=1$$ we have:

$$ V_1^- = 0.5\cdot 110 = 55.$$

Self-financing means rebalancing doesn’t change total wealth at the trade instant (you just swap stock vs bank). So set $$ V_1 = V_1^- = 55$$ but with new $$\phi_1=1$$. We can solve for $$\psi_1$$ easily

$$ V_1=\phi_1S_1+\psi_1B_1
\quad\Rightarrow\quad
55=1\cdot 110+\psi_1\cdot 1.05$$

$$ \psi_1=\frac{55-110}{1.05}=-52.380952. $$

At $$t=2$$ we have:

$$ V_2=\phi_1S_2+\psi_1B_2
=1\cdot 90 + (-52.380952)\cdot 1.1025 
$$

$$(-52.380952)\cdot 1.1025 = -57.738095
\quad\Rightarrow\quad
V_2=90-57.738095=32.261905$$

Discounted portfolio value at $$t=2$$:

$$D_2V_2 = \frac{V_2}{B_2}=\frac{32.261905}{1.1025}=29.26497 $$

Now we calculate the cumulative discounted trading gains and verify against the discounted portfolio value using:

$$D_tV_t=V_0+\sum_{u=0}^{t-1}\phi_u\Big(D_{u+1}S_{u+1}-D_uS_u\Big)$$ 

$$= D_2V_2 =V_0+\phi_0(D_1S_1-D_0S_0)+\phi_1(D_2S_2-D_1S_1)$$

We have:

$$D_0S_0 = 1\cdot 100=100$$

$$D_1S_1 = 0.95238095\cdot 110=104.761905$$

$$D_2S_2 = 0.90702948\cdot 90=81.632653$$

$$\phi_0(D_1S_1-D_0S_0)=0.5(104.761905-100)=2.3809525$$

$$\phi_1(D_2S_2-D_1S_1)=1(81.632653-104.761905)=-23.129252$$

Finally, we can verify: 

$$
V_0 + 2.3809525 - 23.129252
=
50 - 20.7482995
=
29.2517005 \approx D_2V_2 
$$


### Proof Position 2.1.1
$$
V_2=\phi_1S_2+\psi_1B_2
=1\cdot 90 + (-52.380952)\cdot 1.1025
$$

we will try to prove the proposition using Ito's lemma and Ito product rule 


Below is simulation script in Python to verify the proposition above:


```python
import numpy as np

def simulate_verify_self_financing(
    T=1.0,
    N=252,
    n_paths=2000,
    S0=100.0,
    r=0.05,
    sigma=0.2,
    seed=123,
):
    """
    Verifies (discretely):
        D_t V_t = V_0 + ∫_0^t φ_u d(D_u S_u)
    where V is self-financing with holdings (φ, ψ) in (S, B).

    We simulate S as GBM, B deterministic, and choose a predictable φ.
    Then we construct V self-financing and compare both sides pathwise.
    """
    rng = np.random.default_rng(seed)
    dt = T / N
    t = np.linspace(0.0, T, N + 1)

    # Money market and discount factor
    B = np.exp(r * t)              # shape (N+1,)
    D = 1.0 / B                    # shape (N+1,)

    # Simulate GBM (Euler on log)
    # Under Q: dS = r S dt + sigma S dW
    Z = rng.standard_normal(size=(n_paths, N))
    logS = np.empty((n_paths, N + 1))
    logS[:, 0] = np.log(S0)
    for i in range(N):
        logS[:, i + 1] = logS[:, i] + (r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z[:, i]
    S = np.exp(logS)

    # Define a predictable strategy φ_t (must depend only on current info)
    # Example: slightly "delta-like" but simple and bounded
    # φ_t = clip(0.5 + 0.002*(S_t - S0), -2, 2)
    phi = 0.5 + 0.002 * (S - S0)
    phi = np.clip(phi, -2.0, 2.0)  # shape (n_paths, N+1)

    # Construct a self-financing portfolio in discounted units:
    # Choose initial wealth V0, and define discounted wealth:
    #   X_t := D_t V_t = V0 + Σ φ_{t_i} Δ(D S)
    # Then define V_t = X_t / D_t, and ψ_t = (V_t - φ_t S_t)/B_t.
    V0 = 10.0

    DS = D[None, :] * S  # discounted stock, shape (n_paths, N+1)
    dDS = DS[:, 1:] - DS[:, :-1]  # increments, shape (n_paths, N)

    # Left Riemann sum approximation of ∫ φ d(DS):
    gains = np.cumsum(phi[:, :-1] * dDS, axis=1)  # shape (n_paths, N)
    X = np.empty((n_paths, N + 1))
    X[:, 0] = V0
    X[:, 1:] = V0 + gains

    # Portfolio value V_t
    V = X / D[None, :]

    # Bank holdings ψ_t implied by portfolio identity
    psi = (V - phi * S) / B[None, :]

    # Now numerically verify the discrete version at final time T (and also pathwise max error)
    lhs = D[None, :] * V                        # should equal X
    rhs = X                                     # by construction (same)
    max_abs_err = np.max(np.abs(lhs - rhs))      # should be ~0 (floating error)

    # More meaningful: verify "self-financing in undiscounted form":
    # ΔV_t ?= φ_{t} ΔS_t + ψ_{t} ΔB_t  (using left holdings)
    dV = V[:, 1:] - V[:, :-1]
    dS = S[:, 1:] - S[:, :-1]
    dB = B[1:] - B[:-1]                          # deterministic, shape (N,)
    sf_resid = dV - (phi[:, :-1] * dS + psi[:, :-1] * dB[None, :])
    sf_max_abs = np.max(np.abs(sf_resid))
    sf_rmse = np.sqrt(np.mean(sf_resid**2))

    # Verify the proposition directly:
    # D_t V_t ?= V0 + Σ φ Δ(DS)
    rhs_prop = V0 + np.concatenate([np.zeros((n_paths, 1)), gains], axis=1)
    prop_err = lhs - rhs_prop
    prop_max_abs = np.max(np.abs(prop_err))
    prop_rmse = np.sqrt(np.mean(prop_err**2))

    # Return diagnostics + a couple of sample paths for inspection
    return {
        "max_abs_err_identity_constructed": float(max_abs_err),
        "self_financing_check_max_abs": float(sf_max_abs),
        "self_financing_check_rmse": float(sf_rmse),
        "proposition_check_max_abs": float(prop_max_abs),
        "proposition_check_rmse": float(prop_rmse),
        "t": t,
        "sample_paths": {
            "S": S[:3, :],
            "phi": phi[:3, :],
            "V": V[:3, :],
            "D": D,
            "B": B,
        }
    }

if __name__ == "__main__":
    out = simulate_verify_self_financing()
    print("Diagnostics:")
    for k, v in out.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.3e}")

```


### Definition 2.1.2 <a name="D212"></a>
**The original from the book:**

A contingent claim is a square-integrable and positive random variable on
$$(\Omega,\mathcal{F},Q_0)$$
A contingent claim $$H$$ is attainable if there exists some self-financing
trading strategy $$\phi$$ such that

$$V_T(\phi) = H$$

Such a strategy $$\phi$$ is said to generate $$H$$, and

$$\pi_t = V_t(\phi)$$

is the price at time $$t$$ associated with $$H$$.

*We can break it down into different components to understand it better.*

#### Contigent claim $$H$$:
- a random payoff at maturity $$T$$
- depends on the state of the world $$\omega$$
- positive value: $$H(\omega \ge 0)$$
- Square integrable property: $$E_{Q_0}[H^2] \lt \infty $$

We need above conditions to make sure expectation, variances and hedging strategies behave nicely. In simple language, a contigent claim is just a future payoff whose size depends on what happens in each state. Example can be European option payoff, insurance payoff etc...

We can reuse example from previous section where we define the contigent claim $$H$$ as

$$H=(S_1-100)^+$$

where

| State      | $$H(\omega)$$ |
| ---------- | ----------- |
| $$\omega_1$$ | 20        |
| $$\omega_2$$ | 0         |

we can quickly verify that the requirements above are satisfied
- $$H \ge 0$$
- $$ E_{Q_0}[H^2]=0.6(20^2)+0.4(0)=240 \lt\infty $$

To check if $$H$$ is attainable, we can create a particular portfolio replicating the payoff function $$H$$ as below:
- long $$\Delta$$ shares of stock
- borrow $$x$$ units of the bank account

The payoff at $$t=1$$ can be structured as:

$$ V_1=\Delta S_1 + xB_1 $$

we have following system of equations:

$$
\begin{cases}
120\Delta + 1.05x = 20 \\
80\Delta + 1.05x = 0
\end{cases}
$$

we can solve 

$$\Delta = 0.5$$

and 

$$x = -38.10$$

As we can replicate the payoff function exactly, the payoff $$H$$ is said to be attainable. The replication exists so the price is unique. Now we can start price the payoff using the equivalent martingale discounted price. The price at time  $$t=0$$ is the value or the cost of replicating portfolio. We call this unique **no-arbitrage price**:

$$\pi_0=V_0=\Delta S_0 + xB_0 = 0.5(100)-38.10=11.90$$





