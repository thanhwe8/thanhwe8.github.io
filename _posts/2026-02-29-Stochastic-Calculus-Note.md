---
layout: post
title: "Note on stochastic calculus"
date: 2026-03-01 
categories: [Interest Rate Models]
---

Below are some important note on stochastic calculus when you work with different models. The proof will be provided at the next section sequentially.
- Ito Isometry
- Ito's product rule
- Quadratic variation
- Martingale Representation Theorem


## Quadratic variation

Take any real-valued process $$X_t$$ on $$[0,T]$$

Choose a partition

$$\Pi_n = \{0 = t_0 \lt; t_1 \lt; \dots \lt; t_n = T\}$$

with mesh size 

$$|\Pi_n| \to 0$$

Define the quadratic variation along the partition:

$$
[ X ]_T^{(\Pi_n)} =
\sum_{k=0}^{n-1} \bigl(X_{t_{k+1}} - X_{t_k}\bigr)^2
$$

If this converges in probability to a limit as the partition gets finer, we call the limit the quadratic variation

$$
\boxed{
[ X ]_T
=
\lim_{|\Pi_n|\to 0}
\sum_{k}
(\Delta X_{t_k})^2
}
$$

Now the main result. 

Let

$$
Y_t
=
\int_0^t X_s\, dW_s
$$

where $$X_s$$ is an Ito-integrable process.

Quadratic variation of an Ito-integral as:

$$
\boxed{
[ Y ]_T
=
\int_0^T X_s^2\, ds
}
$$

**Proof**:

Over a small interval $$[t_k, t_{k+1}]$$

$$
\Delta Y_k
=
\int_{t_k}^{t_{k+1}} X_s\, dW_s
$$

For small intervals, $$X_s \approx X_{t_k}$$, so:

$$
\Delta Y_k
\approx
X_{t_k}(W_{t_{k+1}} - W_{t_k})
$$

Square it:

$$
(\Delta Y_k)^2
\approx
X_{t_k}^2 (\Delta W_k)^2
$$

Sum over the partition:

$$
\sum_k (\Delta Y_k)^2
\approx
\sum_k X_{t_k}^2 (\Delta W_k)^2
$$

But recall:

$$
(\Delta W_k)^2 \approx \Delta t_k
$$

So:

$$
\sum_k (\Delta Y_k)^2
\;\to\;
\sum_k X_{t_k}^2 \Delta t_k
\;\to\;
\int_0^T X_s^2 ds
$$


## Ito's product rule

Let $$X_t$$ and $$Y_t$$ be Ito process:

$$
\begin{aligned}
dX_t = a_t\,dt + b_t\,dW_t, \\
dY_t = c_t\,dt + d_t\,dW_t,
\end{aligned}
$$

where all coefficients are adapted and square-integrable.

Then the product $$Z_t = X_t Y_t$$ satisfies:

$$
\boxed{
d(X_t Y_t)
=
X_t\, dY_t
+
Y_t\, dX_t
+
d\langle X, Y\rangle_t
}
$$

Here

$$
\boxed{
d\langle X, Y\rangle_t = b_t d_t\, dt
}
$$

is the quadratic covariation of $$X$$ and $$Y$$.














