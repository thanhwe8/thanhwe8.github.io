---
layout: post
title: "Understand Stochastic Volatility Modeling - Chapter 1"
date: 2026-03-06 
categories: [Stochastic Volatility Modeling]
---


> In this post, we will analyze the option on multi-asset basket

## Concepts Review
In this section, we will review concepts used to derive the PNL formula for option on multi-asset basket

### Linear algebra




Let the option depend on $$n$$ assets $$S_1,\dots,S_n$$ each with dividend yield $$q_i$$.

We create a zero initial cost portfolio by short one option and hold $$\Delta_i$$ unit of asset $$i$$. Also keep a cash account $$C_t$$ as:

$$\Pi_t=-P_t+\sum_{i=1}^n \Delta_i(t)\,S_i(t)+C_t.$$

We evaluate the infinitesimal hedging PNL with the package marked to zero at time $$t$$:

$$
\Pi_t=0.
$$

The cash amount is

$$
C_t=P_t-\sum_i \Delta_i S_i.
$$

#### Self-financing increment

Over $$[t,t+dt]$$, the portfolio increment is

$$
d\Pi_t
=
-dP_t
+\sum_i \Delta_i\, dS_i
+\sum_i \Delta_i q_i S_i\,dt
+rC_t\,dt.
$$

now substitute cash account into formula above

$$
d\Pi_t
=
-dP_t
+\sum_i \Delta_i dS_i
+\sum_i \Delta_i q_i S_i\,dt
+r\left(P_t-\sum_i \Delta_i S_i\right)dt.
$$

We can rearrange as:

$$
d\Pi_t
=
-dP_t
+\sum_i \Delta_i dS_i
+rP_t\,dt
-\sum_i (r-q_i)\Delta_i S_i\,dt.
$$

we can rewrite as 

$$
d\Pi_t
=
-dP_t
+\sum_i P_{S_i}\,dS_i
+rP_t\,dt
-\sum_i (r-q_i)S_i P_{S_i}\,dt.
$$ 

where the delta for each asset $$i$$ is

$$
\Delta_i = \frac{\partial P}{\partial S_i} = P_{S_i}.
$$

We can use the multi-variable Ito expansion of $$P(t,S_1,\dots,S_n)$$. Assuming $$P$$ is smooth function, we have:

$$
dP_t
=
P_t\,dt
+\sum_i P_{S_i}\,dS_i
+\frac12\sum_{i,j} P_{S_iS_j}\, d\langle S_i,S_j\rangle_t.
$$

With cross-variation $$d\langle S_i,S_j\rangle_t,$$ over a short-time interval, we can think of 

$$
d\langle S_i,S_j\rangle_t \approx \delta S_i\,\delta S_j.
$$

At this step, we can substitute the expansion of $$dP_t$$ into $$d\Pi_t$$

$$
d\Pi_t
=
-\left(
P_t\,dt
+\sum_i P_{S_i}\,dS_i
+\frac12\sum_{i,j}P_{S_iS_j}\,d\langle S_i,S_j\rangle_t
\right)
+\sum_i P_{S_i}\,dS_i
+rP_t\,dt
-\sum_i (r-q_i)S_iP_{S_i}\,dt.
$$

as the linear $$dS_i$$ term cancel, we get:

$$
d\Pi_t
=
-\left(
P_t-rP+\sum_i (r-q_i)S_iP_{S_i}
\right)dt
-\frac12\sum_{i,j}P_{S_iS_j}\,d\langle S_i,S_j\rangle_t.
$$

where

$$
A(t,S)=
P_t-rP+\sum_i (r-q_i)S_iP_{S_i}.
$$

Finally we can derive Bergomi's formula

$$
d\Pi_t
=
-A(t,S)\,dt
-\frac12\sum_{i,j}P_{S_iS_j}\,d\langle S_i,S_j\rangle_t.
$$




## Review of quadratic variation

A standard theorem in Ito calculus:

if $$X$$ and $$Y$$ are continuous semimartingales and $$a_t,b_t$$ are adapted continous finite-variation processes, then

$$
d\langle \int_0^\cdot a_u\,dX_u,\ \int_0^\cdot b_u\,dY_u\rangle_t
=
a_t b_t\, d\langle X,Y\rangle_t.
$$

If we write it as differential, this is the familiar rule:

$$
d(aX)\,d(bY)=abdXdY
$$

where $$a,b$$ are locally non-random at time $$t$$ to first order

We can denote process:

$$
dR_i = \frac{dS_i}{S_i},\qquad dR_j=\frac{dS_j}{S_j}.
$$

$$
dS_i = S_i\, dR_i,\qquad dS_j = S_j\, dR_j.
$$

then we have

$$
d\langle S_i,S_j\rangle_t
=
S_i(t)S_j(t) \langle R_i, R_j \rangle_t = S_i(t)S_j(t) \frac{dS_i}{S_i} \frac{dS_j}{S_j}.
$$

using property of quadratic variation

$$
d\langle S_i,S_j\rangle_t
=
S_iS_j\, d\left\langle \frac{S_i}{S_i},\frac{S_j}{S_j}\right\rangle_t
\approx
S_iS_j\frac{\delta S_i}{S_i}\frac{\delta S_j}{S_j}.
$$

Define

$$
\phi_{ij}(t,S)=S_iS_jP_{S_iS_j}.
$$

Then we can derive equation (1.6) in Bergomi's book as below

$$
d\Pi_t
=
-A(t,S)\,dt
-\frac12\sum_{i,j}\phi_{ij}(t,S)\,
\frac{\delta S_i}{S_i}\frac{\delta S_j}{S_j}.
$$

## 3-asset examples

Suppose $$P=P(t,S_1,S_2,S_3)$$. Using Ito we have:

$$
dP
=
P_tdt + P_{S_1}dS_1 + P_{S_2}dS_2 + P_{S_3}dS_3
$$

$$
\quad + \frac12 P_{S_1S_1}(dS_1)^2
+ \frac12 P_{S_2S_2}(dS_2)^2
+ \frac12 P_{S_3S_3}(dS_3)^2
$$

$$
\quad + P_{S_1S_2}dS_1dS_2
+ P_{S_1S_3}dS_1dS_3
+ P_{S_2S_3}dS_2dS_3.
$$







# Summary

For an option

$$
P=P(t,S_1,\dots,S_n),
$$

after delta-hedging each asset, the short-option hedged PNL over a short interval is

$$
PNL
=
-A(t,S)\,dt
-\frac12\sum_{i,j}\phi_{ij}(t,S)\,
\frac{\delta S_i}{S_i}\frac{\delta S_j}{S_j},
$$

with 

$$
A(t,S)=
\left(
\frac{\partial P}{\partial t}
-rP
+\sum_i (r-q_i)S_i\frac{\partial P}{\partial S_i}
\right),
$$

and 

$$
\phi_{ij}(t,S)=S_iS_j\frac{\partial^2P}{\partial S_i\partial S_j}.
$$

using property of quadratic variation

