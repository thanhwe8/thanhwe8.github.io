---
layout: post
title:  "Interest Rate Models - Chapter 3: Vasicek Model"
date:   2026-03-01 
categories: [Interest Rate Models]
---

In this post, we analyze short-rate Vasicek model and its application

Table of Contents
--- 

<!-- TOC -->

- [Interest rate models - Chapter 2](#Interest-rate-models)
  - [Table of Contents](#table-of-contents)
- [Proposition 2.1.1](#P211)

<!-- /TOC -->

# Model and bond pricing formula:
Under the risk-neutral measure $$Q$$

$$dr_t=\kappa(\theta-r_t)\,dt+\sigma\,dW_t$$

The zero-coupon bond price is affine in $$r_t$$ 

$$
P(t,T)=A(t,T)\,e^{-B(t,T)r_t},
\quad
B(t,T)=\frac{1-e^{-\kappa (T-t)}}{\kappa}
$$

where $$A(t,T)$$ is deterministic or known closed form formula.

Assuming we have the 