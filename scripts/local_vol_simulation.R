set.seed(123)

# 1. Input setups

S0 <- 100
r <- 0.00
q <- 0.00

K_grid <- seq(80, 120, by = 5)
T_grid <- c(0.5, 1.0, 1.5)

impvol_mkt <- function(K, T) {
    0.20 + 0.08 * (log(K / 100))^2 + 0.03 * exp(-T)
}

bs_call <- function(S0, K, r, q, T, sigma) {
    if (T <= 0 || sigma <= 0) {
        return(max(S0 * exp(-q * T) - K * exp(-r * T), 0))
    }

    d1 <- (log(S0 / K) + (r - q + 0.5 * sigma^2) * T) / (sigma * sqrt(T))
    d2 <- d1 - sigma * sqrt(T)
    S0 * exp(-q * T) * pnorm(d1) - K * exp(-r * T) * pnorm(d2)
}

# 2. Build synthetic market price surface

C_mkt <- matrix(NA, nrow = length(K_grid), ncol = length(T_grid))

for (i in seq_along(K_grid)) {
    for (j in seq_along(T_grid)) {
        K <- K_grid[i]
        T <- T_grid[j]
        vol <- impvol_mkt(K, T)
        C_mkt[i, j] <- bs_call(S0, K, r, q, T, vol)
    }
}

rownames(C_mkt) <- paste0("K=", K_grid)
colnames(C_mkt) <- paste0("T=", T_grid)
cat("Market call price: \n")
print(round(C_mkt, 4))



# 3. Dupire finite-difference calibration

K_dense <- seq(70, 120, by = 2)
T_dense <- seq(0.25, 2.00, by = 0.25)
C_dense <- matrix(NA, nrow = length(K_dense), ncol = length(T_dense))

for (i in seq_along(K_dense)) {
    for (j in seq_along(T_dense)) {
        K <- K_dense[i]
        T <- T_dense[j]
        vol <- impvol_mkt(K, T)
        C_dense[i, j] <- bs_call(S0, K, r, q, T, vol)
    }
}

dK <- K_dense[2] - K_dense[1]
dT <- T_dense[2] - T_dense[1]

dC_dK <- matrix(NA, nrow = length(K_dense), ncol = length(T_dense))
d2C_dK2 <- matrix(NA, nrow = length(K_dense), ncol = length(T_dense))
dC_dT <- matrix(NA, nrow = length(K_dense), ncol = length(T_dense))


# strike derivatives
for (j in seq_along(T_dense)) {
    for (i in 2:(length(K_dense) - 1)) {
        dC_dK[i, j] <- (C_dense[i + 1, j] - C_dense[i - 1, j]) / (2 * dK)
        d2C_dK2[i, j] <- (C_dense[i + 1, j] - 2 * C_dense[i, j] + C_dense[i - 1, j]) / (dK^2)
    }
}

rownames(dC_dK) <- paste0("K=", K_dense)
colnames(dC_dK) <- paste0("T=", T_dense)
cat("dC_dK: \n")
print(round(dC_dK, 4))

rownames(d2C_dK2) <- paste0("K=", K_dense)
colnames(d2C_dK2) <- paste0("T=", T_dense)
cat("dC_dK: \n")
print(round(d2C_dK2, 4))


# maturity derivatives
for (i in seq_along(K_dense)) {
    for (j in 2:(length(T_dense) - 1)) {
        dC_dT[i, j] <- (C_dense[i, j + 1] - C_dense[i, j - 1]) / (2 * dT)
    }
}

local_var <- matrix(NA, nrow = length(K_dense), ncol = length(T_dense))
print(dC_dT)


for (i in 2:(length(K_dense) - 1)) {
    for (j in 2:(length(T_dense) - 1)) {
        K <- K_dense[i]
        numerator <- dC_dT[i, j] + q * C_dense[i, j] + (r - q) * K * dC_dK[i, j]
        denominator <- 0.5 * K^2 * d2C_dK2[i, j]

        if (!is.na(denominator) && denominator > 1e-10 && numerator > 0) {
            local_var[i, j] <- numerator / denominator
        }
    }
}

local_vol <- sqrt(local_var)

rownames(local_vol) <- paste0("K=", K_dense)
colnames(local_vol) <- paste0("T=", T_dense)
cat("dC_dK: \n")
print(round(local_vol, 4))


# Bilinear interpolation for local vol
interp_local_vol <- function(t, s, T_dense, K_dense, local_vol) {
    # clamp to grid
    t <- min(max(t, min(T_dense)), max(T_dense))
    s <- min(max(s, min(K_dense)), max(K_dense))

    iT <- max(which(T_dense <= t))
    iK <- max(which(K_dense <= s))

    if (iT >= length(T_dense)) iT <- length(T_dense) - 1
    if (iK >= length(K_dense)) iK <- length(K_dense) - 1

    T1 <- T_dense[iT]
    T2 <- T_dense[iT + 1]
    K1 <- K_dense[iK]
    K2 <- K_dense[iK + 1]

    z11 <- local_vol[iK, iT]
    z12 <- local_vol[iK, iT + 1]
    z21 <- local_vol[iK + 1, iT]
    z22 <- local_vol[iK + 1, iT + 1]

    # fallback if NA nearby
    vals <- c(z11, z12, z21, z22)
    if (any(is.na(vals))) {
        good <- vals[!is.na(vals)]
        if (length(good) == 0) {
            return(0.20)
        }
        return(mean(good))
    }

    wtT <- (t - T1) / (T2 - T1)
    wtK <- (s - K1) / (K2 - K1)

    z1 <- z11 * (1 - wtK) + z21 * wtK
    z2 <- z12 * (1 - wtK) + z22 * wtK
    z <- z1 * (1 - wtT) + z2 * wtT

    return(max(z, 1e-4))
}

simulate_local_vol_price <- function(K, T, n_paths = 10000, n_steps = 200) {
    dt <- T / n_steps
    S <- rep(S0, n_paths)

    for (step in 1:n_steps) {
        t <- (step - 1) * dt
        z <- rnorm(n_paths)
        sig <- sapply(S, function(s) interp_local_vol(t + 1e-8, s, T_dense, K_dense, local_vol))
        S <- S + (r - q) * S * dt + sig * S * sqrt(dt) * z
        S <- pmax(S, 1e-8)
    }

    payoff <- pmax(S - K, 0)
    exp(-r * T) * mean(payoff)
}


results <- data.frame()

reuslts
