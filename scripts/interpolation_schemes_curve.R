# Nodes data
T1 <- 1.0
T2 <- 2.0
P1 <- 0.95
P2 <- 0.90

# Continous-compounded zero rates at nodes
z1 <- -log(P1) / T1
z2 <- -log(P2) / T2

# Grid from 1.1 to 1.9
T_grid <- seq(1.1, 1.9, by = 0.1)

# 1.Log-linear interpolation on DF
f_log_df <- function(T) {
    slope_lnP <- (log(P2) - log(P1)) / (T2 - T1)
    return(-slope_lnP)
}

# 2. Linear interpolation on DF
f_lin_df <- function(T) {
    slope_P <- (P2 - P1) / (T2 - T1)
    P_T <- P1 + slope_P * (T - T1)
    return(-slope_P / P_T)
}

# 3. Linear interpolation on zero-rate
f_lin_zero <- function(T) {
    b <- (z2 - z1) / (T2 - T1)
    z_T <- z1 + b * (T - T1)
    return(z_T + T * b)
}

# 4. Linear interpolation on log zero rate
f_log_zero <- function(T) {
    b <- (log(z2) - log(z1)) / (T2 - T1)
    z_T <- z1 * exp(b * (T - T1))
    return(z_T * (1 + b * T))
}

# 5. Result
result <- data.frame(
    T = T_grid,
    f_log_linear_DF = sapply(T_grid, f_log_df),
    f_linear_DF = sapply(T_grid, f_lin_df),
    f_linear_zero = sapply(T_grid, f_lin_zero),
    f_log_zero = sapply(T_grid, f_log_zero)
)


# Calculate forward rate
result_pct <- result
result_pct[, -1] <- 100 * result_pct[, -1]

cat("Instantaneous forward rates (decimal):\n")
print(result, row.names = FALSE)

cat("\nInstantaneous forward rates (%):\n")
print(result_pct, row.names = FALSE)


matplot(
    result$T,
    result[, -1],
    type = "l",
    lty = 1,
    lwd = 2,
    xlab = "Maturity T",
    ylab = "Instantaneous forward rate",
    main = "Forward rates under 4 interpolation schemes"
)
legend(
    "topleft",
    legend = c(
        "Log-linear DF",
        "Linear DF",
        "Linear zero",
        "Log zero"
    ),
    col = 1:4,
    lty = 1,
    lwd = 2,
    bty = "n"
)

# Full-blown example
pillars <- c(1 / 12, 0.25, 0.5, 1, 2, 3, 5, 7, 10, 15, 20, 30)

zero_rates <- c(
    0.0200, 0.0210, 0.0220, 0.0240, 0.0270, 0.0290,
    0.0320, 0.0340, 0.0360, 0.0380, 0.0390, 0.0400
)

dfs <- exp(-zero_rates * pillars)

curve_df <- data.frame(
    T = pillars,
    z = zero_rates,
    P = dfs
)

print(curve_df)

# Build grid from 1 month to 30 years
T_grid <- seq(1 / 12, 30, by = 1 / 12)

find_interval <- function(T, pillars) {
    if (T < min(pillars) || T > max(pillars)) {
        stop("T outside pillar range")
    }
    if (T == max(pillars)) {
        return(length(pillars) - 1)
    }
    i <- max(which(pillars <= T))
    if (i == length(pillars)) i <- i - 1
    return(i)
}

# We have to rewrite above function to adapt to algorithm on full grid
