# Survey reference values for cross-validation with diff-diff Python tests.
#
# This script generates reference OLS/WLS coefficients and Taylor-linearized
# standard errors using R's survey package (Lumley 2004).
#
# To regenerate:
#   Rscript tests/reference/survey_reference.R
#
# The output CSV is read by tests/test_survey.py for Tier 2 cross-validation.

library(survey)

set.seed(42)
n <- 200

# Generate survey DID data with 4 strata, 5 PSUs per stratum
strata <- rep(1:4, each = n / 4)
psu <- rep(1:20, each = n / 20)
weights_raw <- runif(n, 0.5, 2.0)

treated <- rep(c(0, 1), each = n / 2)
post <- rep(c(0, 1, 0, 1), each = n / 4)
interaction <- treated * post

# DGP: y = 1 + 0.5*treated + 0.3*post + 2.0*interaction + noise
noise <- rnorm(n, 0, 1)
y <- 1 + 0.5 * treated + 0.3 * post + 2.0 * interaction + noise

df <- data.frame(
  y = y,
  treated = treated,
  post = post,
  interaction = interaction,
  strata = strata,
  psu = psu,
  wt = weights_raw
)

# --- WLS without survey design (just weights) ---
wls_fit <- lm(y ~ treated + post + interaction, data = df, weights = wt)
wls_coef <- coef(wls_fit)
wls_se <- sqrt(diag(vcov(wls_fit)))

# --- Survey-weighted with TSL variance ---
design <- svydesign(
  ids = ~psu,
  strata = ~strata,
  weights = ~wt,
  data = df,
  nest = TRUE
)

svy_fit <- svyglm(y ~ treated + post + interaction, design = design)
svy_coef <- coef(svy_fit)
svy_se <- sqrt(diag(vcov(svy_fit)))
svy_df <- degf(design)

# --- No FPC, lonely_psu = "remove" (R default is "fail", change it) ---
options(survey.lonely.psu = "remove")

# --- Output ---
results <- data.frame(
  parameter = c("intercept", "treated", "post", "interaction"),
  wls_coef = as.numeric(wls_coef),
  wls_se = as.numeric(wls_se),
  svy_coef = as.numeric(svy_coef),
  svy_se = as.numeric(svy_se),
  svy_df = rep(svy_df, 4)
)

write.csv(results, "tests/reference/survey_reference.csv", row.names = FALSE)
cat("Reference values written to tests/reference/survey_reference.csv\n")
print(results)
