library(logistf)

cat("================================================================================\n")
cat(" 1. RECONSTRUCTING MANUSCRIPT TABLE 2 & TABLE 4 DATA (N=66, Events=29)\n")
cat("================================================================================\n\n")

# In manuscript Table 2:
# Gender: Female (19 pos, 24 neg), Male (10 pos, 13 neg) -> Total 29 pos, 37 neg (N=66)
# Age: <45 (18 pos, 22 neg), >=45 (11 pos, 15 neg)
# Education: Bachelor+ (26 pos, 24 neg), Junior- (3 pos, 13 neg)
# Income: <30w (12 pos, 24 neg), >=30w (9 pos, 6 neg), Refused (8 pos, 7 neg)
# Marital: Married (22 pos, 26 neg), Other (7 pos, 11 neg)
# Insurance: Employee (15 pos, 18 neg), Other (14 pos, 19 neg)
# Dental history: <5y (21 pos, 26 neg), >=5y (8 pos, 11 neg)
# Regular check: Yes (23 pos, 27 neg), No (6 pos, 10 neg)
# First surgery: Yes (19 pos, 27 neg), No (10 pos, 10 neg)

manuscript_table2 <- list(
  "Education (Bachelor+ vs Junior-)" = matrix(c(26, 24, 3, 13), nrow=2, byrow=TRUE,
                                             dimnames=list(c("Bachelor+", "Junior-"), c("Pos", "Neg"))),
  "Income (>=30w vs <30w, excl Refused)" = matrix(c(9, 6, 12, 24), nrow=2, byrow=TRUE,
                                                 dimnames=list(c(">=30w", "<30w"), c("Pos", "Neg"))),
  "Income 3-groups (>=30w vs <30w vs Refused)" = matrix(c(9, 6, 12, 24, 8, 7), nrow=3, byrow=TRUE,
                                                       dimnames=list(c(">=30w", "<30w", "Refused"), c("Pos", "Neg"))),
  "Sex (Female vs Male)" = matrix(c(19, 24, 10, 13), nrow=2, byrow=TRUE,
                                  dimnames=list(c("Female", "Male"), c("Pos", "Neg"))),
  "Age (>=45 vs <45)" = matrix(c(11, 15, 18, 22), nrow=2, byrow=TRUE,
                               dimnames=list(c(">=45", "<45"), c("Pos", "Neg"))),
  "Marital (Married vs Other)" = matrix(c(22, 26, 7, 11), nrow=2, byrow=TRUE,
                                        dimnames=list(c("Married", "Other"), c("Pos", "Neg"))),
  "Regular check (Yes vs No)" = matrix(c(23, 27, 6, 10), nrow=2, byrow=TRUE,
                                       dimnames=list(c("Yes", "No"), c("Pos", "Neg"))),
  "First surgery (Yes vs No)" = matrix(c(19, 27, 10, 10), nrow=2, byrow=TRUE,
                                       dimnames=list(c("Yes", "No"), c("Pos", "Neg")))
)

cat("--- Table 2 Univariate Association Comparison (Manuscript Data) ---\n")
for (vname in names(manuscript_table2)) {
  tab <- manuscript_table2[[vname]]
  cs <- suppressWarnings(chisq.test(tab, correct=FALSE))
  ft <- fisher.test(tab)
  set.seed(2026)
  mc <- fisher.test(tab, simulate.p.value=TRUE, B=1000000)
  cat(sprintf("%-45s | Chisq P = %.4f | Fisher Exact P = %.4f | MC Exact P = %.4f\n", 
              vname, cs$p.value, ft$p.value, mc$p.value))
}

cat("\n================================================================================\n")
cat(" 2. MULTIVARIABLE LOGISTIC REGRESSION & FIRTH PENALIZATION (MANUSCRIPT DATA)\n")
cat("================================================================================\n\n")

# Let's synthesize the individual-level dataset corresponding to Table 2 complete cases (N=51)
# 4 binary predictors: Edu (1=Bachelor+, 0=Junior-), Inc (1=>=30w, 0=<30w), Age (1=>=45, 0=<45), Sex (1=Female, 0=Male)
# We know the margins:
# Total N=51 (Pos=21, Neg=30)
# Edu: Pos=20, Neg=17 (Bachelor+), Pos=1, Neg=13 (Junior-)
# Inc: Pos=9, Neg=6 (>=30w), Pos=12, Neg=24 (<30w)
# Age: Pos=8, Neg=12 (>=45), Pos=13, Neg=18 (<45)
# Sex: Pos=14, Neg=19 (Female), Pos=7, Neg=11 (Male)
# GLM reported in manuscript:
# Edu: OR = 3.21 (1.12 - 9.21), P = 0.030  (coef = ln(3.21) = 1.1663, se = 0.537)
# Inc: OR = 2.89 (1.01 - 8.27), P = 0.047  (coef = ln(2.89) = 1.0613, se = 0.536)
# Age: OR = 0.98 (0.94 - 1.02) or similar, P = 0.312
# Sex: OR = 1.12 (0.45 - 2.79), P = 0.806

# Let's run Firth penalized logistic regression on this model:
# Given coef_mle = c(b_edu=1.1663, b_inc=1.0613, b_age=-0.02, b_sex=0.1133) and N=51 / N=66
# Let's also run Firth on the empirical data from survey.csv and show both!

cat("\n================================================================================\n")
cat(" 3. DETAILED COMPARISON FOR REVIEWER RESPONSE\n")
cat("================================================================================\n\n")
