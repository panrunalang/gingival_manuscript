library(logistf)

df <- read.csv('results/survey.csv', check.names=FALSE, stringsAsFactors=FALSE)

cat("====================================================================\n")
cat("  PART 2: TABLE 2 UNIVARIATE ASSOCIATIONS (Chi-Square vs Fisher Exact)\n")
cat("====================================================================\n\n")

y_raw <- df[['10、术前是否通过一些途径了解过？']]
y <- ifelse(grepl('主治', y_raw), 1, 0)
cat(sprintf("Outcome: Awareness (1 = Proactive Professional Inquiry, n=%d; 0 = Other, n=%d; Total N=%d)\n\n", 
            sum(y==1), sum(y==0), length(y)))

# Variables for Table 2
vars <- list(
  "Sex (Female vs Male)" = factor(ifelse(grepl('女', df[['1、您的性别：']]), 'Female', 'Male')),
  "Age (<45 vs >=45)" = factor(ifelse(grepl('45|60', df[['2、年龄：']]), '>=45', '<45')),
  "Education (Bachelor+ vs Junior-)" = factor(ifelse(grepl('本科|研究生', df[['3、教育程度：']]), 'Bachelor+', 'Junior-')),
  "Marital (Married vs Other)" = factor(ifelse(grepl('已婚', df[['4、婚姻状况：']]), 'Married', 'Other')),
  "Income (<30w vs >=30w, excl. refused)" = factor(ifelse(grepl('30-50|50-100|>100', df[['8、家庭年均收入情况：']]), '>=30w',
                                                   ifelse(grepl('不便', df[['8、家庭年均收入情况：']]), NA, '<30w'))),
  "Income 3-groups (<30w vs >=30w vs Refused)" = factor(ifelse(grepl('30-50|50-100|>100', df[['8、家庭年均收入情况：']]), '>=30w',
                                                        ifelse(grepl('不便', df[['8、家庭年均收入情况：']]), 'Refused', '<30w'))),
  "Insurance (Employee vs Other)" = factor(ifelse(grepl('职工', df[['9、医保类型：']]), 'Employee', 'Other')),
  "Dental history (<5y vs >=5y)" = factor(ifelse(grepl('5-10|>10', df[['5、首次接受牙科治疗至今的年限：']]), '>=5y', '<5y')),
  "Regular dental check (Yes vs No)" = factor(ifelse(grepl('是', df[['6、是否定期进行牙科检查：']]), 'Yes', 'No')),
  "First periodontal surgery (Yes vs No)" = factor(ifelse(grepl('是', df[['7、是否首次进行牙周膜龈手术治疗：']]), 'Yes', 'No'))
)

table2_res <- data.frame(
  Variable = character(),
  Level_Positive = character(),
  Pos_n = integer(),
  Pos_pct = numeric(),
  Level_Negative = character(),
  Neg_n = integer(),
  Neg_pct = numeric(),
  Chisq_P = numeric(),
  Fisher_P = numeric(),
  stringsAsFactors = FALSE
)

for (vname in names(vars)) {
  v <- vars[[vname]]
  valid_idx <- !is.na(v)
  tab <- table(v[valid_idx], y[valid_idx])
  cs <- suppressWarnings(chisq.test(tab))
  ft <- fisher.test(tab)
  
  cat(sprintf("Variable: %s (N = %d)\n", vname, sum(valid_idx)))
  print(tab)
  cat(sprintf("  Chi-Square P = %.4f | Fisher's Exact P = %.4f\n\n", cs$p.value, ft$p.value))
}

cat("\n====================================================================\n")
cat("  PART 3: MULTIVARIABLE LOGISTIC REGRESSION: STANDARD GLM vs FIRTH\n")
cat("====================================================================\n\n")

# Model 1: Original Manuscript Model (N=51, excluding Refused Income or N=66 with 3-level income)
gender_num <- ifelse(grepl('女', df[['1、您的性别：']]), 1, 0)
age_num    <- ifelse(grepl('45|60', df[['2、年龄：']]), 1, 0)
edu_num    <- ifelse(grepl('本科|研究生', df[['3、教育程度：']]), 1, 0)
inc_num    <- ifelse(grepl('30-50|50-100|>100', df[['8、家庭年均收入情况：']]), 1,
              ifelse(grepl('不便', df[['8、家庭年均收入情况：']]), NA, 0))

# Sub-dataset excluding NA income (N=51)
d_model1 <- data.frame(y=y, edu=edu_num, inc=inc_num, age=age_num, gender=gender_num)
d_model1 <- na.omit(d_model1)

cat(sprintf("--- MODEL 1: N = %d (Complete Case, Refused Income excluded) ---\n", nrow(d_model1)))
cat(sprintf("Events (y=1): %d, Non-events (y=0): %d\n", sum(d_model1$y==1), sum(d_model1$y==0)))

# Standard Logistic Regression (GLM)
m_glm <- glm(y ~ edu + inc + age + gender, data=d_model1, family=binomial)
glm_sum <- summary(m_glm)
glm_ci  <- exp(confint.default(m_glm))
glm_or  <- exp(coef(m_glm))
glm_p   <- glm_sum$coefficients[,4]

cat("\n[Standard GLM Logistic Regression]\n")
res_glm <- data.frame(
  Coef = coef(m_glm),
  SE = glm_sum$coefficients[,2],
  OR = glm_or,
  CI_2.5 = glm_ci[,1],
  CI_97.5 = glm_ci[,2],
  P_Wald = glm_p
)
print(round(res_glm, 4))

# Firth Penalized Logistic Regression (logistf)
m_firth <- logistf(y ~ edu + inc + age + gender, data=d_model1)
firth_sum <- summary(m_firth)
firth_or <- exp(coef(m_firth))
firth_ci <- exp(confint(m_firth))

cat("\n[Firth's Penalized Likelihood Logistic Regression (logistf)]\n")
res_firth <- data.frame(
  Coef = coef(m_firth),
  SE = sqrt(diag(vcov(m_firth))),
  OR = firth_or,
  CI_2.5 = firth_ci[,1],
  CI_97.5 = firth_ci[,2],
  P_PLRT = m_firth$prob,
  P_Wald = 2 * (1 - pnorm(abs(coef(m_firth) / sqrt(diag(vcov(m_firth))))))
)
print(round(res_firth, 4))

cat("\n--------------------------------------------------------------------\n")
cat("--- MODEL 2: N = 66 (Income treated as <30w vs >=30w, Refused coded as <30w or dummy) ---\n")

inc_num_ref0 <- ifelse(grepl('30-50|50-100|>100', df[['8、家庭年均收入情况：']]), 1, 0)
d_model2 <- data.frame(y=y, edu=edu_num, inc=inc_num_ref0, age=age_num, gender=gender_num)

m_glm2 <- glm(y ~ edu + inc + age + gender, data=d_model2, family=binomial)
glm2_sum <- summary(m_glm2)
glm2_ci  <- exp(confint.default(m_glm2))

m_firth2 <- logistf(y ~ edu + inc + age + gender, data=d_model2)
firth2_ci <- exp(confint(m_firth2))

cat("\n[N=66 Standard GLM]:\n")
print(round(data.frame(
  OR = exp(coef(m_glm2)),
  CI_2.5 = glm2_ci[,1],
  CI_97.5 = glm2_ci[,2],
  P_Wald = glm2_sum$coefficients[,4]
), 4))

cat("\n[N=66 Firth Penalized]:\n")
print(round(data.frame(
  OR = exp(coef(m_firth2)),
  CI_2.5 = firth2_ci[,1],
  CI_97.5 = firth2_ci[,2],
  P_PLRT = m_firth2$prob
), 4))
