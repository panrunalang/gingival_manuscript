library(logistf)

df <- read.csv('results/survey.csv', check.names=FALSE, stringsAsFactors=FALSE)

cat("====================================================================\n")
cat("  PART 1: FIGURE 3 SPARSE CONTINGENCY TABLES & EXACT TESTS (N=66)\n")
cat("====================================================================\n\n")

anxiety_raw <- df[['11、术前是否焦虑甚至恐惧？']]
postop_raw  <- df[['15、术后反应如何？']]
psych_raw   <- df[['18、心理负担会直接影响您的选择：']]
fsurg_raw   <- df[['20、本次手术治疗是否会影响到您今后的方案选择']]

anxiety <- factor(ifelse(grepl('完全没有', anxiety_raw), 'No anxiety',
                  ifelse(grepl('轻度', anxiety_raw), 'Mild anxiety',
                  ifelse(grepl('中度', anxiety_raw), 'Moderate anxiety', 'Other'))),
                  levels=c('No anxiety', 'Mild anxiety', 'Moderate anxiety'))

postop <- factor(ifelse(grepl('无特殊不适', postop_raw), 'No discomfort',
                 ifelse(grepl('轻微', postop_raw), 'Mild symptoms',
                 ifelse(grepl('明显', postop_raw), 'Marked discomfort', 'Other'))),
                 levels=c('No discomfort', 'Mild symptoms', 'Marked discomfort'))

psych <- factor(ifelse(grepl('是', psych_raw), 'Yes', 'No'), levels=c('Yes', 'No'))

fsurg <- factor(ifelse(grepl('完全不影响', fsurg_raw), 'No influence',
                ifelse(grepl('轻微影响', fsurg_raw), 'Mild influence',
                ifelse(grepl('比较影响', fsurg_raw), 'Moderate influence', 'Other'))),
                levels=c('No influence', 'Mild influence', 'Moderate influence'))

# Panel A
cat("--------------------------------------------------------------------\n")
cat("Panel A: Preoperative Anxiety (3) vs Postoperative Discomfort (3)\n")
cat("--------------------------------------------------------------------\n")
tab_a <- table(anxiety, postop)
print(tab_a)
cat("\nExpected counts:\n")
chisq_a <- suppressWarnings(chisq.test(tab_a))
print(round(chisq_a$expected, 2))
fisher_a <- fisher.test(tab_a)
set.seed(2026)
mc_a <- fisher.test(tab_a, simulate.p.value=TRUE, B=1000000)

cat(sprintf("Pearson Chi-Square: X2 = %.4f (df=%d), P = %.4f (manuscript reported P = 0.008)\n", 
            chisq_a$statistic, chisq_a$parameter, chisq_a$p.value))
cat(sprintf("Fisher-Freeman-Halton Exact Test: P = %.4f\n", fisher_a$p.value))
cat(sprintf("Monte Carlo Exact Test (B=1,000,000): P = %.4f\n\n", mc_a$p.value))

# Panel B
cat("--------------------------------------------------------------------\n")
cat("Panel B: Preoperative Anxiety (3) vs Psychological Burden (2)\n")
cat("--------------------------------------------------------------------\n")
tab_b <- table(anxiety, psych)
print(tab_b)
cat("\nExpected counts:\n")
chisq_b <- suppressWarnings(chisq.test(tab_b))
print(round(chisq_b$expected, 2))
fisher_b <- fisher.test(tab_b)
set.seed(2026)
mc_b <- fisher.test(tab_b, simulate.p.value=TRUE, B=1000000)

cat(sprintf("Pearson Chi-Square: X2 = %.4f (df=%d), P = %.4f (manuscript reported P = 0.001)\n", 
            chisq_b$statistic, chisq_b$parameter, chisq_b$p.value))
cat(sprintf("Fisher-Freeman-Halton Exact Test: P = %.4f\n", fisher_b$p.value))
cat(sprintf("Monte Carlo Exact Test (B=1,000,000): P = %.4f\n\n", mc_b$p.value))

# Panel C
cat("--------------------------------------------------------------------\n")
cat("Panel C: Postoperative Discomfort (3) vs Future Surgery Willingness (3)\n")
cat("--------------------------------------------------------------------\n")
tab_c <- table(postop, fsurg)
print(tab_c)
cat("\nExpected counts:\n")
chisq_c <- suppressWarnings(chisq.test(tab_c))
print(round(chisq_c$expected, 2))
fisher_c <- fisher.test(tab_c)
set.seed(2026)
mc_c <- fisher.test(tab_c, simulate.p.value=TRUE, B=1000000)

cat(sprintf("Pearson Chi-Square: X2 = %.4f (df=%d), P = %.4e (manuscript reported P < 0.001)\n", 
            chisq_c$statistic, chisq_c$parameter, chisq_c$p.value))
cat(sprintf("Fisher-Freeman-Halton Exact Test: P = %.4e (%.5f)\n", fisher_c$p.value, fisher_c$p.value))
cat(sprintf("Monte Carlo Exact Test (B=1,000,000): P = %.4e (%.5f)\n\n", mc_c$p.value, mc_c$p.value))
