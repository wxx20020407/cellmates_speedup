##################################################################
#
# Script to generate plots from quadruplet_accuracy_X.csv files
#   It will save a PDF file named quadruplet_accuracy_X_plots.pdf
#   with the plots in the same directory as the input file
#
# Usage: Rscript quadruplet_accuracy_plots.r <experiment_file>
#
##################################################################

library(dplyr)
library(ggplot2)
library(tidyr)
library(readr)
library(GGally)
library(forcats)

args <- commandArgs(trailingOnly = TRUE)
# input the quadruplet_accuracy_X.csv file
if (length(args) < 1) {
  stop("Usage: Rscript quadruplet_accuracy_plots.r <experiment_file>")
}
experiment_file <- args[1]
# experiment_file <- "quadruplet_accuracy_250210235309.csv"

# Extract unique string from filename for PDF output
pdf_filename <- sub("\\.csv$", "_plots.pdf", experiment_file)

# read experiment csv
df <- read_csv(experiment_file, col_types = cols(
  seed = col_integer(),
  n_sites = col_factor(levels = c("200", "500", "1000")),
  n_states = col_integer(),
  length_params = col_factor(),
  p_change_u = col_double(),
  p_change_v = col_double(),
  p_change_w = col_double(),
  lu_em = col_double(),
  lv_em = col_double(),
  lw_em = col_double(),
  lu_err = col_double(),
  lv_err = col_double(),
  lw_err = col_double(),
  exec_time = col_double(),
  n_iter = col_integer(),
  loglik = col_double(),
  true_ll = col_double(),
  variance = col_double(),
  obs_model = col_factor(),
  obs_var = col_double()
))

df <- df %>%
  mutate(lu_true = lu_em + lu_err, lv_true = lv_em + lv_err, lw_true = lw_em + lw_err) %>%
  mutate(err_percent_u = abs(lu_err) / lu_true * 100, err_percent_v = abs(lv_err) / lv_true * 100, err_percent_w = abs(lw_err) / lw_true * 100) %>%
  mutate(greater_loglik = loglik > true_ll) %>%
  mutate(loglik_diff = loglik - true_ll)

# Open PDF device
pdf(pdf_filename, width = 10, height = 6)

# Plot 1: Error for u edge length
p1 <- df %>%
  mutate(p_change_u = factor(round(p_change_u,digits=3))) %>%
  mutate(p_change_vw = factor(round((p_change_v + p_change_w)/2,digits=3))) %>%
  ggplot(aes(x = p_change_u, y = abs(lu_err) / lu_true * 100)) +
  geom_boxplot(aes(fill=p_change_vw)) +
  facet_wrap(~n_sites, labeller=label_both) +
  ylim(0, 100) +
  labs(title = "Error (%) for u edge length",
       x = "p_change mean from r to u",
       y = "Abs relative error",
       caption = sprintf("")) +
  theme_minimal()
print(p1)

# Plot 2: Error for v edge length
p2 <- df %>% 
  mutate(p_change_v = factor(round(p_change_v,digits=3))) %>%
  ggplot(aes(x = p_change_v, y = abs(lv_err) / lv_true * 100)) +
  geom_boxplot() +
  facet_wrap(~n_sites, labeller=label_both) +
  ylim(0, 100) +
  labs(title = "Error (%) for v edge length",
       x = "p_change mean from r to v",
       y = "Abs relative error") +
  theme_minimal()
print(p2)

# Plot 3: Error for w edge length
p3 <- df %>%
  mutate(p_change_w = factor(round(p_change_w,digits=3))) %>%
  ggplot(aes(x = p_change_w, y = abs(lw_err) / lw_true * 100)) +
  geom_boxplot() +
  facet_wrap(~n_sites, labeller=label_both) +
  ylim(0, 100) +
  labs(title = "Error (%) for w edge length",
       x = "p_change mean from r to w",
       y = "Abs relative error") +
  theme_minimal()
print(p3)

# Plot 4: Number of iterations
p4 <- df %>%
  ggplot() +
  geom_boxplot(aes(x = length_params, y = n_iter, fill=greater_loglik), outlier.shape = NA, size = 0.3) +
  geom_hline(aes(color="max_iter", yintercept = 60), linetype = "dashed") +
  facet_wrap(~n_sites, ncol=3, labeller=label_both) +
  labs(title = "Iterations to converge",
       x = "Length mean size",
       y = "Number of iterations",
       caption = "5 runs with same length mean. Length mean labels: 's_ru[-s_uv[-s_w]]'
  From 'xs' to 'xl', prob of change: [0.005, 0.01, 0.05, 0.1, 0.2], and length variance = 0.001") +
  theme_minimal()
print(p4)

# Plot 5: Log likelihood
p5 <- df %>%
  mutate(p_change_u = factor(round(p_change_u,digits=3)), p_change_vw = factor(round((p_change_v + p_change_w)/2,digits=3))) %>%
  ggplot() +
  geom_boxplot(aes(x = p_change_u, y = loglik, fill=p_change_vw)) +
  facet_wrap(~n_sites, labeller=label_both) +
  labs(title = "Log likelihood",
       x = "p_change mean from r to u",
       y = "Log likelihood") +
  theme_minimal()
print(p5)

# Plot 6: Pair plot
p6 <- df %>%
  select(n_sites, n_states, p_change_u, p_change_v, p_change_w, err_percent_u, err_percent_v, err_percent_w, loglik) %>%
  ggpairs()
print(p6)

# Plot 7: Likelihood vs length_params
p7 <- df %>%
  ggplot() +
  geom_boxplot(aes(x = length_params, y = loglik - true_ll), outlier.shape = NA, size = 0.3) +
  geom_hline(aes(yintercept = 0), linetype = "dashed") +
  facet_wrap(~n_sites, ncol=3, scales = "free") +
  labs(title = "Log likelihood difference vs length params",
       x = "Length mean size",
       y = "Log likelihood improvement") +
  theme_minimal()
print(p7)

# Plot 8: Log likelihood difference vs p_change
p8 <- df %>%
  gather(key = "edge", value = "p_change", p_change_u, p_change_v, p_change_w) %>%
  ggplot() +
  geom_point(aes(x = p_change, y = loglik - true_ll)) +
  geom_smooth(aes(x = p_change, y = loglik - true_ll), method = "lm", se = FALSE) +
  facet_grid(edge~n_sites, scales = "free", labeller=label_both) +
  labs(title = "Log likelihood difference vs p_change over each edge",
       x = "p_change mean",
       y = "Log likelihood improvement (diff)") +
  theme_minimal()
print(p8)

# Plot 9: Proportion of runs that increased loglik
p9 <- df %>%
  ggplot() +
  geom_bar(aes(x = n_sites, fill = greater_loglik), position = "fill") +
  geom_hline(aes(yintercept = 0.5), linetype = "dashed") +
  labs(title = "Proportion of runs that increased loglik",
       x = "Number of sites",
       y = "Proportion of runs", caption = sprintf("total runs per n_sites category: %s", nrow(df %>% filter(n_sites == "200")))) +
  theme_minimal()
print(p9)

# order by number of runs where loglik decreased
fac <- df %>%
  group_by(length_params) %>%
  summarise(n_runs_decreased = sum(!greater_loglik) / n()) %>%
  ungroup() %>%
  arrange(n_runs_decreased) %>%
  pull(length_params)

p9b <- df %>%
  mutate(length_params = factor(length_params, levels = fac)) %>%
  ggplot() +
  geom_bar(aes(x = length_params, fill = greater_loglik), position = "fill") +
  geom_hline(aes(yintercept = 0.5), linetype = "dashed") +
  labs(title = "Proportion of runs that increased loglik",
       x = "Length mean size",
       y = "Proportion of runs", caption = sprintf("total runs per length_params category: %s", nrow(df %>% filter(length_params == "s_ru")))) +
  theme_minimal()

print(p9b)

# Plot 10: Log likelihood difference by length_params
df_grouped <- df %>%
  group_by(length_params) %>%
  summarise(greater_loglik_med = median(loglik_diff) > 0) %>%
  ungroup() %>%
  left_join(df, by = "length_params")

fac <- with(df_grouped, reorder(length_params, loglik_diff, median, order = TRUE))
p10 <- df_grouped %>%
  mutate(length_params = factor(length_params, levels = levels(fac))) %>%
  ggplot() +
  geom_boxplot(aes(x = length_params, y = loglik_diff, fill = greater_loglik_med)) +
  geom_hline(aes(yintercept = 0), linetype = "dashed") +
  ylim(-10, 10) +
  labs(title = "Log likelihood difference by length_params",
       x = "Length mean size",
       y = "Log likelihood improvement (diff)") +
  theme_minimal()
print(p10)

# Plot 11: Log likelihood difference when loglik increased and decreased
# group by obs params

p11 <- df %>%
  mutate(obs_params = paste(obs_model, obs_var, sep = "_var")) %>%
  ggplot() +
  geom_boxplot(aes(x = length_params, y = loglik_diff, fill = greater_loglik), outlier.shape = NA, size = 0.3) +
  geom_hline(aes(yintercept = 0), linetype = "dashed") +
  facet_wrap(~greater_loglik + obs_params, ncol=3, scales = "free") +
  labs(title = "Log likelihood difference when loglik increased and decreased",
       x = "Length mean size",
       y = "Log likelihood difference",
       caption = "Quantify the likelihood fall in case of failure") +
  theme_minimal()
print(p11)

# Close PDF device
dev.off()
