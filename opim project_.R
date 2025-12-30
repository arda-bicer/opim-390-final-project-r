##################
#Exploratory Data Analysis (EDA)

# Working directory'yi script'in bulunduğu klasöre ayarla
setwd("/Users/almiraaygun/Downloads/pumpitup-opim390/opim-390-final-project-r")
getwd()
list.files()

# loading the necessary libraries
library(readr)
library(dplyr)
library(ggplot2)
library(tidyr)

# optional: run the heavier global association tests (ANOVA + chi-square for all features)
RUN_FULL_ASSOCIATION <- TRUE

train_values <- read_csv("train_values.csv", show_col_types = FALSE)
train_labels <- read_csv("train_labels.csv", show_col_types = FALSE)
test_values  <- read_csv("test_values.csv",  show_col_types = FALSE)
sub_format   <- read_csv("submission_format.csv", show_col_types = FALSE)

dim(train_values); dim(train_labels); dim(test_values); dim(sub_format)

#merging the two data sets: train values and train labels
train <- train_values %>%
  left_join(train_labels, by = "id")

#check of merged data
nrow(train)                      # must be the same as train_values
sum(is.na(train$status_group))   # must be 0

table(train$status_group)
prop.table(table(train$status_group))

#name of the merged data is train. 


##################
# Step 1: Dataset overview

# Basic dimensions
dim(train)

# Column names
names(train)

# Quick structure check
glimpse(train)

##################
# Step 2: Target distribution

# Count and proportion tables
target_counts <- table(train$status_group)
target_props  <- prop.table(target_counts)

target_counts
round(target_props, 4)

ggplot(train, aes(x = status_group)) +
  geom_bar() +
  labs(
    title = "Distribution of Target Variable (status_group)",
    x = "Status Group",
    y = "Count"
  ) +
  theme(axis.text.x = element_text(angle = 15, hjust = 1))

# Insight:
# The target variable is imbalanced. The "functional needs repair" class is much smaller
# than the other two classes, which may affect model performance and should be considered
# during validation.



##################
# Step 3: Missing values (NA)


# NA count per column
na_count <- colSums(is.na(train))

# NA percentage per column
na_pct <- round(na_count / nrow(train) * 100, 2)

# Combine into a table
na_summary <- data.frame(
  column = names(na_count),
  na_count = as.integer(na_count),
  na_pct = as.numeric(na_pct)
) %>%
  arrange(desc(na_count))

# Show top 10 columns with most missing values
head(na_summary, 10)

top10_na <- na_summary %>% slice(1:10)

ggplot(top10_na, aes(x = reorder(column, na_count), y = na_count)) +
  geom_col() +
  coord_flip() +
  labs(
    title = "Top 10 Columns with Missing Values",
    x = "Column",
    y = "Number of Missing Values"
  )

# Insight:
# Missing values are not evenly distributed across features.
# Columns with high missingness may require special imputation strategies
# or could be excluded depending on their usefulness.

###################
# Step 4: Obvious anomalies / suspicious zeros

# We check the share of zero values across numeric columns.
# In this dataset, zeros can sometimes indicate missing/unknown values (e.g., construction_year = 0).


num_cols <- names(train)[sapply(train, is.numeric)]

zero_rate_tbl <- data.frame(
  column = num_cols,
  zero_count = sapply(train[num_cols], function(x) sum(x == 0, na.rm = TRUE)),
  zero_pct = round(sapply(train[num_cols], function(x) mean(x == 0, na.rm = TRUE)) * 100, 2)
) %>%
  arrange(desc(zero_pct))

# Show top 10 numeric columns with the highest zero share
head(zero_rate_tbl, 10)

# Insight (Step 4):
# Several numeric features have many zeros (amount_tsh ~70%; population/construction_year/gps_height ~35%).
# These zeros may represent unknown/missing values rather than true zeros, so they may need special handling
# during preprocessing. Longitude has ~3% zeros (likely missing/invalid GPS), while num_private is ~99% zeros
# and is probably a genuine distribution.

#####################
# Step 5: Numeric distributions
# Goal:
#   - Inspect the distribution of key numeric features
#   - Check skewness, outliers, and the impact of many zeros
#   - Compare raw scale vs log-transformed scale (log1p)

# amount_tsh
ggplot(train, aes(x = amount_tsh)) +
  geom_histogram() +
  labs(title = "Histogram: amount_tsh", x = "amount_tsh", y = "Count")

ggplot(train, aes(x = log1p(amount_tsh))) +
  geom_histogram() +
  labs(title = "Histogram: log1p(amount_tsh)", x = "log1p(amount_tsh)", y = "Count")

# population
ggplot(train, aes(x = population)) +
  geom_histogram() +
  labs(title = "Histogram: population", x = "population", y = "Count")

ggplot(train, aes(x = log1p(population))) +
  geom_histogram() +
  labs(title = "Histogram: log1p(population)", x = "log1p(population)", y = "Count")

# gps_height
ggplot(train, aes(x = gps_height)) +
  geom_histogram() +
  labs(title = "Histogram: gps_height", x = "gps_height", y = "Count")

#####################
# Step 6: Numeric features vs target (status_group)

ggplot(train, aes(x = status_group, y = log1p(amount_tsh), fill = status_group)) +
  geom_boxplot(outlier.alpha = 0.2) +
  labs(title = "log1p(amount_tsh) by status_group") +
  theme(axis.text.x = element_text(angle = 15, hjust = 1))

ggplot(train, aes(x = status_group, y = log1p(population), fill = status_group)) +
  geom_boxplot(outlier.alpha = 0.2) +
  labs(title = "log1p(population) by status_group") +
  theme(axis.text.x = element_text(angle = 15, hjust = 1))

ggplot(train, aes(x = status_group, y = gps_height, fill = status_group)) +
  geom_boxplot(outlier.alpha = 0.2) +
  labs(title = "gps_height by status_group") +
  theme(axis.text.x = element_text(angle = 15, hjust = 1))

##########################
# Step 7: Categorical features vs target (status_group)

# Goal:
#   - Compare category distributions across status_group
#   - Identify categorical variables that strongly separate classes
# Note:
#   - We plot proportions (position = "fill") to compare class composition within each category.


cat_vars <- c(
  "quantity_group",
  "water_quality",
  "source_class",
  "extraction_type_class",
  "management_group",
  "payment_type",
  "waterpoint_type_group"
)
cat_vars <- cat_vars[cat_vars %in% names(train)]

for (v in cat_vars) {
  p <- ggplot(train, aes(x = .data[[v]], fill = status_group)) +
    geom_bar(position = "fill") +
    labs(
      title = paste("status_group proportion by", v),
      x = v,
      y = "Proportion"
    ) +
    theme(axis.text.x = element_text(angle = 30, hjust = 1))
  
  print(p)
}

###################
# Step 7b: Rank categorical variables by association with target (chi-square)


chi_tbl <- lapply(cat_vars, function(v) {
  tab <- table(train[[v]], train$status_group)
  test <- suppressWarnings(chisq.test(tab))
  data.frame(
    variable = v,
    p_value = test$p.value,
    chi_square = unname(test$statistic)
  )
}) %>% bind_rows() %>%
  arrange(p_value)

chi_tbl


###############################
# FULL FEATURE ASSOCIATION CHECK
###############################
if (RUN_FULL_ASSOCIATION) {

# 1️⃣ Prepare data
train_full <- train %>%
  mutate(across(where(is.logical), ~ ifelse(is.na(.x), "Unknown",
                                            ifelse(.x, "TRUE", "FALSE"))))

# 2️⃣ Separate feature types
num_cols <- names(train_full)[sapply(train_full, is.numeric)]
cat_cols <- names(train_full)[sapply(train_full, function(x) is.character(x) || is.factor(x))]
cat_cols <- setdiff(cat_cols, "status_group")  # remove target

#######################
# 3️⃣ Numeric Variables - ANOVA
#######################
num_results <- lapply(num_cols, function(v) {
  f <- as.formula(paste(v, "~ status_group"))
  p_value <- suppressWarnings(summary(aov(f, data = train_full))[[1]][["Pr(>F)"]][1])
  data.frame(variable = v, p_value = p_value)
}) %>% bind_rows() %>%
  arrange(p_value) %>%
  mutate(strength = case_when(
    p_value < 0.001 ~ "very strong",
    p_value < 0.01 ~ "strong",
    p_value < 0.05 ~ "moderate",
    TRUE ~ "weak"
  ))

#######################
# 4️⃣ Categorical Variables - Chi-Square
#######################
cat_results <- lapply(cat_cols, function(v) {
  tab <- table(train_full[[v]], train_full$status_group)
  test <- suppressWarnings(chisq.test(tab))
  data.frame(variable = v, chi_square = unname(test$statistic), p_value = test$p.value)
}) %>%
  bind_rows() %>%
  arrange(desc(chi_square)) %>%
  mutate(strength = case_when(
    p_value < 0.001 ~ "very strong",
    p_value < 0.01 ~ "strong",
    p_value < 0.05 ~ "moderate",
    TRUE ~ "weak"
  ))

#######################
# 5️⃣ Output Summary
#######################

print(num_results)
print(cat_results)
}


###############################################################
# FEATURE SELECTION ANALYSIS (clean, no re-loading)
###############################################################

# Prepare for selection analysis
train_full <- train %>%
  mutate(across(where(is.logical), ~ factor(ifelse(is.na(.x), "Unknown",
                                                  ifelse(.x, "TRUE", "FALSE")))))

num_cols_fs <- names(train_full)[sapply(train_full, is.numeric)]
num_cols_fs <- setdiff(num_cols_fs, c("id"))

cat_cols_fs <- names(train_full)[sapply(train_full, function(x) is.character(x) || is.factor(x))]
cat_cols_fs <- setdiff(cat_cols_fs, "status_group")

# 1) High cardinality analysis
cardinality_analysis <- data.frame(
  variable = cat_cols_fs,
  unique_count = sapply(cat_cols_fs, function(v) length(unique(train_full[[v]])))
) %>%
  arrange(desc(unique_count)) %>%
  mutate(category = case_when(
    unique_count > 1000 ~ "Very High (>1000)",
    unique_count > 100 ~ "High (100-1000)",
    unique_count > 20 ~ "Medium (20-100)",
    TRUE ~ "Low (<20)"
  ))

head(cardinality_analysis, 30)

high_card_vars <- cardinality_analysis %>%
  filter(unique_count > 100) %>%
  pull(variable)

# 2) Redundant numeric features (high correlation pairs)
numeric_for_corr <- train_full %>% select(all_of(num_cols_fs))
cor_matrix <- cor(numeric_for_corr, use = "complete.obs")

high_corr_pairs <- data.frame(var1 = character(), var2 = character(), correlation = numeric())
for (i in 1:(nrow(cor_matrix) - 1)) {
  for (j in (i + 1):ncol(cor_matrix)) {
    cor_val <- cor_matrix[i, j]
    if (!is.na(cor_val) && abs(cor_val) > 0.8) {
      high_corr_pairs <- rbind(
        high_corr_pairs,
        data.frame(
          var1 = rownames(cor_matrix)[i],
          var2 = colnames(cor_matrix)[j],
          correlation = round(cor_val, 4)
        )
      )
    }
  }
}

high_corr_pairs %>% arrange(desc(abs(correlation)))

# 3) Grouped / redundant categorical variables (report only)
grouped_vars <- list(
  extraction = c("extraction_type", "extraction_type_group", "extraction_type_class"),
  management = c("management", "management_group"),
  payment = c("payment", "payment_type"),
  quality = c("water_quality", "quality_group"),
  quantity = c("quantity", "quantity_group"),
  source = c("source", "source_type", "source_class"),
  waterpoint = c("waterpoint_type", "waterpoint_type_group")
)

for (group_name in names(grouped_vars)) {
  existing_vars <- grouped_vars[[group_name]][grouped_vars[[group_name]] %in% names(train_full)]
  if (length(existing_vars) > 1) {
    cat("\n", group_name, "group:\n")
    print(existing_vars)
  }
}

# 4) Low variance numeric
low_variance_vars <- numeric_for_corr %>%
  summarise(across(everything(), ~ var(.x, na.rm = TRUE))) %>%
  pivot_longer(everything(), names_to = "variable", values_to = "variance") %>%
  filter(variance < 0.01 | is.na(variance)) %>%
  arrange(variance)

low_variance_vars

# 5) Final removal list (analysis-based)
features_to_remove <- list(
  id = c("id"),
  high_cardinality = high_card_vars,
  low_variance = low_variance_vars$variable,
  redundant_groups = c(
    "extraction_type_group", "management_group", "quality_group",
    "quantity_group", "source_type", "waterpoint_type_group"
  ),
  redundant_payment = c("payment"),
  near_constant = c("recorded_by")
)

all_features_to_remove <- unique(unlist(features_to_remove))
all_features_to_remove <- all_features_to_remove[all_features_to_remove %in% names(train_full)]

cat("\nFEATURES TO REMOVE:\n")
print(all_features_to_remove)

recommended_features <- setdiff(names(train_full), c(all_features_to_remove, "id", "status_group"))
cat("\nRECOMMENDED FEATURES COUNT:", length(recommended_features), "\n")

###############################################################
# FEATURE ENGINEERING RECOMMENDATIONS (Text Only)
###############################################################
cat("\nSuggested engineered features (not implemented):\n")
cat(" - pump_age = current_year - construction_year\n")
cat(" - years_since_recorded from date_recorded\n")
cat(" - gps_quality flag (gps_height==0 or longitude==0)\n")
cat(" - log1p(population), log1p(amount_tsh)\n")
cat(" - missing indicators (has_funder, has_installer, etc.)\n")


