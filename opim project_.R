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
library(lubridate)  # for date operations
library(caret)      # for dummyVars (one-hot encoding)

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


###############################################################
# DATA PREPROCESSING & FEATURE ENGINEERING (Implementation)
###############################################################

cat(paste0(rep("=", 70), collapse = ""), "\n\n")
cat("STARTING DATA PREPROCESSING & FEATURE ENGINEERING\n")
cat(paste0(rep("=", 70), collapse = ""), "\n\n")

# Train ve test set'lerini birleştirerek preprocessing yapacağız
# Böylece aynı dönüşümler her iki set'e de uygulanır
train_original <- train
test_original <- test_values

# Test set'ine status_group ekle (NA olarak, sadece birleştirme için)
test_original$status_group <- NA

# date_recorded'ı her iki set'te de character olarak tut (birleştirme için)
# Sonra preprocessing sırasında Date'e çevireceğiz
train_original$date_recorded <- as.character(train_original$date_recorded)
test_original$date_recorded <- as.character(test_original$date_recorded)

# Birleştirilmiş dataset
combined <- bind_rows(
  train_original %>% mutate(set = "train"),
  test_original %>% mutate(set = "test")
)

cat("Combined dataset dimensions:", dim(combined), "\n\n")

###############################
# STEP 1: FEATURE ENGINEERING
# Yeni öznitelikler türetme
###############################

cat("=== STEP 1: FEATURE ENGINEERING ===\n")

# 1a. Temporal Features (Tarih bazlı özellikler)
cat("Creating temporal features...\n")

# date_recorded'ı parse et
combined$date_recorded <- as.Date(combined$date_recorded, format = "%m/%d/%Y")

# Parse edilemeyen tarihleri (NA) train set'indeki median tarihle doldur
if (sum(is.na(combined$date_recorded)) > 0) {
  median_date <- median(combined$date_recorded[combined$set == "train"], na.rm = TRUE)
  combined$date_recorded[is.na(combined$date_recorded)] <- median_date
  cat(" ✓ Filled", sum(is.na(combined$date_recorded) == FALSE & 
                       is.na(as.Date(combined$date_recorded, format = "%m/%d/%Y")) == TRUE),
      "missing dates with median:", as.character(median_date), "\n")
}

# Mevcut yıl (2013 olarak varsayıyoruz, veri setindeki en son yıl)
current_year <- 2013

# Pump yaşı
combined$pump_age <- current_year - combined$construction_year
combined$pump_age[combined$pump_age < 0 | is.na(combined$pump_age)] <- 0

# Kayıt tarihinden özellikler
combined$year_recorded <- year(combined$date_recorded)
combined$month_recorded <- month(combined$date_recorded)
combined$quarter_recorded <- quarter(combined$date_recorded)
combined$day_of_year <- yday(combined$date_recorded)

# Kayıt üzerinden geçen yıl
combined$years_since_recorded <- current_year - combined$year_recorded
combined$years_since_recorded[is.na(combined$years_since_recorded)] <- 0

cat(" ✓ Temporal features created\n")

# 1b. Geographic Features
cat("Creating geographic features...\n")

# GPS kalitesi
combined$gps_quality <- ifelse(combined$gps_height == 0 | combined$longitude == 0, "poor", "good")
combined$gps_quality <- as.factor(combined$gps_quality)

# Ekvatordan uzaklık
combined$distance_to_equator <- abs(combined$latitude)

# Bölge-İlçe kombinasyonu
combined$region_district <- paste(combined$region, combined$district_code, sep = "_")

cat(" ✓ Geographic features created\n")

# 1c. Population and Usage Features
cat("Creating population/usage features...\n")

# Log transformations
combined$log_population <- log1p(combined$population)
combined$log_amount_tsh <- log1p(combined$amount_tsh)

# Binary indicators
combined$has_population <- ifelse(combined$population > 0, 1, 0)
combined$has_amount_tsh <- ifelse(combined$amount_tsh > 0, 1, 0)
combined$has_construction_year <- ifelse(combined$construction_year > 0, 1, 0)
combined$has_gps_height <- ifelse(combined$gps_height > 0, 1, 0)

cat(" ✓ Population/usage features created\n")

# 1d. Categorical Interactions
cat("Creating categorical interaction features...\n")

# Funder-Installer eşleşmesi
combined$funder_installer_match <- ifelse(
  !is.na(combined$funder) & !is.na(combined$installer) &
  tolower(combined$funder) == tolower(combined$installer),
  1, 0
)

# Management-Payment interaction
combined$management_payment <- paste(
  ifelse(is.na(combined$management_group), "Unknown", combined$management_group),
  ifelse(is.na(combined$payment_type), "Unknown", combined$payment_type),
  sep = "_"
)

# Source-Extraction interaction
combined$source_extraction <- paste(
  ifelse(is.na(combined$source_class), "Unknown", combined$source_class),
  ifelse(is.na(combined$extraction_type_class), "Unknown", combined$extraction_type_class),
  sep = "_"
)

cat(" ✓ Categorical interaction features created\n")

# 1e. Missing Value Indicators
cat("Creating missing value indicators...\n")

combined$has_scheme_name <- ifelse(is.na(combined$scheme_name), 0, 1)
combined$has_scheme_management <- ifelse(is.na(combined$scheme_management), 0, 1)
combined$has_installer <- ifelse(is.na(combined$installer), 0, 1)
combined$has_funder <- ifelse(is.na(combined$funder), 0, 1)
combined$has_public_meeting <- ifelse(is.na(combined$public_meeting), 0, 1)
combined$has_permit <- ifelse(is.na(combined$permit), 0, 1)

cat(" ✓ Missing value indicators created\n")

cat("Feature engineering completed!\n\n")

###############################
###############################
# STEP 2: FEATURE SELECTION
# Gereksiz sütunları temizleme
###############################

cat("=== STEP 2: FEATURE SELECTION ===\n")

# Elenecek feature'lar (Feature Selection analizinden)
features_to_remove <- c(
  "id",                      # Identifier, predictive değil
  "date_recorded",           # Raw Date column; we use derived temporal features instead
  "wpt_name",                # Çok yüksek kardinalite
  "subvillage",              # Çok yüksek kardinalite
  "scheme_name",             # Çok yüksek kardinalite ve yüksek missing
  "recorded_by",             # Neredeyse sabit
  "extraction_type_group",   # Redundant (extraction_type_class daha spesifik)
  "management_group",        # Redundant (management daha detaylı)
  "quality_group",           # Redundant (water_quality daha detaylı)
  "quantity_group",          # Redundant (quantity daha detaylı)
  "source_type",             # Redundant (source_class daha kategorik)
  "waterpoint_type_group",   # Redundant (waterpoint_type daha detaylı)
  "payment",                 # Redundant (payment_type daha spesifik)
  "num_private"              # %99 zero, düşük varyans
)

# Var olan feature'ları kontrol et
features_to_remove <- features_to_remove[features_to_remove %in% names(combined)]

cat("Removing", length(features_to_remove), "features:\n")
print(features_to_remove)

# Feature'ları çıkar
combined_cleaned <- combined %>%
  select(-all_of(features_to_remove))

cat("\nFeatures removed. New dimensions:", dim(combined_cleaned), "\n\n")

###############################
# STEP 3: MISSING VALUE IMPUTATION
# NA değerlerini doldurma stratejileri
###############################

cat("=== STEP 3: MISSING VALUE IMPUTATION ===\n")

# 3a. Numeric değişkenler için imputation
cat("Imputing numeric variables...\n")

numeric_cols <- names(combined_cleaned)[sapply(combined_cleaned, is.numeric)]
numeric_cols <- setdiff(numeric_cols, c("set"))  # set değişkenini çıkar

# Her numeric değişken için median ile doldur
for (col in numeric_cols) {
  if (sum(is.na(combined_cleaned[[col]])) > 0) {
    median_val <- median(combined_cleaned[[col]], na.rm = TRUE)
    na_count_before <- sum(is.na(combined_cleaned[[col]]))
    combined_cleaned[[col]][is.na(combined_cleaned[[col]])] <- median_val
    cat(" ✓", col, "- filled", na_count_before, "values with median:", median_val, "\n")
  }
}

# 3b. Categorical değişkenler için imputation
cat("\nImputing categorical variables...\n")

categorical_cols <- names(combined_cleaned)[sapply(combined_cleaned, function(x) 
  is.character(x) || is.factor(x))]
categorical_cols <- setdiff(categorical_cols, c("set", "status_group"))

for (col in categorical_cols) {
  if (sum(is.na(combined_cleaned[[col]])) > 0) {
    # En sık görülen değerle doldur (mode)
    mode_val <- names(sort(table(combined_cleaned[[col]]), decreasing = TRUE))[1]
    na_count_before <- sum(is.na(combined_cleaned[[col]]))
    combined_cleaned[[col]][is.na(combined_cleaned[[col]])] <- mode_val
    cat(" ✓", col, "- filled", na_count_before, "values with mode:", mode_val, "\n")
  }
}

# 3c. Logical değişkenler için imputation
cat("\nImputing logical variables...\n")

logical_cols <- names(combined_cleaned)[sapply(combined_cleaned, is.logical)]
logical_cols <- setdiff(logical_cols, c("set"))

for (col in logical_cols) {
  if (sum(is.na(combined_cleaned[[col]])) > 0) {
    # FALSE ile doldur (en yaygın değer genellikle FALSE)
    na_count_before <- sum(is.na(combined_cleaned[[col]]))
    combined_cleaned[[col]][is.na(combined_cleaned[[col]])] <- FALSE
    cat(" ✓", col, "- filled", na_count_before, "values with FALSE\n")
  }
}

# Missing value kontrolü
remaining_na <- sum(is.na(combined_cleaned))
cat("\nRemaining NA values:", remaining_na, "\n")

if (remaining_na > 0) {
  na_by_col <- colSums(is.na(combined_cleaned))
  cat("NA by column:\n")
  print(na_by_col[na_by_col > 0])
}

cat("\nMissing value imputation completed!\n\n")

###############################
# STEP 4: CATEGORICAL ENCODING
# Kategorik verileri sayısal hale getirme
###############################

cat("=== STEP 4: CATEGORICAL ENCODING ===\n")

# 4a. Yüksek kardinaliteli değişkenler için Label Encoding
cat("Label encoding for high cardinality variables...\n")

high_card_vars <- c(
  "funder", "installer", "ward", "lga", "region", "basin",
  "extraction_type", "extraction_type_class", "management",
  "payment_type", "water_quality", "quantity",
  "source", "source_class", "waterpoint_type",
  "region_district", "management_payment", "source_extraction"
)

# Sadece var olan değişkenleri al
high_card_vars <- high_card_vars[high_card_vars %in% names(combined_cleaned)]

# Label encoding için mapping oluştur
label_encodings <- list()

for (var in high_card_vars) {
  if (is.character(combined_cleaned[[var]]) || is.factor(combined_cleaned[[var]])) {
    unique_vals <- unique(combined_cleaned[[var]])
    mapping <- setNames(1:length(unique_vals), unique_vals)
    label_encodings[[var]] <- mapping
    
    # Encode et
    combined_cleaned[[paste0(var, "_encoded")]] <- as.numeric(
      factor(combined_cleaned[[var]], levels = names(mapping))
    )
    cat(" ✓", var, "- encoded to", paste0(var, "_encoded"), "\n")
  }
}

 # 4b. Düşük kardinaliteli değişkenler için One-Hot Encoding
cat("\nOne-hot encoding for low cardinality variables...\n")

low_card_vars <- c(
  "public_meeting", "permit", "gps_quality",
  "scheme_management"  # manageable (13 levels) -> one-hot
)

# Sadece var olan değişkenleri al
low_card_vars <- low_card_vars[low_card_vars %in% names(combined_cleaned)]

# Keep truly low/medium-cardinality vars (including scheme_management)
low_card_vars <- low_card_vars[sapply(low_card_vars, function(v) {
  length(unique(combined_cleaned[[v]])) <= 20
})]

# Ensure they are treated as categorical for dummyVars
if (length(low_card_vars) > 0) {
  combined_cleaned <- combined_cleaned %>%
    mutate(across(all_of(low_card_vars), ~ as.factor(.x)))
}

# One-hot encoding
if (length(low_card_vars) > 0) {
  # dummyVars kullanarak one-hot encoding
  dummies <- dummyVars(~ ., data = combined_cleaned %>% select(all_of(low_card_vars)))
  encoded_df <- predict(dummies, newdata = combined_cleaned)
  
  # Orijinal değişkenleri çıkar ve encoded'ları ekle
  combined_cleaned <- combined_cleaned %>%
    select(-all_of(low_card_vars)) %>%
    bind_cols(as.data.frame(encoded_df))
  
  cat(" ✓ One-hot encoded", length(low_card_vars), "variables\n")
}

# 4c. Orijinal kategorik değişkenleri çıkar (encoded versiyonları kullanacağız)
cat("\nRemoving original categorical variables (keeping encoded versions)...\n")

# Encoded olanların orijinallerini çıkar
vars_to_remove_after_encoding <- high_card_vars
vars_to_remove_after_encoding <- vars_to_remove_after_encoding[
  vars_to_remove_after_encoding %in% names(combined_cleaned)
]
# Also remove scheme_management original if it still exists (after one-hot it should not, but keep safe)
vars_to_remove_after_encoding <- unique(c(vars_to_remove_after_encoding, "scheme_management"))
vars_to_remove_after_encoding <- vars_to_remove_after_encoding[vars_to_remove_after_encoding %in% names(combined_cleaned)]

combined_cleaned <- combined_cleaned %>%
  select(-all_of(vars_to_remove_after_encoding))

cat("Removed", length(vars_to_remove_after_encoding), "original categorical variables\n")

cat("\nCategorical encoding completed!\n\n")

###############################
# STEP 5: FINAL DATA PREPARATION
# Model için temiz matris oluşturma
###############################

cat("=== STEP 5: FINAL DATA PREPARATION ===\n")

# 5a. Train ve test set'lerini ayır
train_processed <- combined_cleaned %>%
  filter(set == "train") %>%
  select(-set)
# Drop raw date column; we use derived temporal features instead
if ("date_recorded" %in% names(train_processed)) {
  train_processed <- train_processed %>% select(-date_recorded)
}

test_processed <- combined_cleaned %>%
  filter(set == "test") %>%
  select(-set, -status_group)  # Test set'inde status_group yok
# Drop raw date column; we use derived temporal features instead
if ("date_recorded" %in% names(test_processed)) {
  test_processed <- test_processed %>% select(-date_recorded)
}

# 5b. Status group'u factor yap (train için)
train_processed$status_group <- as.factor(train_processed$status_group)

# 5c. Son kontrol: NA, Inf, NaN değerleri
cat("Final data quality check...\n")

# Inf ve NaN değerlerini kontrol et ve düzelt
numeric_cols_final <- names(train_processed)[sapply(train_processed, is.numeric)]
numeric_cols_final <- setdiff(numeric_cols_final, "status_group")

for (col in numeric_cols_final) {
  # Inf değerlerini NA'ya çevir, sonra median ile doldur
  if (any(is.infinite(train_processed[[col]]), na.rm = TRUE)) {
    train_processed[[col]][is.infinite(train_processed[[col]])] <- NA
    median_val <- median(train_processed[[col]], na.rm = TRUE)
    train_processed[[col]][is.na(train_processed[[col]])] <- median_val
  }
  if (any(is.infinite(test_processed[[col]]), na.rm = TRUE)) {
    test_processed[[col]][is.infinite(test_processed[[col]])] <- NA
    median_val <- median(test_processed[[col]], na.rm = TRUE)
    test_processed[[col]][is.na(test_processed[[col]])] <- median_val
  }
  
  # NaN değerlerini düzelt
  if (any(is.nan(train_processed[[col]]), na.rm = TRUE)) {
    median_val <- median(train_processed[[col]], na.rm = TRUE)
    train_processed[[col]][is.nan(train_processed[[col]])] <- median_val
  }
  if (any(is.nan(test_processed[[col]]), na.rm = TRUE)) {
    median_val <- median(test_processed[[col]], na.rm = TRUE)
    test_processed[[col]][is.nan(test_processed[[col]])] <- median_val
  }
}

# Date sütunlarındaki NA'ları düzelt
date_cols <- names(train_processed)[sapply(train_processed, function(x) inherits(x, "Date"))]

for (col in date_cols) {
  if (col %in% names(test_processed)) {
    if (sum(is.na(test_processed[[col]])) > 0) {
      median_date <- median(train_processed[[col]], na.rm = TRUE)
      na_count_before <- sum(is.na(test_processed[[col]]))
      test_processed[[col]][is.na(test_processed[[col]])] <- median_date
      cat(" ✓ Filled", na_count_before, "missing", col, "values in test set\n")
    }
  }
}

# Tüm kalan NA'ları kontrol et ve düzelt
if (sum(is.na(test_processed)) > 0) {
  for (col in names(test_processed)) {
    if (sum(is.na(test_processed[[col]])) > 0) {
      if (is.numeric(test_processed[[col]])) {
        median_val <- median(train_processed[[col]], na.rm = TRUE)
        test_processed[[col]][is.na(test_processed[[col]])] <- median_val
      } else if (is.character(test_processed[[col]]) || is.factor(test_processed[[col]])) {
        mode_val <- names(sort(table(train_processed[[col]]), decreasing = TRUE))[1]
        test_processed[[col]][is.na(test_processed[[col]])] <- mode_val
      }
    }
  }
}

# 5d. Sonuç özeti
cat("\n=== PREPROCESSING SUMMARY ===\n")
cat("Train set dimensions:", dim(train_processed), "\n")
cat("Test set dimensions:", dim(test_processed), "\n")
cat("Number of features (excluding target):", ncol(train_processed) - 1, "\n")
cat("Target variable:", "status_group", "\n")
cat("Target classes:", levels(train_processed$status_group), "\n")

# Feature listesi
cat("\nFinal feature list:\n")
feature_names <- setdiff(names(train_processed), "status_group")
cat(paste(feature_names, collapse = ", "), "\n")

# Sanity check: ensure all predictors are numeric
non_numeric_predictors <- feature_names[!sapply(train_processed[feature_names], is.numeric)]
if (length(non_numeric_predictors) > 0) {
  cat("\nWARNING: Non-numeric predictors remain:\n")
  print(non_numeric_predictors)
} else {
  cat("\nAll predictors are numeric.\n")
}

# Data quality metrics
cat("\nData Quality Metrics:\n")
cat(" - NA values in train:", sum(is.na(train_processed)), "\n")
cat(" - NA values in test:", sum(is.na(test_processed)), "\n")
cat(" - Inf values in train:", sum(sapply(train_processed[numeric_cols_final], function(x) sum(is.infinite(x), na.rm = TRUE))), "\n")
cat(" - Inf values in test:", sum(sapply(test_processed[numeric_cols_final], function(x) sum(is.infinite(x), na.rm = TRUE))), "\n")

cat("\n")
cat(paste0(rep("=", 70), collapse = ""), "\n")
cat("DATA PREPROCESSING & FEATURE ENGINEERING COMPLETED!\n")
cat("Data is now ready for machine learning models.\n")
cat(paste0(rep("=", 70), collapse = ""), "\n\n")

# Final datasets hazır:
# - train_processed: Model eğitimi için hazır train set
# - test_processed: Tahmin için hazır test set


