# ============================================================
# PUMP IT UP: TARGET 85%+ ACCURACY - V5
# Fixes: XGBoost, RF balanced sampling removed, geo-clustering,
#        installer cleaning, subvillage rates, optimized ensemble
# ============================================================

rm(list = ls())
gc()

library(readr)
library(caret)
library(randomForest)
library(xgboost)
library(gbm)
library(dplyr)
library(readr)
library(lubridate)

cat("============================================================\n")
cat("PUMP IT UP: TARGET 85%+ ACCURACY - V5\n")
cat("============================================================\n\n")

set.seed(1975)

# ============================================================
# 1) LOAD DATA
# ============================================================

train_values <- read_csv("/Users/ardabitcher/Documents/University/opim390/Pump it up/Pump_it_Up_Data_Mining_the_Water_Table_-_Training_set_values.csv", show_col_types = FALSE)
train_labels <- read_csv("/Users/ardabitcher/Documents/University/opim390/Pump it up/Pump_it_Up_Data_Mining_the_Water_Table_-_Training_set_labels.csv", show_col_types = FALSE)
test_values <- read_csv("/Users/ardabitcher/Documents/University/opim390/Pump it up/Pump_it_Up_Data_Mining_the_Water_Table_-_Test_set_values.csv", show_col_types = FALSE)

train <- train_values %>% left_join(train_labels, by = "id")

cat("Full data dimensions:", nrow(train), "x", ncol(train), "\n\n")

# ============================================================
# 2) COMPREHENSIVE FEATURE ENGINEERING
# ============================================================

cat("=== FEATURE ENGINEERING ===\n")

test_values$status_group <- NA

# Ensure date_recorded is character in both before binding
train$date_recorded <- as.character(train$date_recorded)
test_values$date_recorded <- as.character(test_values$date_recorded)

combined <- bind_rows(
    train %>% mutate(set = "train"),
    test_values %>% mutate(set = "test")
)

# --- Store medians from training data only ---
train_data <- combined %>% filter(set == "train")

med_longitude <- median(train_data$longitude[train_data$longitude != 0], na.rm = TRUE)
med_latitude <- median(train_data$latitude[train_data$latitude < -0.5], na.rm = TRUE)
med_gps_height <- median(train_data$gps_height[train_data$gps_height != 0], na.rm = TRUE)
med_population <- median(train_data$population[train_data$population > 0], na.rm = TRUE)
med_construction_year <- median(train_data$construction_year[train_data$construction_year > 0], na.rm = TRUE)

# --- Date parsing ---
# Data is in YYYY-MM-DD format. We remove the hardcoded format to let R detect it.
combined$date_recorded <- as.Date(combined$date_recorded)

# If standard ISO conversion fails, fallback to US format
if (mean(is.na(combined$date_recorded)) > 0.5) {
    combined$date_recorded <- as.Date(as.character(combined$date_recorded), format = "%m/%d/%Y")
}

# Fill any remaining NAs with the median date
if (sum(is.na(combined$date_recorded)) > 0) {
    combined$date_recorded[is.na(combined$date_recorded)] <- as.Date("2013-01-01")
}

# --- Temporal features ---
combined$year_recorded <- year(combined$date_recorded)
combined$month_recorded <- month(combined$date_recorded)

# --- SEASONALITY (Tanzania dry/rainy seasons) ---
cat("--- FE: SEASONALITY ---\n")
combined$season_long_rain <- as.integer(combined$month_recorded %in% c(3, 4, 5))
combined$season_short_rain <- as.integer(combined$month_recorded %in% c(10, 11, 12))
combined$season_dry <- as.integer(!combined$month_recorded %in% c(3, 4, 5, 10, 11, 12))

# --- GPS quality flags (BEFORE imputation) ---
combined$gps_missing <- as.integer(combined$longitude == 0 | combined$gps_height == 0 | combined$latitude > -0.5)

# --- Impute zeros ---
combined$longitude[combined$longitude == 0] <- med_longitude
combined$latitude[combined$latitude > -0.5] <- med_latitude
combined$gps_height[combined$gps_height == 0] <- med_gps_height
combined$population[combined$population == 0] <- med_population
combined$construction_year[combined$construction_year == 0] <- med_construction_year

# --- Age feature ---
combined$pump_age <- 2013 - combined$construction_year
combined$pump_age[combined$pump_age < 0 | combined$pump_age > 100] <- median(combined$pump_age[combined$pump_age > 0 & combined$pump_age < 100], na.rm = TRUE)

# Age buckets
combined$age_bucket <- cut(combined$pump_age,
    breaks = c(-Inf, 5, 10, 15, 20, 25, 30, Inf),
    labels = 1:7
)
combined$age_bucket <- as.numeric(combined$age_bucket)

# --- Population features ---
combined$log_population <- log1p(combined$population)
combined$pop_bucket <- cut(combined$population,
    breaks = c(-Inf, 25, 100, 250, 500, 1000, 5000, Inf),
    labels = 1:7
)
combined$pop_bucket <- as.numeric(combined$pop_bucket)

# --- Amount TSH features ---
combined$tsh_missing <- as.integer(combined$amount_tsh == 0)
combined$log_amount_tsh <- log1p(combined$amount_tsh)
combined$has_tsh <- as.integer(combined$amount_tsh > 0)

# --- Geographic features ---
combined$dist_dar <- sqrt((combined$latitude - (-6.79))^2 + (combined$longitude - 39.21)^2)
combined$dist_dodoma <- sqrt((combined$latitude - (-6.16))^2 + (combined$longitude - 35.75)^2)
combined$dist_mwanza <- sqrt((combined$latitude - (-2.52))^2 + (combined$longitude - 32.92)^2)
combined$dist_arusha <- sqrt((combined$latitude - (-3.37))^2 + (combined$longitude - 36.68)^2)
combined$min_city_dist <- pmin(combined$dist_dar, combined$dist_dodoma, combined$dist_mwanza, combined$dist_arusha)

# Elevation categories
combined$elev_bucket <- cut(combined$gps_height,
    breaks = c(-Inf, 0, 300, 600, 1000, 1500, 2000, Inf),
    labels = 1:7
)
combined$elev_bucket <- as.numeric(combined$elev_bucket)

# ============================================================
# SPATIAL CLUSTERING (K-Means Neighborhoods)
# ============================================================
cat("--- FE: SPATIAL CLUSTERING ---\n")

coords <- combined[, c("longitude", "latitude")]
set.seed(1975)
kmeans_fit <- kmeans(coords, centers = 200, nstart = 25, iter.max = 50)
combined$geo_cluster <- kmeans_fit$cluster

# Calculate cluster stats from training data only
cluster_stats <- combined %>%
    filter(set == "train") %>%
    group_by(geo_cluster) %>%
    summarise(
        cluster_func_rate = mean(status_group == "functional", na.rm = TRUE),
        cluster_nonfunc_rate = mean(status_group == "non functional", na.rm = TRUE),
        cluster_gps_mean = mean(gps_height, na.rm = TRUE),
        cluster_count = n(),
        .groups = "drop"
    )

combined <- combined %>% left_join(cluster_stats, by = "geo_cluster")
combined$cluster_func_rate[is.na(combined$cluster_func_rate)] <- 0.54
combined$cluster_nonfunc_rate[is.na(combined$cluster_nonfunc_rate)] <- 0.38
combined$cluster_gps_mean[is.na(combined$cluster_gps_mean)] <- med_gps_height
combined$cluster_count[is.na(combined$cluster_count)] <- 1

# Height deviation from cluster mean
combined$height_deviation <- combined$gps_height - combined$cluster_gps_mean

# ============================================================
# TEXT CLEANING (Installer & Funder)
# ============================================================
cat("--- FE: TEXT CLEANING ---\n")

clean_text_col <- function(x) {
    x <- tolower(as.character(x))
    x[is.na(x) | x == "" | x == "0"] <- "unknown"
    x[grepl("gov|central|ministry|council|district", x)] <- "government"
    x[grepl("commu|vill|wanan", x)] <- "community"
    x[grepl("church|roman|catholic|missi|kkkt|lutheran", x)] <- "religious"
    x[grepl("world bank|unicef|hesawa|rwssp|undp|dhv|dwsp|danida|norad|jica", x)] <- "international_aid"
    x[grepl("priva|ltd|comp", x)] <- "private"
    return(x)
}

combined$installer_clean <- clean_text_col(combined$installer)
combined$funder_clean <- clean_text_col(combined$funder)

# Frequency encoding
installer_counts <- combined %>% count(installer_clean, name = "installer_freq")
funder_counts <- combined %>% count(funder_clean, name = "funder_freq")

combined <- combined %>% left_join(installer_counts, by = "installer_clean")
combined <- combined %>% left_join(funder_counts, by = "funder_clean")

# Top installers/funders encoding
top_installers <- combined %>%
    count(installer_clean) %>%
    slice_max(n, n = 20) %>%
    pull(installer_clean)
top_funders <- combined %>%
    count(funder_clean) %>%
    slice_max(n, n = 20) %>%
    pull(funder_clean)

combined$installer_cat <- ifelse(combined$installer_clean %in% top_installers, combined$installer_clean, "other")
combined$funder_cat <- ifelse(combined$funder_clean %in% top_funders, combined$funder_clean, "other")

combined$installer_cat_enc <- as.numeric(factor(combined$installer_cat))
combined$funder_cat_enc <- as.numeric(factor(combined$funder_cat))

# ============================================================
# SUBVILLAGE STATS (Smoothed)
# ============================================================
cat("--- FE: SUBVILLAGE STATS ---\n")

subvillage_stats <- combined %>%
    filter(set == "train") %>%
    group_by(subvillage) %>%
    summarise(
        subvillage_count = n(),
        subvillage_func_rate = (sum(status_group == "functional") + 1) / (n() + 2),
        .groups = "drop"
    )

combined <- combined %>% left_join(subvillage_stats, by = "subvillage")
combined$subvillage_func_rate[is.na(combined$subvillage_func_rate)] <- 0.54
combined$subvillage_count[is.na(combined$subvillage_count)] <- 0

# --- Quality and quantity flags (VERY PREDICTIVE) ---
combined$quantity_enough <- as.integer(combined$quantity == "enough")
combined$quantity_insufficient <- as.integer(combined$quantity == "insufficient")
combined$quantity_dry <- as.integer(combined$quantity == "dry")
combined$quantity_seasonal <- as.integer(combined$quantity == "seasonal")

combined$quality_soft <- as.integer(combined$water_quality == "soft")
combined$quality_salty <- as.integer(grepl("salty", combined$water_quality))
combined$quality_unknown <- as.integer(combined$water_quality == "unknown")

# --- Payment features ---
combined$never_pay <- as.integer(combined$payment_type == "never pay")
combined$pay_per_bucket <- as.integer(combined$payment_type == "per bucket")
combined$pay_monthly <- as.integer(combined$payment_type == "monthly")

# --- Extraction type ---
combined$extract_gravity <- as.integer(grepl("gravity", combined$extraction_type))
combined$extract_handpump <- as.integer(grepl("handpump|nira|afridev|india|swn|mono", combined$extraction_type))
combined$extract_motor <- as.integer(grepl("motor|submersible", combined$extraction_type))
combined$extract_rope <- as.integer(grepl("rope", combined$extraction_type))

# --- Waterpoint type ---
combined$wp_communal <- as.integer(grepl("communal", combined$waterpoint_type))
combined$wp_hand_pump <- as.integer(grepl("hand pump", combined$waterpoint_type))
combined$wp_other <- as.integer(combined$waterpoint_type == "other")

# --- Management ---
combined$mgmt_vwc <- as.integer(combined$management == "vwc")
combined$mgmt_wug <- as.integer(combined$management == "wug")
combined$mgmt_private <- as.integer(grepl("private", combined$management))

# --- Source features ---
combined$source_spring <- as.integer(combined$source == "spring")
combined$source_shallow_well <- as.integer(combined$source == "shallow well")
combined$source_river <- as.integer(grepl("river", combined$source))

# --- Boolean features ---
combined$public_meeting[is.na(combined$public_meeting)] <- FALSE
combined$permit[is.na(combined$permit)] <- FALSE
combined$public_meeting_num <- as.integer(combined$public_meeting)
combined$permit_num <- as.integer(combined$permit)

# --- REGION FAILURE RATE ---
region_stats <- train_data %>%
    group_by(region) %>%
    summarise(
        region_functional_rate = mean(status_group == "functional", na.rm = TRUE),
        region_nonfunc_rate = mean(status_group == "non functional", na.rm = TRUE),
        region_count = n(),
        .groups = "drop"
    )

combined <- combined %>% left_join(region_stats, by = "region")
combined$region_functional_rate[is.na(combined$region_functional_rate)] <- 0.54
combined$region_nonfunc_rate[is.na(combined$region_nonfunc_rate)] <- 0.38

# --- BASIN FAILURE RATE ---
basin_stats <- train_data %>%
    group_by(basin) %>%
    summarise(
        basin_functional_rate = mean(status_group == "functional", na.rm = TRUE),
        basin_nonfunc_rate = mean(status_group == "non functional", na.rm = TRUE),
        .groups = "drop"
    )

combined <- combined %>% left_join(basin_stats, by = "basin")
combined$basin_functional_rate[is.na(combined$basin_functional_rate)] <- 0.54
combined$basin_nonfunc_rate[is.na(combined$basin_nonfunc_rate)] <- 0.38

# --- LGA FAILURE RATE ---
lga_stats <- train_data %>%
    group_by(lga) %>%
    summarise(
        lga_functional_rate = mean(status_group == "functional", na.rm = TRUE),
        lga_count = n(),
        .groups = "drop"
    )

combined <- combined %>% left_join(lga_stats, by = "lga")
combined$lga_functional_rate[is.na(combined$lga_functional_rate)] <- 0.54
combined$lga_count[is.na(combined$lga_count)] <- 1

# --- WARD FAILURE RATE ---
ward_stats <- train_data %>%
    group_by(ward) %>%
    summarise(
        ward_functional_rate = mean(status_group == "functional", na.rm = TRUE),
        ward_count = n(),
        .groups = "drop"
    )

combined <- combined %>% left_join(ward_stats, by = "ward")
combined$ward_functional_rate[is.na(combined$ward_functional_rate)] <- 0.54
combined$ward_count[is.na(combined$ward_count)] <- 1
combined$ward_functional_rate <- ifelse(combined$ward_count < 10, 0.54, combined$ward_functional_rate)

# --- EXTRACTION TYPE FAILURE RATE ---
extraction_stats <- train_data %>%
    group_by(extraction_type) %>%
    summarise(
        extraction_functional_rate = mean(status_group == "functional", na.rm = TRUE),
        .groups = "drop"
    )

combined <- combined %>% left_join(extraction_stats, by = "extraction_type")
combined$extraction_functional_rate[is.na(combined$extraction_functional_rate)] <- 0.54

# --- INTERACTION FEATURES ---
combined$age_x_dry <- combined$pump_age * combined$quantity_dry
combined$age_x_season_dry <- combined$pump_age * combined$season_dry
combined$pop_density_proxy <- combined$population / (combined$min_city_dist + 0.1)

# --- LABEL ENCODING for categorical ---
cat_cols <- c(
    "region", "basin", "lga", "ward", "extraction_type", "extraction_type_class",
    "management", "payment_type", "water_quality", "quantity", "source",
    "waterpoint_type", "scheme_management"
)

for (col in cat_cols) {
    if (col %in% names(combined)) {
        combined[[col]][is.na(combined[[col]])] <- "Unknown"
        combined[[col]][combined[[col]] == ""] <- "Unknown"
        combined[[paste0(col, "_enc")]] <- as.numeric(factor(combined[[col]]))
    }
}

cat("Feature engineering complete!\n\n")

# ============================================================
# 3) SELECT FEATURES
# ============================================================

cat("=== SELECTING FEATURES ===\n")

cols_to_remove <- c(
    "id", "set", "status_group", "date_recorded",
    "wpt_name", "subvillage", "scheme_name", "recorded_by", "funder", "installer",
    "num_private", "region_code", "district_code",
    "region", "basin", "lga", "ward", "extraction_type", "extraction_type_class",
    "management", "payment_type", "water_quality", "quantity", "source",
    "waterpoint_type", "scheme_management",
    "extraction_type_group", "management_group", "quality_group", "quantity_group",
    "source_type", "source_class", "waterpoint_type_group", "payment",
    "public_meeting", "permit",
    "installer_clean", "funder_clean", "installer_cat", "funder_cat"
)

train_final <- combined %>%
    filter(set == "train") %>%
    select(-any_of(cols_to_remove))
test_final <- combined %>%
    filter(set == "test") %>%
    select(-any_of(cols_to_remove))

train_final$status_group <- gsub(" ", "_", train$status_group)
train_final$status_group <- factor(train_final$status_group,
    levels = c("functional", "functional_needs_repair", "non_functional")
)

# Clean up NAs
for (col in names(train_final)) {
    if (col != "status_group" && is.numeric(train_final[[col]])) {
        med <- median(train_final[[col]], na.rm = TRUE)
        if (is.na(med)) med <- 0
        train_final[[col]][is.na(train_final[[col]])] <- med
        train_final[[col]][is.infinite(train_final[[col]])] <- med
        if (col %in% names(test_final)) {
            test_final[[col]][is.na(test_final[[col]])] <- med
            test_final[[col]][is.infinite(test_final[[col]])] <- med
        }
    }
}

cat("Train dimensions:", dim(train_final), "\n")
cat("Test dimensions:", dim(test_final), "\n")
cat("Features:", ncol(train_final) - 1, "\n\n")

# ============================================================
# 4) TRAIN-VALIDATION SPLIT
# ============================================================

cat("=== TRAIN-VALIDATION SPLIT ===\n")

set.seed(1975)
train_idx <- createDataPartition(y = train_final$status_group, p = 0.85, list = FALSE)

training <- train_final[train_idx, ]
validation <- train_final[-train_idx, ]

cat("Training:", nrow(training), "\n")
cat("Validation:", nrow(validation), "\n\n")

# ============================================================
# 5) RANDOM FOREST (NO class balancing - it hurts accuracy)
# ============================================================

cat("=== TRAINING RANDOM FOREST ===\n")

set.seed(1975)
rf_model <- randomForest(
    status_group ~ .,
    data = training,
    ntree = 1000,
    mtry = 15,
    importance = TRUE,
    do.trace = 100
)

rf_pred <- predict(rf_model, newdata = validation)
rf_cm <- confusionMatrix(rf_pred, validation$status_group)
cat("\nRandom Forest Validation Accuracy:", round(rf_cm$overall["Accuracy"], 4), "\n")

# ============================================================
# 6) XGBOOST (FIXED parameters)
# ============================================================

cat("\n=== TRAINING XGBOOST ===\n")

feature_cols <- setdiff(names(training), "status_group")
train_matrix <- as.matrix(training[, feature_cols])
val_matrix <- as.matrix(validation[, feature_cols])
test_matrix <- as.matrix(test_final[, feature_cols])

train_label <- as.numeric(training$status_group) - 1
val_label <- as.numeric(validation$status_group) - 1

dtrain <- xgb.DMatrix(data = train_matrix, label = train_label)
dval <- xgb.DMatrix(data = val_matrix, label = val_label)
dtest <- xgb.DMatrix(data = test_matrix)

# FIXED: Conservative parameters that actually work
params <- list(
    objective = "multi:softmax",
    num_class = 3,
    eval_metric = "merror",
    eta = 0.05,
    max_depth = 8,
    subsample = 0.8,
    colsample_bytree = 0.8,
    min_child_weight = 1,
    gamma = 0.1
)

set.seed(1975)
xgb_model <- xgb.train(
    params = params,
    data = dtrain,
    nrounds = 1000,
    evals = list(train = dtrain, val = dval),
    early_stopping_rounds = 50,
    print_every_n = 50,
    verbose = 1
)

best_nrounds <- xgb_model$best_iteration
if (is.null(best_nrounds) || best_nrounds == 0) best_nrounds <- 500
cat("Best XGBoost iteration:", best_nrounds, "\n")

xgb_pred_num <- predict(xgb_model, dval)
xgb_pred <- factor(xgb_pred_num + 1, levels = 1:3, labels = levels(validation$status_group))
xgb_cm <- confusionMatrix(xgb_pred, validation$status_group)
cat("\nXGBoost Validation Accuracy:", round(xgb_cm$overall["Accuracy"], 4), "\n")

# ============================================================
# 7) GBM
# ============================================================

cat("\n=== TRAINING GBM ===\n")

set.seed(1975)
gbm_model <- gbm(
    status_group ~ .,
    data = training,
    distribution = "multinomial",
    n.trees = 500,
    interaction.depth = 10,
    shrinkage = 0.05,
    n.minobsinnode = 15,
    bag.fraction = 0.8,
    cv.folds = 5,
    verbose = TRUE
)

best_iter <- gbm.perf(gbm_model, method = "cv", plot.it = FALSE)
cat("Best GBM iteration:", best_iter, "\n")

gbm_probs <- predict(gbm_model, newdata = validation, n.trees = best_iter, type = "response")
gbm_pred_num <- apply(gbm_probs, 1, which.max)
gbm_pred <- factor(gbm_pred_num, levels = 1:3, labels = levels(validation$status_group))
gbm_cm <- confusionMatrix(gbm_pred, validation$status_group)
cat("GBM Validation Accuracy:", round(gbm_cm$overall["Accuracy"], 4), "\n")

# ============================================================
# 8) OPTIMIZED ENSEMBLE (Grid Search for Best Weights)
# ============================================================

cat("\n=== OPTIMIZED ENSEMBLE ===\n")

# Get probabilities from RF
rf_probs <- predict(rf_model, newdata = validation, type = "prob")

# Get probabilities from XGBoost (need softprob model)
params_prob <- list(
    objective = "multi:softprob",
    num_class = 3,
    eval_metric = "mlogloss",
    eta = 0.05,
    max_depth = 8,
    subsample = 0.8,
    colsample_bytree = 0.8,
    min_child_weight = 1,
    gamma = 0.1
)

set.seed(1975)
xgb_model_prob <- xgb.train(
    params = params_prob,
    data = dtrain,
    nrounds = best_nrounds,
    verbose = 0
)

xgb_probs_raw <- predict(xgb_model_prob, dval)
xgb_probs <- matrix(xgb_probs_raw, ncol = 3, byrow = TRUE)
colnames(xgb_probs) <- levels(validation$status_group)

# GBM probabilities
gbm_probs_df <- as.data.frame(gbm_probs[, , 1])
colnames(gbm_probs_df) <- levels(validation$status_group)

# Grid search for optimal weights
cat("Searching for optimal ensemble weights...\n")
best_acc <- 0
best_w <- c(0.5, 0.3, 0.2)

for (w1 in seq(0.2, 0.7, 0.1)) {
    for (w2 in seq(0.1, 0.6, 0.1)) {
        w3 <- 1 - w1 - w2
        if (w3 >= 0.1 && w3 <= 0.5) {
            blended_probs <- w1 * as.matrix(rf_probs) + w2 * xgb_probs + w3 * as.matrix(gbm_probs_df)
            pred_idx <- apply(blended_probs, 1, which.max)
            pred_class <- factor(pred_idx, levels = 1:3, labels = levels(validation$status_group))
            acc <- mean(pred_class == validation$status_group)
            if (acc > best_acc) {
                best_acc <- acc
                best_w <- c(w1, w2, w3)
            }
        }
    }
}

cat("Best Weights (RF, XGB, GBM):", best_w, "\n")
cat("Optimized Ensemble Accuracy:", round(best_acc, 4), "\n")

# Final ensemble with best weights
weights <- c(rf = best_w[1], xgb = best_w[2], gbm = best_w[3])
ensemble_probs <- weights["rf"] * as.matrix(rf_probs) +
    weights["xgb"] * xgb_probs +
    weights["gbm"] * as.matrix(gbm_probs_df)

ensemble_pred_num <- apply(ensemble_probs, 1, which.max)
ensemble_pred <- factor(ensemble_pred_num, levels = 1:3, labels = levels(validation$status_group))
ensemble_cm <- confusionMatrix(ensemble_pred, validation$status_group)

# ============================================================
# 9) RESULTS
# ============================================================

cat("\n========== FINAL RESULTS ==========\n")

results <- data.frame(
    Model = c("Random Forest", "XGBoost", "GBM", "Optimized Ensemble"),
    Accuracy = c(
        rf_cm$overall["Accuracy"],
        xgb_cm$overall["Accuracy"],
        gbm_cm$overall["Accuracy"],
        ensemble_cm$overall["Accuracy"]
    ),
    Kappa = c(
        rf_cm$overall["Kappa"],
        xgb_cm$overall["Kappa"],
        gbm_cm$overall["Kappa"],
        ensemble_cm$overall["Kappa"]
    )
)
results <- results %>% arrange(desc(Accuracy))
print(results)

best_model <- results$Model[1]
best_accuracy <- results$Accuracy[1]

cat("\n========================================\n")
cat("BEST MODEL:", best_model, "\n")
cat("VALIDATION ACCURACY:", round(best_accuracy * 100, 2), "%\n")
cat("========================================\n")

# ============================================================
# 10) GENERATE SUBMISSION
# ============================================================

cat("\n=== GENERATING SUBMISSION ===\n")

# Use the best performing approach
rf_test_probs <- predict(rf_model, newdata = test_final, type = "prob")

xgb_test_probs_raw <- predict(xgb_model_prob, dtest)
xgb_test_probs <- matrix(xgb_test_probs_raw, ncol = 3, byrow = TRUE)

gbm_test_probs <- predict(gbm_model, newdata = test_final, n.trees = best_iter, type = "response")
gbm_test_probs_df <- as.data.frame(gbm_test_probs[, , 1])

final_probs <- weights["rf"] * as.matrix(rf_test_probs) +
    weights["xgb"] * xgb_test_probs +
    weights["gbm"] * as.matrix(gbm_test_probs_df)

final_pred_num <- apply(final_probs, 1, which.max)
final_pred <- levels(training$status_group)[final_pred_num]
final_pred_labels <- gsub("_", " ", final_pred)

cat("Prediction distribution:\n")
print(table(final_pred_labels))

submission <- data.frame(
    id = test_values$id,
    status_group = final_pred_labels
)

write.csv(submission, "submission_v5_85_target.csv", row.names = FALSE)

cat("\n========================================\n")
cat("SUBMISSION SAVED: submission_v5_85_target.csv\n")
cat("========================================\n")

saveRDS(list(
    rf_model = rf_model,
    xgb_model = xgb_model,
    xgb_model_prob = xgb_model_prob,
    gbm_model = gbm_model,
    best_iter = best_iter,
    weights = weights,
    results = results
), "models_v5.rds")

cat("\nModels saved to: models_v5.rds\n")
