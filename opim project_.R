# ============================================================
# PUMP IT UP: MODEL BUILDING & ASSESSMENT
# Person 3: Model Builder
# ============================================================

# uncomment next lines if packages not installed yet
# install.packages("caret")
# install.packages("randomForest")
# install.packages("xgboost")
# install.packages("rpart")
# install.packages("rpart.plot")
# install.packages("dplyr")
# install.packages("ggplot2")
# install.packages("readr")

# load necessary libraries
library(caret)
library(randomForest)
library(xgboost)
library(rpart)
library(rpart.plot)
library(dplyr)
library(ggplot2)
library(readr)

# ---------------------------
# 1) Load Data
# ---------------------------

# read in files
train_values = read_csv("train_values.csv", show_col_types = FALSE)
train_labels = read_csv("train_labels.csv", show_col_types = FALSE)
test_values  = read_csv("test_values.csv", show_col_types = FALSE)

# merge training data with labels
df = train_values %>%
  left_join(train_labels, by = "id")

dim(df)
str(df)

# target variable distribution
table(df$status_group)
prop.table(table(df$status_group))

# ---------------------------
# 2) Data Preprocessing (from Person 2)
# ---------------------------

# NOTE: This section should come from Person 2's preprocessing script
# For now, implementing basic preprocessing so models can run

# drop high cardinality columns
cols_to_drop = c("id", "recorded_by", "wpt_name", "subvillage", 
                 "ward", "scheme_name", "installer", "funder",
                 "quantity_group", "source_type", "source_class",
                 "waterpoint_type_group", "payment", "extraction_type_group",
                 "extraction_type_class", "management_group", "region_code",
                 "lga", "scheme_management", "extraction_type")

df = df %>% select(-any_of(cols_to_drop))

# handle zeros (missing values)
df$longitude[df$longitude == 0] = median(df$longitude[df$longitude != 0], na.rm = TRUE)
df$gps_height[df$gps_height == 0] = median(df$gps_height[df$gps_height != 0], na.rm = TRUE)
df$construction_year[df$construction_year == 0] = median(df$construction_year[df$construction_year != 0], na.rm = TRUE)
df$population[df$population == 0] = median(df$population[df$population != 0], na.rm = TRUE)

# create pump age feature
df$pump_age = 2024 - df$construction_year

# log transform skewed features
df$log_amount_tsh = log1p(df$amount_tsh)
df$log_population = log1p(df$population)

# convert character columns to factors (limit levels)
char_cols = names(df)[sapply(df, is.character)]
for (col in char_cols) {
  if (col != "status_group") {
    top_levels = names(sort(table(df[[col]]), decreasing = TRUE))[1:15]
    df[[col]] = ifelse(df[[col]] %in% top_levels, df[[col]], "Other")
    df[[col]] = as.factor(df[[col]])
  }
}

# drop original columns
df = df %>% select(-amount_tsh, -construction_year, -population)

# encode target as factor
df$status_group = factor(df$status_group,
                         levels = c("functional", "functional needs repair", "non functional"))

str(df)
dim(df)

# ---------------------------
# 3) Train-Validation Split
# ---------------------------

set.seed(1975)

# stratified sampling
indxTrain = createDataPartition(y = df$status_group, p = 0.8, list = FALSE)

training = df[indxTrain, ]
testing = df[-indxTrain, ]

# verify proportions
prop.table(table(training$status_group))
prop.table(table(testing$status_group))

dim(training)
dim(testing)

# ---------------------------
# 4) Cross-Validation Setup
# ---------------------------

# 5-fold CV for faster training
ctrl = trainControl(
  method = "cv",
  number = 5,
  classProbs = TRUE,
  summaryFunction = multiClassSummary,
  verboseIter = TRUE
)

# ============================================================
# MODEL 1: DECISION TREE (Baseline)
# ============================================================

cp_grid = expand.grid(cp = seq(0.001, 0.02, length.out = 10))

set.seed(1975)
treeFit = train(
  status_group ~ .,
  data = training,
  method = "rpart",
  trControl = ctrl,
  tuneGrid = cp_grid,
  metric = "Accuracy"
)

treeFit$bestTune
treeFit

# visualize tree
rpart.plot(treeFit$finalModel)
rpart.rules(treeFit$finalModel)

# evaluate on test set
treePred = predict(treeFit, newdata = testing)
tree_cm = confusionMatrix(treePred, testing$status_group)
tree_cm

# ============================================================
# MODEL 2: RANDOM FOREST
# ============================================================

mtry_grid = expand.grid(mtry = c(3, 5, 7, 10))

set.seed(1975)
rfFit = train(
  status_group ~ .,
  data = training,
  method = "rf",
  trControl = ctrl,
  tuneGrid = mtry_grid,
  ntree = 200,
  importance = TRUE,
  metric = "Accuracy"
)

rfFit$bestTune
rfFit

# plot accuracy vs mtry
plot(rfFit)

# variable importance
varImp(rfFit, scale = FALSE)

# evaluate on test set
rfPred = predict(rfFit, newdata = testing)
rf_cm = confusionMatrix(rfPred, testing$status_group)
rf_cm

# ============================================================
# MODEL 3: XGBOOST
# ============================================================

xgb_grid = expand.grid(
  nrounds = c(100, 150),
  max_depth = c(4, 6, 8),
  eta = c(0.1, 0.3),
  gamma = 0,
  colsample_bytree = 0.8,
  min_child_weight = c(1, 3),
  subsample = 0.8
)

set.seed(1975)
xgbFit = train(
  status_group ~ .,
  data = training,
  method = "xgbTree",
  trControl = ctrl,
  tuneGrid = xgb_grid,
  metric = "Accuracy",
  verbose = FALSE
)

xgbFit$bestTune
xgbFit

# evaluate on test set
xgbPred = predict(xgbFit, newdata = testing)
xgb_cm = confusionMatrix(xgbPred, testing$status_group)
xgb_cm

# ============================================================
# MODEL 4: KNN
# ============================================================

knn_grid = expand.grid(k = seq(3, 25, by = 2))

set.seed(1975)
knnFit = train(
  status_group ~ .,
  data = training,
  method = "knn",
  trControl = ctrl,
  tuneGrid = knn_grid,
  preProcess = c("center", "scale"),
  metric = "Accuracy"
)

knnFit$bestTune
knnFit

plot(knnFit)

# evaluate on test set
knnPred = predict(knnFit, newdata = testing)
knn_cm = confusionMatrix(knnPred, testing$status_group)
knn_cm

# ============================================================
# MODEL COMPARISON
# ============================================================

# compare cross-validation results
resamps = resamples(list(
  DecisionTree = treeFit,
  RandomForest = rfFit,
  XGBoost = xgbFit,
  KNN = knnFit
))

summary(resamps, metric = "Accuracy")

# boxplot comparison
bwplot(resamps, metric = "Accuracy")

# dotplot comparison
dotplot(resamps, metric = "Accuracy")

# collect test set results
results = data.frame(
  Model = c("Decision Tree", "Random Forest", "XGBoost", "KNN"),
  Accuracy = c(
    tree_cm$overall["Accuracy"],
    rf_cm$overall["Accuracy"],
    xgb_cm$overall["Accuracy"],
    knn_cm$overall["Accuracy"]
  ),
  Kappa = c(
    tree_cm$overall["Kappa"],
    rf_cm$overall["Kappa"],
    xgb_cm$overall["Kappa"],
    knn_cm$overall["Kappa"]
  )
)

results = results %>% arrange(desc(Accuracy))
results

# best model
best_model = results$Model[1]
best_acc = results$Accuracy[1]

cat("\n========================================\n")
cat("BEST MODEL:", best_model, "\n")
cat("TEST ACCURACY:", round(best_acc, 4), "\n")
cat("========================================\n")

# ---------------------------
# Class-wise Performance Analysis
# ---------------------------

# show confusion matrix of best model
if (best_model == "XGBoost") {
  best_cm = xgb_cm
} else if (best_model == "Random Forest") {
  best_cm = rf_cm
} else if (best_model == "KNN") {
  best_cm = knn_cm
} else {
  best_cm = tree_cm
}

best_cm$table

# class-wise metrics
class_metrics = data.frame(
  Class = rownames(best_cm$byClass),
  Sensitivity = round(best_cm$byClass[, "Sensitivity"], 4),
  Specificity = round(best_cm$byClass[, "Specificity"], 4),
  Precision = round(best_cm$byClass[, "Precision"], 4),
  F1 = round(best_cm$byClass[, "F1"], 4)
)
class_metrics

# NOTE: "functional needs repair" class typically has lower performance
# due to class imbalance (only ~7% of data)

# ---------------------------
# Generate Test Predictions for Person 4
# ---------------------------

# store training medians for imputation
train_median_lon = median(train_values$longitude[train_values$longitude != 0], na.rm = TRUE)
train_median_height = median(train_values$gps_height[train_values$gps_height != 0], na.rm = TRUE)
train_median_year = median(train_values$construction_year[train_values$construction_year != 0], na.rm = TRUE)
train_median_pop = median(train_values$population[train_values$population != 0], na.rm = TRUE)

# apply same preprocessing to test data
test_processed = test_values %>% select(-any_of(cols_to_drop))

# impute zeros with training medians
test_processed$longitude[test_processed$longitude == 0] = train_median_lon
test_processed$gps_height[test_processed$gps_height == 0] = train_median_height
test_processed$construction_year[test_processed$construction_year == 0] = train_median_year
test_processed$population[test_processed$population == 0] = train_median_pop

# create features
test_processed$pump_age = 2024 - test_processed$construction_year
test_processed$log_amount_tsh = log1p(test_processed$amount_tsh)
test_processed$log_population = log1p(test_processed$population)

# convert character to factors with same levels as training
test_char_cols = names(test_processed)[sapply(test_processed, is.character)]
for (col in test_char_cols) {
  if (col %in% names(training)) {
    # map unseen levels to "Other"
    test_processed[[col]] = ifelse(test_processed[[col]] %in% levels(training[[col]]), 
                                    test_processed[[col]], "Other")
    test_processed[[col]] = factor(test_processed[[col]], levels = levels(training[[col]]))
  }
}

# drop original columns
test_processed = test_processed %>% select(-amount_tsh, -construction_year, -population)

# verify test data has same columns as training (except target)
train_cols = setdiff(names(training), "status_group")
test_cols = names(test_processed)
cat("Training features:", length(train_cols), "\n")
cat("Test features:", length(test_cols), "\n")

# check for any NA values
na_count = sum(is.na(test_processed))
cat("NA values in test set:", na_count, "\n")

# generate predictions with best model
if (best_model == "XGBoost") {
  final_pred = predict(xgbFit, newdata = test_processed)
} else if (best_model == "Random Forest") {
  final_pred = predict(rfFit, newdata = test_processed)
} else if (best_model == "KNN") {
  final_pred = predict(knnFit, newdata = test_processed)
} else {
  final_pred = predict(treeFit, newdata = test_processed)
}

# prediction distribution
table(final_pred)
prop.table(table(final_pred))

# create submission file for Person 4
submission = data.frame(
  id = test_values$id,
  status_group = as.character(final_pred)
)

head(submission, 10)
write.csv(submission, "submission.csv", row.names = FALSE)

# save best model for Person 4
if (best_model == "XGBoost") {
  saveRDS(xgbFit, "best_model.rds")
} else if (best_model == "Random Forest") {
  saveRDS(rfFit, "best_model.rds")
} else if (best_model == "KNN") {
  saveRDS(knnFit, "best_model.rds")
} else {
  saveRDS(treeFit, "best_model.rds")
}

# save results summary
saveRDS(results, "model_comparison_results.rds")

cat("\nFiles saved for Person 4:\n")
cat("  - submission.csv\n")
cat("  - best_model.rds\n")
cat("  - model_comparison_results.rds\n")
