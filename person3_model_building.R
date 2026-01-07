# ============================================================
# PUMP IT UP: MODEL BUILDING & ASSESSMENT
# Person 3: Arda - Model Builder
# ============================================================

# load libraries
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

setwd("/Users/ardabitcher/Documents/University/opim390/Pump it up/opim-390-final-project-r")

train_values = read_csv("train_values.csv", show_col_types = FALSE)
train_labels = read_csv("train_labels.csv", show_col_types = FALSE)
test_values  = read_csv("test_values.csv", show_col_types = FALSE)

df = train_values %>% left_join(train_labels, by = "id")

dim(df)
table(df$status_group)

# ---------------------------
# 2) Preprocessing
# ---------------------------

# fix target variable names (R doesn't like spaces)
df$status_group = gsub(" ", "_", df$status_group)
df$status_group = factor(df$status_group, 
                         levels = c("functional", "functional_needs_repair", "non_functional"))

# drop high cardinality columns
cols_to_drop = c("id", "recorded_by", "wpt_name", "subvillage", 
                 "ward", "scheme_name", "installer", "funder",
                 "quantity_group", "source_type", "source_class",
                 "waterpoint_type_group", "payment", "extraction_type_group",
                 "extraction_type_class", "management_group", "region_code",
                 "lga", "scheme_management", "extraction_type", "date_recorded")

df = df %>% select(-any_of(cols_to_drop))

# impute zeros with median
df$longitude[df$longitude == 0] = median(df$longitude[df$longitude != 0], na.rm = TRUE)
df$gps_height[df$gps_height == 0] = median(df$gps_height[df$gps_height != 0], na.rm = TRUE)
df$construction_year[df$construction_year == 0] = median(df$construction_year[df$construction_year != 0], na.rm = TRUE)
df$population[df$population == 0] = median(df$population[df$population != 0], na.rm = TRUE)

# feature engineering
df$pump_age = 2024 - df$construction_year
df$log_amount_tsh = log1p(df$amount_tsh)
df$log_population = log1p(df$population)

# handle NA in logical columns
df$public_meeting[is.na(df$public_meeting)] = FALSE
df$permit[is.na(df$permit)] = FALSE
df$public_meeting = factor(df$public_meeting)
df$permit = factor(df$permit)

# convert character to factors (limit levels to top 15)
char_cols = names(df)[sapply(df, is.character)]
for (col in char_cols) {
  top_levels = names(sort(table(df[[col]]), decreasing = TRUE))[1:15]
  df[[col]] = ifelse(df[[col]] %in% top_levels, df[[col]], "Other")
  df[[col]] = as.factor(df[[col]])
}

# drop original columns
df = df %>% select(-amount_tsh, -construction_year, -population)

str(df)
dim(df)

# ---------------------------
# 3) Train-Validation Split
# ---------------------------

set.seed(1975)
indxTrain = createDataPartition(y = df$status_group, p = 0.8, list = FALSE)

training = df[indxTrain, ]
testing = df[-indxTrain, ]

prop.table(table(training$status_group))
dim(training)
dim(testing)

# ---------------------------
# 4) Cross-Validation Setup
# ---------------------------

ctrl = trainControl(
  method = "cv",
  number = 5,
  classProbs = TRUE,
  summaryFunction = multiClassSummary,
  verboseIter = TRUE
)

# ============================================================
# MODEL 1: DECISION TREE
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

rpart.plot(treeFit$finalModel)

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

plot(rfFit)
varImp(rfFit, scale = FALSE)

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

knnPred = predict(knnFit, newdata = testing)
knn_cm = confusionMatrix(knnPred, testing$status_group)
knn_cm

# ============================================================
# MODEL COMPARISON
# ============================================================

resamps = resamples(list(
  DecisionTree = treeFit,
  RandomForest = rfFit,
  XGBoost = xgbFit,
  KNN = knnFit
))

summary(resamps, metric = "Accuracy")
bwplot(resamps, metric = "Accuracy")
dotplot(resamps, metric = "Accuracy")

# results table
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

best_model = results$Model[1]
best_acc = results$Accuracy[1]

cat("\n========================================\n")
cat("BEST MODEL:", best_model, "\n")
cat("TEST ACCURACY:", round(best_acc, 4), "\n")
cat("========================================\n")

# ---------------------------
# Class-wise Performance
# ---------------------------

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

class_metrics = data.frame(
  Class = rownames(best_cm$byClass),
  Sensitivity = round(best_cm$byClass[, "Sensitivity"], 4),
  Specificity = round(best_cm$byClass[, "Specificity"], 4),
  Precision = round(best_cm$byClass[, "Precision"], 4),
  F1 = round(best_cm$byClass[, "F1"], 4)
)
class_metrics

# ---------------------------
# Generate Submission for Person 4
# ---------------------------

# store medians from training
train_median_lon = median(train_values$longitude[train_values$longitude != 0], na.rm = TRUE)
train_median_height = median(train_values$gps_height[train_values$gps_height != 0], na.rm = TRUE)
train_median_year = median(train_values$construction_year[train_values$construction_year != 0], na.rm = TRUE)
train_median_pop = median(train_values$population[train_values$population != 0], na.rm = TRUE)

# preprocess test
test_processed = test_values %>% select(-any_of(cols_to_drop))

test_processed$longitude[test_processed$longitude == 0] = train_median_lon
test_processed$gps_height[test_processed$gps_height == 0] = train_median_height
test_processed$construction_year[test_processed$construction_year == 0] = train_median_year
test_processed$population[test_processed$population == 0] = train_median_pop

test_processed$pump_age = 2024 - test_processed$construction_year
test_processed$log_amount_tsh = log1p(test_processed$amount_tsh)
test_processed$log_population = log1p(test_processed$population)

test_processed$public_meeting[is.na(test_processed$public_meeting)] = FALSE
test_processed$permit[is.na(test_processed$permit)] = FALSE
test_processed$public_meeting = factor(test_processed$public_meeting, levels = levels(training$public_meeting))
test_processed$permit = factor(test_processed$permit, levels = levels(training$permit))

test_char_cols = names(test_processed)[sapply(test_processed, is.character)]
for (col in test_char_cols) {
  if (col %in% names(training)) {
    test_processed[[col]] = ifelse(test_processed[[col]] %in% levels(training[[col]]), 
                                    test_processed[[col]], "Other")
    test_processed[[col]] = factor(test_processed[[col]], levels = levels(training[[col]]))
  }
}

test_processed = test_processed %>% select(-amount_tsh, -construction_year, -population)

# predict
if (best_model == "XGBoost") {
  final_pred = predict(xgbFit, newdata = test_processed)
} else if (best_model == "Random Forest") {
  final_pred = predict(rfFit, newdata = test_processed)
} else if (best_model == "KNN") {
  final_pred = predict(knnFit, newdata = test_processed)
} else {
  final_pred = predict(treeFit, newdata = test_processed)
}

# convert back to original labels for submission
final_pred_labels = gsub("_", " ", as.character(final_pred))

table(final_pred_labels)

# create submission
submission = data.frame(
  id = test_values$id,
  status_group = final_pred_labels
)

head(submission, 10)
write.csv(submission, "submission.csv", row.names = FALSE)

# save model
saveRDS(list(
  treeFit = treeFit,
  rfFit = rfFit,
  xgbFit = xgbFit,
  knnFit = knnFit,
  results = results,
  best_model = best_model
), "person3_models.rds")

cat("\nDone! Files saved:\n")
cat("  - submission.csv\n")
cat("  - person3_models.rds\n")

