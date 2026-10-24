# modeling_functions.R ------------------------------------------------------
# Home Credit Default Risk | Bradyn Jackson | IS 6812
#
# Reusable modeling helpers for modeling_home_credit.qmd, following the
# tidymodels conventions (rsample for resampling, yardstick for metrics).
#
# Evaluation runs in three tiers, each sized to the job it does:
#   1. Comparison - 20% stratified sample, 3-fold CV. Every candidate model
#      and imbalance strategy is scored on these same folds.
#   2. Tuning     - 5,000-row stratified sample, 3-fold CV. Randomized
#      hyperparameter search, where many fits make the full data too slow.
#   3. Final      - all 307,511 training rows. One fit, then predict on test.
#
# TARGET is a factor with levels c("1", "0"). Putting default first makes it
# the event level under yardstick's default event_level = "first", so every
# AUC in this project measures how well a model ranks defaulters.
#
# Sections:
#   0. Setup
#   1. Loading
#   2. Sampling and resampling
#   3. Baseline
#   4. Predictor screening
#   5. Recipes and model specifications
#   6. Resampling and comparison
#   7. Class imbalance
#   8. Hyperparameter tuning
#   9. Final model
#  10. Kaggle submission
#
#   prepared      <- load_prepared()
#   compare_data  <- stratified_sample(prepared$train, prop = COMPARE_PROP)
#   compare_folds <- make_cv_folds(compare_data)
#   baseline_majority(compare_folds)$summary

library(tidymodels)
library(themis)


# 0. Setup -------------------------------------------------------------------
# Shared constants, so every tier draws from the same seed and fold count.

MODEL_SEED   <- 6812
CV_FOLDS     <- 3
COMPARE_PROP <- 0.20
TUNE_N       <- 5000
IMBALANCE_STRATEGIES <- c("none", "downsample", "upsample", "smote")


# 1. Loading -----------------------------------------------------------------

#' Make TARGET a factor with default ("1") as the first, event, level.
#' Accepts numeric or factor input; leaves frames without TARGET (test) alone.
set_target_factor <- function(df) {
  if (!"TARGET" %in% names(df)) return(df)
  df |> mutate(TARGET = factor(as.character(TARGET), levels = c("1", "0")))
}

#' Load the prepared data, running data_preparation.R only when needed.
#' Prep reads two large CSVs and takes minutes, so the result is cached. The
#' cache rebuilds automatically when data_preparation.R is newer than it, so
#' editing the prep never leaves a render on stale data.
#' The cache holds prepare_data()'s output untouched; the TARGET factor is
#' applied after loading.
load_prepared <- function(cache_path  = "data/prepared.rds",
                          prep_script = "data_preparation.R",
                          rebuild     = FALSE) {
  stale <- !file.exists(cache_path) ||
    file.mtime(prep_script) > file.mtime(cache_path)

  if (rebuild || stale) {
    # Own environment, so the prep helpers don't clutter the notebook's.
    prep_env <- new.env()
    source(prep_script, local = prep_env)
    prepared <- prep_env$prepare_data()
    prep_env$validate_preparation(prepared$train, prepared$test)
    dir.create(dirname(cache_path), showWarnings = FALSE, recursive = TRUE)
    saveRDS(prepared, cache_path)
  } else {
    prepared <- readRDS(cache_path)
  }

  prepared$train <- set_target_factor(prepared$train)
  prepared
}


# 2. Sampling and resampling -------------------------------------------------

#' Draw a sample that keeps the training default rate. Supply prop or n.
#' Sizes are allocated per class, not passed to slice_sample(n = ) on grouped
#' data - that draws n rows from EACH class and silently returns a balanced
#' 50/50 sample, which is an imbalance strategy, not a sample.
#' Rounding: floor each class's share, then hand leftover rows to the largest
#' remainders, so the total is exactly n. (slice_sample(prop = ) floors per
#' group: n = 5000 would give 4,999.)
#' withr::with_seed() leaves the session's random state untouched.
stratified_sample <- function(df, prop = NULL, n = NULL, seed = MODEL_SEED) {
  if (is.null(prop) == is.null(n)) stop("Supply exactly one of `prop` or `n`.")
  total <- if (is.null(n)) round(prop * nrow(df)) else n
  if (total > nrow(df)) stop("Requested ", total, " rows from ", nrow(df), ".")

  counts <- table(df$TARGET)
  exact  <- as.numeric(counts) * total / nrow(df)
  size   <- floor(exact)
  short  <- total - sum(size)
  bump   <- order(exact - size, decreasing = TRUE)[seq_len(short)]
  size[bump] <- size[bump] + 1

  # split() and table() order classes by factor level, so sizes line up.
  rows <- split(seq_len(nrow(df)), df$TARGET)
  keep <- withr::with_seed(seed, unlist(
    map2(rows, size, \(i, s) i[sample.int(length(i), s)])
  ))
  df[sort(keep), ]
}

#' Build stratified V-fold CV splits. Stratifying on TARGET gives every
#' assessment fold the same ~8% default rate, so fold-to-fold AUC differences
#' reflect the model rather than the luck of the split.
#' pool = 0.05, not rsample's default 0.1: strata under `pool` are dissolved
#' into the others, and defaults are 8.07% of rows. At 0.1 the "1" stratum is
#' dissolved, every row falls in one stratum, and the folds are silently
#' unstratified. rsample warns that pools under 0.1 are "statistically risky";
#' that concern is tiny strata, and even the 5,000-row tuning sample holds
#' ~400 defaults, so the warning is expected and safe here.
make_cv_folds <- function(df, v = CV_FOLDS, seed = MODEL_SEED, pool = 0.05) {
  withr::with_seed(seed, vfold_cv(df, v = v, strata = TARGET, pool = pool))
}


# 3. Baseline ----------------------------------------------------------------

#' Score the majority-class classifier on a set of CV folds.
#' In each fold the majority class and default rate are learned from the
#' analysis set and applied to the assessment set, the same protocol every
#' candidate model follows. Predicting one constant probability for everyone
#' ranks no one above anyone else, so AUC is 0.5 by construction; accuracy
#' equals the non-default share. That gap is why accuracy is not the metric.
#' Returns per-fold metrics and a summary shaped like tune::collect_metrics().
baseline_majority <- function(folds) {
  metrics <- metric_set(accuracy, roc_auc)

  per_fold <- map2(folds$splits, folds$id, function(split, id) {
    fit_on   <- analysis(split)
    score_on <- assessment(split)
    majority <- names(which.max(table(fit_on$TARGET)))

    tibble(
      TARGET      = score_on$TARGET,
      .pred_class = factor(majority, levels = levels(score_on$TARGET)),
      .pred_1     = mean(fit_on$TARGET == "1")
    ) |>
      metrics(truth = TARGET, estimate = .pred_class, .pred_1) |>
      mutate(id = id)
  }) |>
    list_rbind()

  summary <- per_fold |>
    group_by(.metric, .estimator) |>
    summarise(mean    = mean(.estimate),
              n       = n(),
              std_err = sd(.estimate) / sqrt(n()),
              .groups = "drop") |>
    mutate(model = "majority_class", .before = 1)

  list(per_fold = per_fold, summary = summary)
}


# 4. Predictor screening -----------------------------------------------------

#' Single-variable AUC for each predictor: a sanity check on a hand-picked
#' set. Numeric columns are scored by their raw value, factors by their level's
#' default rate. AUC is folded to >= 0.5, since a predictor that lowers risk is
#' as useful as one that raises it; `direction` records which way it runs.
#' Run it on rows OUTSIDE the comparison sample. Screening predictors on the
#' rows they are later scored on would leak into the CV estimate.
univariate_auc <- function(df, predictors) {
  map(predictors, function(col) {
    x <- df[[col]]
    score <- if (is.numeric(x)) x else {
      rate <- tapply(df$TARGET == "1", x, mean)
      unname(rate[as.character(x)])
    }
    raw_auc <- roc_auc_vec(df$TARGET, score)
    tibble(
      predictor = col,
      type      = if (is.numeric(x)) "numeric" else "factor",
      auc       = max(raw_auc, 1 - raw_auc),
      direction = case_when(!is.numeric(x) ~ "by level",
                            raw_auc >= 0.5 ~ "higher = riskier",
                            TRUE           ~ "higher = safer")
    )
  }) |>
    list_rbind() |>
    arrange(desc(auc))
}

#' Default rate by quantile bin of one numeric predictor, plus the AUC of the
#' binned rates. A raw-value AUC only sees monotone relationships; binning
#' shows whether a near-0.5 score hides a curved one.
binned_default_rate <- function(df, col, bins = 10) {
  binned <- df |> mutate(bin = ntile(.data[[col]], bins))
  table  <- binned |>
    group_by(bin) |>
    summarise(from         = min(.data[[col]]),
              to           = max(.data[[col]]),
              applicants   = n(),
              default_rate = mean(TARGET == "1"),
              .groups      = "drop")
  list(table = table,
       auc   = roc_auc_vec(binned$TARGET, table$default_rate[binned$bin]))
}


# 5. Recipes and model specifications ----------------------------------------

N_THREADS <- parallel::detectCores()

#' Build the preprocessing recipe shared by every candidate.
#' predictors = NULL keeps every column; otherwise only those named.
#' step_other() pools any level under 1% of rows into "other". Rare levels
#' (NAME_INCOME_TYPE's Businessman and Student, NAME_EDUCATION_TYPE's Academic
#' degree, much of ORGANIZATION_TYPE) hold a handful of rows and often zero
#' defaults, which pushes glm coefficients toward infinity (separation) and
#' hands tree models a level to memorise. It runs inside each fold, so the 1%
#' cut is measured on analysis data only.
#' dummy = FALSE for ranger, which splits on factors natively.
#' lincomb = TRUE drops exact linear combinations (AGE_YEARS is
#' -DAYS_BIRTH / 365.25; EXT_SOURCE_COUNT is 3 minus the three missing flags),
#' which glm would otherwise report as aliased, rank-deficient coefficients.
#' imbalance adds a class-imbalance step (see section 7 for the strategies).
#' exclude drops named columns from the recipe entirely, e.g. CODE_GENDER for
#' the final-stage models. Workflows ignore columns the recipe doesn't know,
#' so the same data frames still fit and predict.
#' The recipe keeps a zero-row template: only column types are needed, and a
#' full template would be copied into every saved resampling result.
build_recipe <- function(df, predictors = NULL, dummy = TRUE, lincomb = FALSE,
                         imbalance = "none", exclude = NULL, seed = MODEL_SEED) {
  imbalance <- match.arg(imbalance, IMBALANCE_STRATEGIES)
  if (!is.null(predictors)) df <- select(df, SK_ID_CURR, TARGET, all_of(predictors))
  if (!is.null(exclude))    df <- select(df, -all_of(exclude))

  rec <- recipe(TARGET ~ ., data = df[0, ]) |>
    update_role(SK_ID_CURR, new_role = "id") |>
    step_other(all_nominal_predictors(), threshold = 0.01)
  if (imbalance == "smote") rec <- add_smotenc(rec, seed)
  if (dummy)   rec <- rec |> step_dummy(all_nominal_predictors())
  rec <- rec |> step_zv(all_predictors())
  if (lincomb) rec <- rec |> step_lincomb(all_numeric_predictors())
  if (imbalance %in% c("downsample", "upsample")) rec <- add_resampling(rec, imbalance, seed)
  rec
}

#' Plain logistic regression: the interpretable reference point.
logistic_spec <- function() {
  logistic_reg() |> set_engine("glm") |> set_mode("classification")
}

#' Random forest. Impurity importance is computed now so the final model can
#' report it without a refit.
rf_spec <- function(trees = 500, threads = N_THREADS) {
  rand_forest(trees = trees) |>
    set_engine("ranger", importance = "impurity", num.threads = threads) |>
    set_mode("classification")
}

#' Gradient-boosted trees. A learn rate below the 0.3 default with more trees
#' is the usual stable starting point; everything else stays at defaults until
#' the tuning tier.
xgb_spec <- function(trees = 500, learn_rate = 0.05, threads = N_THREADS) {
  boost_tree(trees = trees, learn_rate = learn_rate) |>
    set_engine("xgboost", nthread = threads) |>
    set_mode("classification")
}


# 6. Resampling and comparison -----------------------------------------------

#' Run a resampling job, or load its saved result if one exists.
#' The result is saved with its wall-clock time (all folds, recipe prep
#' included), so a cached load still reports how long the job took.
#' The cache is keyed by name only: after changing a recipe or spec, delete
#' the file or pass refit = TRUE.
#' cat(), not message(): the notebook suppresses messages, and a silent cache
#' hit is exactly how a stale result goes unnoticed.
run_cached <- function(name, run, dir = "data", refit = FALSE, prefix = "cv_") {
  path <- file.path(dir, paste0(prefix, name, ".rds"))

  if (file.exists(path) && !refit) {
    cached <- readRDS(path)
    cat(sprintf("Loaded cached result for %s from %s (fit %s, took %.1f s)\n",
                name, path, format(cached$fitted_at, "%Y-%m-%d %H:%M"),
                cached$seconds))
    return(cached)
  }

  start  <- Sys.time()
  result <- run()
  out <- list(name      = name,
              result    = result,
              seconds   = as.numeric(difftime(Sys.time(), start, units = "secs")),
              fitted_at = Sys.time())
  dir.create(dir, showWarnings = FALSE, recursive = TRUE)
  saveRDS(out, path)
  cat(sprintf("Fit %s in %.1f s; saved to %s\n", name, out$seconds, path))
  out
}

#' fit_resamples() through the cache.
fit_resamples_cached <- function(name, workflow, folds, metrics,
                                 control = control_resamples(),
                                 dir = "data", refit = FALSE, seed = MODEL_SEED) {
  run_cached(name, dir = dir, refit = refit, run = \()
    withr::with_seed(seed, fit_resamples(workflow, folds, metrics = metrics,
                                         control = control)))
}

#' One row per model: mean CV AUC, its standard error, accuracy and fit time,
#' with the baseline included (when given) so every candidate is read against
#' the floor.
compare_models <- function(baseline, fits) {
  candidates <- map(fits, \(f) collect_metrics(f$result) |>
                      mutate(model = f$name, seconds = f$seconds)) |>
    list_rbind()

  baseline_row <- if (!is.null(baseline)) mutate(baseline$summary, seconds = NA_real_)

  bind_rows(baseline_row, candidates) |>
    select(model, .metric, mean, std_err, seconds) |>
    pivot_wider(names_from = .metric, values_from = c(mean, std_err)) |>
    transmute(model,
              roc_auc         = mean_roc_auc,
              roc_auc_std_err = std_err_roc_auc,
              accuracy        = mean_accuracy,
              fit_seconds     = seconds) |>
    arrange(desc(roc_auc))
}

#' Per-fold AUC for two models scored on the same folds, side by side.
#' The folds are shared, so fold-level differences remove the variation
#' between folds that the two models' separate std_errs both include.
paired_folds <- function(fit_a, fit_b, metric = "roc_auc") {
  per_fold <- \(f) collect_metrics(f$result, summarize = FALSE) |>
    filter(.metric == metric) |>
    select(id, .estimate)

  inner_join(per_fold(fit_a), per_fold(fit_b), by = "id",
             suffix = c("_a", "_b")) |>
    rename(!!fit_a$name := .estimate_a, !!fit_b$name := .estimate_b) |>
    mutate(difference = .data[[fit_a$name]] - .data[[fit_b$name]],
           winner     = if_else(difference > 0, fit_a$name, fit_b$name))
}


# 7. Class imbalance ---------------------------------------------------------

#' Every themis step below uses skip = TRUE (themis's default, set
#' explicitly): it runs only when the recipe is prepped on a fold's analysis
#' set. Assessment rows pass through untouched, so every model is scored on
#' the true 8% default rate. A fixed step seed makes resampled rows
#' reproducible. build_recipe() places the steps.

#' Down- or upsample to 50/50. Row selection ignores column values, so the
#' step can go last, after the dummies.
add_resampling <- function(rec, strategy, seed = MODEL_SEED) {
  switch(strategy,
    downsample = rec |> step_downsample(TARGET, skip = TRUE, seed = seed),
    upsample   = rec |> step_upsample(TARGET, skip = TRUE, seed = seed))
}

#' Synthetic defaults that a model cannot tell apart from real ones.
#' First attempt: step_smote() after step_dummy interpolated every 0/1 column
#' into values like 0.37 that no real applicant has, and every such row was a
#' synthetic default, so xgboost learned to spot synthetic rows instead of
#' risk. step_smotenc() runs BEFORE step_dummy and gives categorical columns
#' the most common value among the neighbours instead of interpolating. It
#' only protects factors, so numeric 0/1 columns (FLAG_*, REG_*,
#' FLAG_MISSING_*) are converted to factors first. Binary columns are found
#' inside each fold, at prep time. Neighbours use Gower distance, which scales
#' numeric columns itself, so no normalising step is needed.
add_smotenc <- function(rec, seed = MODEL_SEED) {
  is_binary <- \(x) is.numeric(x) && all(x %in% c(0, 1))
  rec |>
    step_bin2factor(all_numeric_predictors() & where(is_binary)) |>
    step_smotenc(TARGET, skip = TRUE, seed = seed)
}

#' One row per fit: CV AUC, its std_err, accuracy, and the mean predicted
#' default probability on the assessment sets. Resampling to 50/50 inflates
#' predicted probabilities, and comparing that mean to the true default rate
#' shows by how much. Needs fits run with save_pred = TRUE.
summarise_fits <- function(fits) {
  map(fits, function(f) {
    metrics <- collect_metrics(f$result)
    preds   <- collect_predictions(f$result)
    tibble(
      fit               = f$name,
      roc_auc           = metrics$mean[metrics$.metric == "roc_auc"],
      roc_auc_std_err   = metrics$std_err[metrics$.metric == "roc_auc"],
      accuracy          = metrics$mean[metrics$.metric == "accuracy"],
      mean_pred_default = mean(preds$.pred_1),
      fit_seconds       = f$seconds
    )
  }) |>
    list_rbind()
}

#' How many SMOTE training rows can a model recognise as synthetic?
#' SMOTE interpolates every column, so a 0/1 dummy or flag can come out as
#' 0.37 - a value no real applicant has. Every such row is a synthetic
#' default, so a tree model can split on it and learn to spot synthetic rows
#' instead of risk. Counts rows holding any value outside the two that a
#' binary column takes on real rows, for one fold. Should be ~0 with
#' add_smotenc().
smote_fractional_rows <- function(recipe, split) {
  fitted <- prep(recipe, analysis(split))
  real   <- bake(fitted, analysis(split))   # skip = TRUE: SMOTE not applied
  smoted <- bake(fitted, NULL)              # the rows the model trains on
  binary <- names(real)[map_lgl(real, \(x) is.numeric(x) && n_distinct(x) == 2)]
  fractional <- map(binary, \(col) !smoted[[col]] %in% unique(real[[col]])) |>
    reduce(`|`)

  tibble(binary_columns        = length(binary),
         training_rows         = nrow(smoted),
         synthetic_rows        = nrow(smoted) - nrow(real),
         recognisable_rows     = sum(fractional),
         recognisable_defaults = mean(smoted$TARGET[fractional] == "1"))
}


#' Adversarial check: can a model tell synthetic defaults from real ones?
#' smote_fractional_rows() only inspects binary columns. This trains xgboost
#' to separate synthetic from real defaults in one fold's SMOTE training data,
#' using every column. AUC near 0.5 means synthetic rows pass as real; near 1
#' means a tree model can learn "synthetic" instead of "risky".
#' Also reports the share of synthetic defaults holding a non-integer value in
#' a column that is integer-valued on real rows (counts, DAYS_*), the
#' interpolation artifact that step_bin2factor() does not cover.
#' Real defaults are ~9% of these rows, under rsample's default pool of 0.1,
#' so the folds pass pool = 0.05 for the same reason as make_cv_folds().
synthetic_detectability <- function(recipe, split, v = 3, seed = MODEL_SEED) {
  smote_step <- which(map_lgl(recipe$steps, \(s) inherits(s, "step_smotenc")))
  recipe$steps[[smote_step]]$indicator_column <- "synthetic"

  defaults <- prep(recipe, analysis(split)) |>
    bake(NULL) |>
    filter(TARGET == "1") |>
    mutate(synthetic = factor(synthetic, levels = c(1, 0))) |>
    select(-TARGET, -SK_ID_CURR)
  is_synth <- defaults$synthetic == "1"

  integer_cols <- names(defaults)[map_lgl(defaults, \(x)
    is.numeric(x) && n_distinct(x[!is_synth]) > 2 && all(x[!is_synth] == round(x[!is_synth])))]
  off_grid <- map(integer_cols, \(col) defaults[[col]][is_synth] != round(defaults[[col]][is_synth])) |>
    reduce(`|`)

  folds <- withr::with_seed(seed, vfold_cv(defaults, v = v, strata = synthetic, pool = 0.05))
  detect <- withr::with_seed(seed, fit_resamples(
    workflow(recipe(synthetic ~ ., data = defaults[0, ]), xgb_spec(trees = 200)),
    folds, metrics = metric_set(roc_auc)))

  tibble(real_defaults           = sum(!is_synth),
         synthetic_defaults      = sum(is_synth),
         integer_columns         = length(integer_cols),
         synthetic_off_integer   = mean(off_grid),
         detection_auc           = collect_metrics(detect)$mean)
}

# 8. Hyperparameter tuning ---------------------------------------------------

#' XGBoost with every tree parameter open for tuning except the tree count,
#' which stays at 500 so learn_rate and tree count don't trade off against
#' each other inside a 20-point search.
xgb_tune_spec <- function(trees = 500, threads = N_THREADS) {
  boost_tree(trees = trees, tree_depth = tune(), learn_rate = tune(),
             min_n = tune(), loss_reduction = tune(), sample_size = tune(),
             mtry = tune()) |>
    set_engine("xgboost", nthread = threads) |>
    set_mode("classification")
}

#' Random grid over a workflow's tuning parameters. mtry's upper bound is the
#' number of predictors AFTER preprocessing (dummies included), so it is
#' finalized against the recipe baked on the tuning data.
random_tune_grid <- function(workflow, data, size = 20, seed = MODEL_SEED) {
  predictors <- extract_preprocessor(workflow) |>
    prep(data) |>
    bake(NULL) |>
    select(-any_of(c("TARGET", "SK_ID_CURR")))

  params <- extract_parameter_set_dials(workflow) |> finalize(predictors)
  withr::with_seed(seed, grid_random(params, size = size))
}

#' tune_grid() through the cache.
tune_grid_cached <- function(name, workflow, folds, grid, metrics,
                             dir = "data", refit = FALSE, seed = MODEL_SEED) {
  run_cached(name, dir = dir, refit = refit, run = \()
    withr::with_seed(seed, tune_grid(workflow, folds, grid = grid,
                                     metrics = metrics)))
}

#' Re-score the best n configurations from a tuning run on other folds.
#' 5,000 rows hold ~400 defaults, so tuning AUCs are noisy and the top few
#' are often within noise of each other; the comparison folds decide.
#' Each re-score is cached under its tuning .config. Those names are only
#' stable while the grid is: if the grid changes, refit these too.
rescore_top <- function(tuning, workflow, folds, metrics, n = 3,
                        prefix = "xgb_tuned") {
  best <- show_best(tuning$result, metric = "roc_auc", n = n)
  fits <- map(seq_len(nrow(best)), \(i) fit_resamples_cached(
    paste(prefix, best$.config[i], sep = "_"),
    finalize_workflow(workflow, best[i, ]),
    folds, metrics))
  set_names(fits, best$.config)
}

#' Score one tuned configuration at several tree counts on the same folds.
#' Tuning held trees at 500 while learn_rate varied; a small winning rate may
#' still be improving at 500 trees. Each count is cached as xgb_trees_<n>.
sweep_trees <- function(recipe, params, trees, folds, metrics, prefix = "xgb_trees") {
  map(trees, \(n) fit_resamples_cached(
    paste(prefix, n, sep = "_"),
    finalize_workflow(workflow(recipe, xgb_tune_spec(trees = n)), params),
    folds, metrics)) |>
    set_names(trees)
}

#' Fewest trees whose mean AUC is within `tolerance` of the best. Gains below
#' the tolerance count as flat: they are a small fraction of what tuning
#' added, and more trees cost fit time with no reliable return.
pick_trees <- function(sweep, tolerance = 0.001) {
  aucs <- map_dbl(sweep, \(f) collect_metrics(f$result) |>
                    filter(.metric == "roc_auc") |> pull(mean))
  as.integer(min(as.numeric(names(aucs))[aucs >= max(aucs) - tolerance]))
}


# 9. Final model -------------------------------------------------------------

#' Fit a workflow on the full training set, or load the saved fit.
#' Saved to models/, the fitted workflow carries its prepped recipe, so it
#' predicts on raw prepared data (the test set) directly.
fit_final_cached <- function(name, workflow, data, dir = "models", refit = FALSE,
                             seed = MODEL_SEED) {
  run_cached(name, dir = dir, refit = refit, prefix = "", run = \()
    withr::with_seed(seed, fit(workflow, data)))
}

#' Top n features by gain from a fitted xgboost workflow. Gain is each
#' feature's share of the total loss reduction from all splits on it, so the
#' values sum to 1 across features. Names are post-recipe (dummies included).
xgb_importance <- function(fitted, n = 20) {
  xgboost::xgb.importance(model = extract_fit_engine(fitted)) |>
    as_tibble() |>
    select(feature = Feature, gain = Gain) |>
    slice_max(gain, n = n)
}


# 10. Kaggle submission -------------------------------------------------------

#' Compare numeric predictors between train and test. The standardised mean
#' difference (difference in means over the pooled SD) is scale-free; above
#' about 0.1 is the usual threshold for a shift worth a closer look.
drift_table <- function(train, test, cols) {
  map(cols, function(col) {
    a <- train[[col]]; b <- test[[col]]
    tibble(predictor   = col,
           train_median = median(a), test_median = median(b),
           train_mean   = mean(a),   test_mean   = mean(b),
           smd          = (mean(b) - mean(a)) / sqrt((var(a) + var(b)) / 2))
  }) |>
    list_rbind()
}

#' Check a submission against the test set before it goes to Kaggle. Every
#' check must pass; the table shows each so a failure is visible, and the
#' function stops if any fails so a bad file is never written silently.
validate_submission <- function(submission, test) {
  checks <- tibble(
    check = c("Columns are SK_ID_CURR, TARGET",
              "Row count equals test",
              "SK_ID_CURR unique",
              "SK_ID_CURR matches test exactly",
              "No missing values",
              "TARGET within [0, 1]",
              "Mean TARGET near training default rate (0.05-0.12)"),
    value = c(paste(names(submission), collapse = ", "),
              format(nrow(submission), big.mark = ","),
              format(n_distinct(submission$SK_ID_CURR), big.mark = ","),
              as.character(setequal(submission$SK_ID_CURR, test$SK_ID_CURR)),
              format(sum(is.na(submission))),
              sprintf("%.4f to %.4f", min(submission$TARGET), max(submission$TARGET)),
              sprintf("%.4f", mean(submission$TARGET))),
    pass = c(identical(names(submission), c("SK_ID_CURR", "TARGET")),
             nrow(submission) == nrow(test),
             n_distinct(submission$SK_ID_CURR) == nrow(submission),
             setequal(submission$SK_ID_CURR, test$SK_ID_CURR),
             !anyNA(submission),
             all(submission$TARGET >= 0 & submission$TARGET <= 1),
             between(mean(submission$TARGET), 0.05, 0.12))
  )
  if (!all(checks$pass)) {
    print(checks)
    stop("Submission failed validation: ", paste(checks$check[!checks$pass], collapse = "; "))
  }
  checks
}


# 11. Business impact ---------------------------------------------------------
# Used by business_impact.qmd. Every number there comes from out-of-fold
# predictions: each applicant is scored by a model that never saw their
# outcome, as a new applicant would be.

#' Out-of-fold predicted default probabilities for every row the folds cover.
#' Cached like the other fits, but only the predictions and per-fold metrics
#' are kept: a full resampling object also stores a copy of every training
#' row, and here that is all 307,511.
oof_predictions_cached <- function(name, workflow, folds, metrics,
                                   dir = "data", refit = FALSE, seed = MODEL_SEED) {
  run_cached(name, dir = dir, refit = refit, run = \() {
    res <- withr::with_seed(seed, fit_resamples(
      workflow, folds, metrics = metrics,
      control = control_resamples(save_pred = TRUE)))
    list(predictions = collect_predictions(res) |> select(id, .row, .pred_1, TARGET),
         per_fold    = collect_metrics(res, summarize = FALSE))
  })
}

#' The portfolio at every approval count: approve the k lowest-risk applicants,
#' for k = 1..n. `risk` is any score where higher means riskier (pass
#' -EXT_SOURCE_MEAN for a score where higher means safer). order() is stable,
#' so tied scores are approved in row order.
#' `credit` is each applicant's loan amount; defaulted_credit is the total
#' credit held by the defaulters among those approved.
approval_path <- function(risk, default, credit) {
  o <- order(risk)
  tibble(approved         = seq_along(o),
         approval_rate    = approved / length(o),
         defaults         = cumsum(default[o]),
         default_rate     = defaults / approved,
         defaulted_credit = cumsum(credit[o] * default[o]))
}

#' The largest portfolio whose default rate stays at or under `target`.
#' Reads the path at single-applicant resolution, not in 1-point steps.
at_target_rate <- function(path, target) {
  path[max(which(path$default_rate <= target)), ]
}

#' Per-feature contributions to each row's predicted log-odds of default,
#' from xgboost's predcontrib = TRUE (SHAP values). Rows are baked through the
#' fitted recipe, so features are post-recipe (dummies included). BIAS is the
#' model's baseline log-odds; BIAS plus every contribution is the row's
#' log-odds, which is checked against the workflow's own predicted
#' probability before anything is returned.
#' The matrix is built the way parsnip builds it for fitting and predicting:
#' step_dummy() returns sparse columns, so parsnip hands xgboost a sparse
#' matrix, where every zero counts as missing. A dense matrix gives different
#' predictions from the same booster.
#' Signs are flipped if needed so a positive contribution always pushes
#' toward default, whichever class the booster treats as positive.
xgb_contributions <- function(fitted, new_data) {
  booster <- extract_fit_engine(fitted)
  baked   <- bake(extract_recipe(fitted), new_data, all_predictors())
  baked   <- baked[, variable.names(booster)]
  x <- if (sparsevctrs::has_sparse_elements(baked)) {
    sparsevctrs::coerce_to_sparse_matrix(baked)
  } else {
    as.matrix(baked)
  }
  contrib <- predict(booster, xgboost::xgb.DMatrix(x, missing = NA), predcontrib = TRUE)

  p_booster <- plogis(rowSums(contrib))
  p_default <- predict(fitted, new_data, type = "prob")$.pred_1
  sign <- if (isTRUE(all.equal(p_booster, p_default, tolerance = 1e-5))) 1
          else if (isTRUE(all.equal(1 - p_booster, p_default, tolerance = 1e-5))) -1
          else stop("Contributions do not reproduce the predicted probabilities.")

  # xgboost 3 names the baseline column "(Intercept)"; earlier versions "BIAS".
  as_tibble(sign * contrib) |>
    mutate(.row = seq_len(n()), .before = 1) |>
    pivot_longer(-.row, names_to = "feature", values_to = "contribution") |>
    mutate(feature = if_else(feature %in% c("(Intercept)", "BIAS"), "BIAS", feature))
}
