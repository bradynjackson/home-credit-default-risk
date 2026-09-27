# data_preparation.R --------------------------------------------------------
# Home Credit Default Risk | Bradyn Jackson | IS 6812
#
# Applies the decisions from eda_home_credit.qmd identically to train and test.
#
# Learned values (medians, caps, factor levels) come from TRAINING data only.
# fit_prep_params() measures them; apply_prep() only looks them up. That split
# is what prevents test data from influencing its own preparation.
#
# Sections follow the assignment task list:
#   0. Setup and loading
#   1. Clean and transform the application data
#   2. Create engineered features
#   3. Supplementary tables (out of scope - see note)
#   4. Ensure train/test consistency
#   5. Run and validate
#
#   prepared <- prepare_data()
#   validate_preparation(prepared$train, prepared$test)

library(tidyverse)


# 0. Setup and loading ------------------------------------------------------

#' Read an application CSV. guess_max beats readr's 1000-row default, which
#' mis-types the columns that are empty near the top of the file.
read_application <- function(path) {
  read_csv(path, show_col_types = FALSE, guess_max = 50000)
}


# 1. Clean and transform the application data -------------------------------
# Implements the cleaning decisions recorded in the EDA notebook. Nothing here
# depends on a measured value, so both the fitting and applying stages below
# can call these safely.

#' Replace sentinel values with NA, keeping a flag for the fact of absence.
#' EDA Decision: DAYS_EMPLOYED = 365,243 is sentinel. Not a real value. 
#' It works out to 1,000 years. It appears in 55,374 rows, 18% of the data, 
#' and 55,352 of those are pensioners. Since pensioners turned out to be the 
#' lowest-risk group in the data, dropping these rows would remove the safest 
#' applicants, so we recode to NA and add a flag instead. 
#' EDA Decision: Four rows in CODE_GENDER carry the code XNA, which is 
#' Home Credit's house code for an unknown value rather than a real category. 
#' We treat it as NA because then it reads as an unknown value.  
recode_sentinels <- function(df) {
  df |>
    mutate(
      FLAG_DAYS_EMPLOYED_PLACEHOLDER = as.integer(DAYS_EMPLOYED == 365243),
      DAYS_EMPLOYED = if_else(DAYS_EMPLOYED == 365243, NA_real_, DAYS_EMPLOYED)
    ) |>
    mutate(across(where(is.character),
                  ~ if_else(.x == "XNA", NA_character_, .x)))
}

#' Flag where values were missing. Must run before imputation removes the
#' evidence.
#' EDA Decision: The emptiest columns in the data are all building attributes - 
#' COMMONAREA, NONLIVINGAPARTMENTS, LIVINGAPARTMENTS, FLOORSMIN - at 67-70% 
#' missing, which suggests they were only collected for certain housing types. 
#' We flag these missing values because they act as a filler for housing type, 
#' and insinuation alone would take that away. 
add_missing_indicators <- function(df) {
  flag_cols <- intersect(
    c("EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3",
      "AMT_ANNUITY", "AMT_GOODS_PRICE", "OWN_CAR_AGE"),
    names(df)
  )
  df |>
    mutate(across(all_of(flag_cols), ~ as.integer(is.na(.x)),
                  .names = "FLAG_MISSING_{.col}"))
}

# 2. Create engineered features ---------------------------------------------

#' Derive new columns: demographics, financial ratios, external-score summary
#' and a binned variable. Every feature is a deterministic function of columns
#' already present - no medians, percentiles or levels.
#' EDA Decision: AMT_CREDIT runs from 45,000 to 4,050,000 with a median of 
#' 513,531, against a median income of 147,150. A raw loan amount doesn't 
#' distinguish applicants, ratios help show correlations that raw data doesn't. 
#' EDA Decision: EXT_SOURCE_3, _2, _1 correlate with TARGET at -0.179, -0.160, 
#' and -0.155, the strongest relationships in the data, but they are missing 
#' for 56.4%, 0.2%, and 19.8% of applicants respectively. Relying on a single 
#' one leaves a lot of applicants with nothing, where a mean or max across all 
#' three produces a usable value for anyone holding at least one score.  
engineer_features <- function(df, age_breaks = seq(20, 70, by = 5)) {

  # DAYS_* columns count backwards from application date and are negative.
  df <- df |>
    mutate(
      AGE_YEARS          = -DAYS_BIRTH / 365.25,
      YEARS_EMPLOYED     = -DAYS_EMPLOYED / 365.25,
      YEARS_REGISTERED   = -DAYS_REGISTRATION / 365.25,
      YEARS_ID_PUBLISHED = -DAYS_ID_PUBLISH / 365.25,
      EMPLOYED_AGE_RATIO = DAYS_EMPLOYED / DAYS_BIRTH   # share of life employed
    )

  # Amounts mean little without the income they sit against.
  df <- df |>
    mutate(
      CREDIT_INCOME_RATIO  = AMT_CREDIT / AMT_INCOME_TOTAL,    # leverage
      ANNUITY_INCOME_RATIO = AMT_ANNUITY / AMT_INCOME_TOTAL,   # affordability
      CREDIT_TERM          = AMT_ANNUITY / AMT_CREDIT,         # implied term
      CREDIT_GOODS_RATIO   = AMT_CREDIT / AMT_GOODS_PRICE,     # cash on top
      INCOME_PER_PERSON    = AMT_INCOME_TOTAL / pmax(CNT_FAM_MEMBERS, 1)
    )

  # The three external scores differ sharply in missingness, so summarising
  # gives a value for anyone holding at least one.
  ext_cols <- intersect(paste0("EXT_SOURCE_", 1:3), names(df))
  if (length(ext_cols) > 0) {
    ext_mat  <- as.matrix(df[ext_cols])
    ext_list <- as.list(df[ext_cols])
    df <- df |>
      mutate(
        EXT_SOURCE_MEAN  = rowMeans(ext_mat, na.rm = TRUE),
        # pmin/pmax are vectorised; apply() over 300k rows is far slower.
        EXT_SOURCE_MIN   = do.call(pmin, c(ext_list, na.rm = TRUE)),
        EXT_SOURCE_MAX   = do.call(pmax, c(ext_list, na.rm = TRUE)),
        EXT_SOURCE_COUNT = rowSums(!is.na(ext_mat))
      )
  }

  # Character, so it flows through the same encoding path as other categoricals.
  # EDA Decision: Default rate falls from about 12% at ages 20-25 to about 4% 
  # at 65-70. We bin by 5 years because the decline isnt linear. It flattens in 
  # the middle. We can also take different actions against different bins, if 
  # applicants under 25 are three times more likely to default than those over 
  # 65, we can seperate the two. 
  df <- df |>
    mutate(AGE_BAND = as.character(
      cut(AGE_YEARS, age_breaks, right = FALSE, include.lowest = TRUE)
    ))

  # Division by zero or NA yields Inf/NaN. Convert so imputation handles them.
  # Assignment preserves column type; if_else() would coerce.
  df |>
    mutate(across(where(is.numeric), \(x) { x[!is.finite(x)] <- NA; x }))
}


# 3. Supplementary tables ---------------------------------------------------
# Out of scope. The EDA notebook analysed application_train and application_test
# only; the seven supplementary tables (bureau, previous_application,
# installments_payments and the rest) were optional at that stage and were not
# explored. Aggregating and joining them without having examined their grain or
# distributions would mean encoding choices this project cannot defend. They
# remain a candidate for a later phase.


# 4. Ensure train/test consistency ------------------------------------------
# The two functions below are the leakage guard. fit_prep_params() measures
# every data-dependent value on train and stores it; apply_prep() only reads
# those stored values, so test data never influences its own preparation.

# 4a. Fitting ----------------------------------------------------------------

#' Learn every data-dependent value from the training set.
#'
#' Walks the same path apply_prep() walks, so medians are measured on the data
#' in the state it will actually be in when imputation runs.
#'
#' @param missing_threshold Drop columns missing more than this share
#' @param winsor_p Percentile at which monetary values are capped
fit_prep_params <- function(train,
                            missing_threshold = 0.60,
                            winsor_p = 0.99,
                            age_breaks = seq(20, 70, by = 5)) {

  # Columns to drop, measured on train and reused for test.
  # EDA Decision: 41 of the 122 columns are more than 50% empty and only 55 are 
  # complete. EXT_SOURCE_1 IS 56.4% missing but is one of the three strongest 
  # predictors, so a 50% cutoff would discard it. We use 60% because LANDAREA_ 
  # is missing 59.4% of data and the next is OWN_CAR_AGE at 66.0%. So anywhere 
  # from 60 to 65 drops the identical 17 columns.  
  missing_share <- train |>
    summarise(across(everything(), ~ mean(is.na(.x)))) |>
    pivot_longer(everything(), names_to = "variable", values_to = "share")

  drop_cols <- missing_share |>
    filter(share > missing_threshold) |>
    pull(variable) |>
    setdiff(c("SK_ID_CURR", "TARGET"))

  # Caps rather than row deletion: apply_prep() runs on test, where every row
  # must survive to receive a prediction.
  # unname() matters - quantile() returns a value named "99%", which would
  # otherwise key the vector as "AMT_INCOME_TOTAL.99%" and break every lookup.
  # EDA Decision: AMT_INCOME_TOTAL reaches 117,000,000 against a 99th percentile 
  # of 472,500, and the largest loan anyone in the data takes is 4,050,000. The 
  # EDA concluded this row should be dropped, but this function also runs on test 
  # data, where every row must produce a prediction - so we cap instead because 
  # the ratio features divide by income. 
  money_cols <- intersect(
    c("AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY", "AMT_GOODS_PRICE"),
    names(train)
  )
  money_caps <- map_dbl(set_names(money_cols),
                        \(col) unname(quantile(train[[col]], winsor_p, na.rm = TRUE)))

  # Walk the transformation path before measuring medians, so the engineered
  # columns get stored medians too.
  train_fe <- train |> recode_sentinels() |> select(-any_of(drop_cols))
  for (col in intersect(money_cols, names(train_fe))) {
    train_fe[[col]] <- pmin(train_fe[[col]], money_caps[[col]])
  }
  train_fe <- engineer_features(train_fe, age_breaks)

  numeric_cols <- train_fe |>
    select(where(is.numeric), -any_of(c("SK_ID_CURR", "TARGET"))) |>
    names()
  numeric_medians <- map_dbl(set_names(numeric_cols),
                             \(col) median(train_fe[[col]], na.rm = TRUE))
  numeric_medians[is.na(numeric_medians)] <- 0   # all-missing column

  # Fixed levels are what make train and test line up after encoding. An unseen
  # test category becomes NA rather than a new column.
  # EDA Decision: There are 16 character columns. We fix their levels from the 
  # training data because of leakage, anything you measure on the test set and 
  # feed back into the pipeline is information the model wouldn't have at 
  # prediction time.  
  cat_cols <- train_fe |> select(where(is.character)) |> names()
  cat_levels <- map(set_names(cat_cols),
                    \(col) sort(unique(na.omit(train_fe[[col]]))))
  cat_modes <- map_chr(set_names(cat_cols), function(col) {
    tbl <- sort(table(train_fe[[col]]), decreasing = TRUE)
    if (length(tbl) == 0) NA_character_ else names(tbl)[1]
  })

  list(
    missing_threshold = missing_threshold,
    winsor_p          = winsor_p,
    age_breaks        = age_breaks,
    drop_cols         = drop_cols,
    money_cols        = money_cols,
    money_caps        = money_caps,
    numeric_medians   = numeric_medians,
    cat_cols          = cat_cols,
    cat_levels        = cat_levels,
    cat_modes         = cat_modes,
    fitted_on_rows    = nrow(train),
    fitted_at         = Sys.time()
  )
}


# 4b. Applying ---------------------------------------------------------------

#' Transform any dataset using the fitted parameters. Computes nothing itself.
apply_prep <- function(df, params) {

  df <- df |>
    recode_sentinels() |>
    add_missing_indicators() |>
    select(-any_of(params$drop_cols))

  for (col in intersect(params$money_cols, names(df))) {
    df[[col]] <- pmin(df[[col]], params$money_caps[[col]])
  }

  df <- engineer_features(df, params$age_breaks)

  # Direct assignment, not if_else(), which errors when a column's type and the
  # replacement value's type differ.
  for (col in names(params$numeric_medians)) {
    if (col %in% names(df)) df[[col]][is.na(df[[col]])] <- params$numeric_medians[[col]]
  }
  for (col in params$cat_cols) {
    if (col %in% names(df)) df[[col]][is.na(df[[col]])] <- params$cat_modes[[col]]
  }

  # Stored levels - the step that guarantees identical columns.
  for (col in params$cat_cols) {
    if (col %in% names(df)) df[[col]] <- factor(df[[col]], params$cat_levels[[col]])
  }

  df
}


# 5. Run and validate --------------------------------------------------------

#' Fit on train, apply to both.
prepare_data <- function(train_path = "data/application_train.csv",
                         test_path  = "data/application_test.csv",
                         ...) {
  message("Reading train..."); train_raw <- read_application(train_path)
  message("Reading test...");  test_raw  <- read_application(test_path)

  message("Fitting parameters on train...")
  params <- fit_prep_params(train_raw, ...)

  message("Applying to train..."); train_prep <- apply_prep(train_raw, params)
  message("Applying to test...");  test_prep  <- apply_prep(test_raw,  params)

  message("Done.")
  list(train = train_prep, test = test_prep, params = params)
}


# Validation -----------------------------------------------------------------

#' Confirm train and test came out structurally identical. Run every time.
validate_preparation <- function(train, test) {

  train_only <- setdiff(names(train), names(test))
  test_only  <- setdiff(names(test),  names(train))

  levels_ok <- all(map_lgl(intersect(names(train), names(test)), function(col) {
    if (!is.factor(train[[col]])) return(TRUE)
    identical(levels(train[[col]]), levels(test[[col]]))
  }))

  checks <- list(
    columns_match = setequal(setdiff(names(train), "TARGET"), names(test)),
    train_only    = train_only,
    test_only     = test_only,
    levels_match  = levels_ok,
    train_dim     = dim(train),
    test_dim      = dim(test),
    train_na      = sum(is.na(select(train, -any_of("TARGET")))),
    test_na       = sum(is.na(test))
  )

  fmt <- function(x) format(x, big.mark = ",")
  cat("\n--- Preparation validation ---\n")
  cat("Columns match (excl. TARGET):", checks$columns_match, "\n")
  cat("In train only:  ", if (length(train_only)) paste(train_only, collapse = ", ") else "(none)", "\n")
  cat("In test only:   ", if (length(test_only))  paste(test_only,  collapse = ", ") else "(none)", "\n")
  cat("Factor levels identical:", checks$levels_match, "\n")
  cat("Train:", fmt(checks$train_dim[1]), "x", checks$train_dim[2], "\n")
  cat("Test: ", fmt(checks$test_dim[1]),  "x", checks$test_dim[2],  "\n")
  cat("NAs remaining - train:", fmt(checks$train_na), " test:", fmt(checks$test_na), "\n")
  cat("------------------------------\n\n")

  invisible(checks)
}


# Example -------------------------------------------------------------------
# prepared <- prepare_data()
# validate_preparation(prepared$train, prepared$test)
# prepared$params$drop_cols
