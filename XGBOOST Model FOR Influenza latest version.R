


# =======================================================
# R Script: Predicting Influenza Cases Using XGBoost
# Date: [19 March]
# Purpose: 
#   - Train an XGBoost model for influenza forecasting
#   - Evaluate model performance on a test set (12 weeks)
#   - Forecast future influenza trends for 12 weeks ahead
# =======================================================





# Install missing packages if necessary
packages <- c("readxl", "dplyr", "xgboost", "ggplot2", "zoo")
new_packages <- packages[!(packages %in% installed.packages()[, "Package"])]
if (length(new_packages)) install.packages(new_packages)

# Load libraries
library(readxl)
library(dplyr)
library(xgboost)
library(ggplot2)
library(zoo)

# File and output directory
file_path  <- "/Users/turkialmalki/Desktop/Influenza modelling/gcc influenza model data (updated one).xlsx"
output_dir <- "/Users/turkialmalki/Desktop/SARIMA results"
dir.create(output_dir, showWarnings = FALSE)

# Define countries
countries <- c("KSA", "UAE", "Oman", "Qatar", "Bahrain")

# Define test and forecast periods
test_weeks     <- 12  # 12 weeks for testing
forecast_weeks <- 12  # 12 weeks for forecasting

###############################################################################
# 1) Read & Process Data
###############################################################################
read_country_data <- function(sheet_name) {
  df <- read_excel(file_path, sheet = sheet_name)
  
  if (!"Date" %in% names(df)) stop("Date column not found in ", sheet_name)
  
  df$Date <- as.Date(df$Date, origin = "1899-12-30")
  
  if (toupper(sheet_name) == "UAE") {
    df <- df %>%
      group_by(Date) %>%
      summarise(`Influenza Positive` = sum(`Influenza Positive`, na.rm = TRUE)) %>%
      ungroup()
  }
  
  df <- df %>% filter(Date < as.Date("2020-01-01") | Date > as.Date("2021-12-31")) %>%
    arrange(Date)
  
  return(df)
}

###############################################################################
# 2) Feature Engineering: Seasonal, Trend and Lag Features
###############################################################################
build_features <- function(dates_vec, lag_values = NULL, train_min = NULL, train_max = NULL) {
  day_num <- as.numeric(dates_vec)
  min_day <- as.numeric(train_min)
  max_day <- as.numeric(train_max)
  trend <- (day_num - min_day) / (max_day - min_day)
  
  df <- data.frame(
    sin_date = sin(2 * pi * day_num / 365.25),
    cos_date = cos(2 * pi * day_num / 365.25),
    trend = trend
  )
  
  if (!is.null(lag_values)) {
    df$lag1 <- lag_values
  }
  
  return(df)
}

build_forecast_features <- function(forecast_date, lag_value, train_min, train_max) {
  day_num <- as.numeric(forecast_date)
  min_day <- as.numeric(train_min)
  max_day <- as.numeric(train_max)
  trend <- (day_num - min_day) / (max_day - min_day)
  data.frame(
    sin_date = sin(2 * pi * day_num / 365.25),
    cos_date = cos(2 * pi * day_num / 365.25),
    trend = trend,
    lag1 = lag_value
  )
}

###############################################################################
# 3) Modeling, Forecasting & Evaluation using XGBoost
###############################################################################
process_country <- function(ctry) {
  cat("\n==== Processing:", ctry, "====\n")
  df <- read_country_data(ctry)
  
  df <- df %>% rename(cases = `Influenza Positive`)
  
  if (nrow(df) < 10) {
    cat("[WARNING]", ctry, "has <10 rows after reading.\n")
    return(NULL)
  }
  
  df <- df %>%
    arrange(Date) %>%
    mutate(lag1 = lag(cases, 1)) %>%
    filter(!is.na(lag1))
  
  last_date  <- max(df$Date, na.rm = TRUE)
  test_start <- last_date - (test_weeks * 7)
  
  train_df <- df %>% filter(Date <= test_start)
  test_df  <- df %>% filter(Date > test_start & Date <= last_date)
  
  if (nrow(train_df) < 2) {
    cat("[WARNING]", ctry, "has <2 training rows.\n")
    return(NULL)
  }
  
  train_min_date <- min(train_df$Date)
  train_max_date <- max(train_df$Date)
  
  train_feat <- build_features(train_df$Date, train_df$lag1, train_min_date, train_max_date)
  test_feat  <- build_features(test_df$Date, test_df$lag1, train_min_date, train_max_date)
  
  X_train <- as.matrix(train_feat)
  y_train <- train_df$cases
  
  X_test <- as.matrix(test_feat)
  y_test <- test_df$cases
  
  set.seed(123)
  dtrain <- xgb.DMatrix(data = X_train, label = y_train)
  
  params <- list(objective = "reg:squarederror", max_depth = 6, eta = 0.1)
  watchlist <- list(train = dtrain)
  
  xgb_model <- xgb.train(
    params = params, data = dtrain, nrounds = 200,
    watchlist = watchlist, early_stopping_rounds = 10, verbose = 0
  )
  
  train_pred <- predict(xgb_model, dtrain)
  test_pred  <- predict(xgb_model, X_test)
  
  # Save Test Predictions & Actual Data
  test_out <- data.frame(Date = test_df$Date, Actual = y_test, Predicted = test_pred)
  write.csv(test_out, file = file.path(output_dir, paste0(ctry, "_TestPred.csv")), row.names = FALSE)
  
  # Compute Performance Metrics for Train & Test
  mse  <- function(actual, pred) mean((actual - pred)^2)
  rmse <- function(actual, pred) sqrt(mse(actual, pred))
  mae  <- function(actual, pred) mean(abs(actual - pred))
  
  train_mse  <- mse(y_train, train_pred)
  train_rmse <- rmse(y_train, train_pred)
  train_mae  <- mae(y_train, train_pred)
  
  test_mse  <- mse(y_test, test_pred)
  test_rmse <- rmse(y_test, test_pred)
  test_mae  <- mae(y_test, test_pred)
  
  # Forecast 12 weeks ahead
  future_dates <- seq(last_date + 7, by = 7, length.out = forecast_weeks)
  future_feat  <- build_forecast_features(future_dates, tail(y_train, 1), train_min_date, train_max_date)
  X_future <- as.matrix(future_feat)
  pred_future <- predict(xgb_model, X_future)
  
  # Save Forecast Data
  future_df <- data.frame(Date = future_dates, Forecast = pred_future)
  write.csv(future_df, file = file.path(output_dir, paste0(ctry, "_Forecast.csv")), row.names = FALSE)
  
  # Test vs. Predicted Plot
  test_plot <- ggplot(test_out, aes(x = Date)) +
    geom_line(aes(y = Actual, color = "Actual"), size = 1) +
    geom_line(aes(y = Predicted, color = "Predicted"), size = 1) +
    scale_color_manual(name = NULL, values = c("Actual" = "blue", "Predicted" = "red")) +
    theme_minimal() + labs(title = paste("Test Predictions for", ctry))
  
  print(test_plot)
  
  # Forecast Plot with Centered Title and No Legend Label
  forecast_plot <- ggplot() +
    geom_line(data = df, aes(x = Date, y = cases, color = "Observed"), size = 1) +
    geom_line(data = future_df, aes(x = Date, y = Forecast, color = "Forecast"), size = 1) +
    scale_color_manual(values = c("Observed" = "grey", "Forecast" = "orange")) +  # No name for legend
    theme_minimal() + 
    theme(
      plot.title = element_text(hjust = 0.5),  # Centered title
      legend.title = element_blank()  # Removes the word "Legend"
    ) +  
    labs(title = ctry, x = "Date", y = "Influenza Positive Cases")
  
  print(forecast_plot)
  
  return(data.frame(
    Country    = ctry,
    Train_MSE  = train_mse,  Test_MSE  = test_mse,
    Train_RMSE = train_rmse, Test_RMSE = test_rmse,
    Train_MAE  = train_mae,  Test_MAE  = test_mae
  ))
}

# Run Model
metrics_all <- do.call(rbind, lapply(countries, process_country))
write.csv(metrics_all, file.path(output_dir, "AllCountries_Metrics.csv"), row.names = FALSE)


