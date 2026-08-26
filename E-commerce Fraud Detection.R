# ============================================================
# BIG DATA ANALYTICS PROJECT
# Real-Time E-Commerce Transaction Fraud Detection using R
# ============================================================

# 1. Install required packages
install.packages(c("dplyr", "ggplot2", "caret", "randomForest"))

# 2. Load libraries
library(dplyr)
library(ggplot2)
library(caret)
library(randomForest)

# ============================================================
# 3. Create / Load Dataset
# ============================================================

# If you have a CSV file, use:
# data <- read.csv("transactions.csv")

# Sample transaction dataset
set.seed(123)

n <- 10000

data <- data.frame(
  Transaction_ID = 1:n,
  Transaction_Amount = round(runif(n, 100, 50000), 2),
  Transaction_Hour = sample(0:23, n, replace = TRUE),
  Customer_Age = sample(18:70, n, replace = TRUE),
  Previous_Transactions = sample(1:100, n, replace = TRUE),
  International = sample(c(0, 1), n, replace = TRUE),
  Device_Change = sample(c(0, 1), n, replace = TRUE)
)

# ============================================================
# 4. Generate Fraud Labels
# ============================================================

fraud_probability <- 
  0.01 +
  0.00002 * data$Transaction_Amount +
  0.08 * data$International +
  0.10 * data$Device_Change +
  0.05 * (data$Transaction_Hour < 5)

fraud_probability <- pmin(fraud_probability, 0.8)

data$Fraud <- rbinom(
  n,
  size = 1,
  prob = fraud_probability
)

# Convert Fraud into categorical variable
data$Fraud <- factor(
  data$Fraud,
  levels = c(0, 1),
  labels = c("Normal", "Fraud")
)

# ============================================================
# 5. Display Dataset
# ============================================================

head(data)

# Dataset dimensions
dim(data)

# Dataset structure
str(data)

# Summary
summary(data)

# ============================================================
# 6. Data Cleaning
# ============================================================

# Check missing values
colSums(is.na(data))

# Remove duplicate transactions
data <- data %>%
  distinct()

# ============================================================
# 7. Exploratory Data Analysis
# ============================================================

# Number of normal and fraudulent transactions
table(data$Fraud)

# Fraud percentage
fraud_percentage <- prop.table(table(data$Fraud)) * 100

print(fraud_percentage)

# ============================================================
# 8. Visualization - Fraud Distribution
# ============================================================

ggplot(data, aes(x = Fraud)) +
  geom_bar() +
  labs(
    title = "Normal vs Fraudulent Transactions",
    x = "Transaction Type",
    y = "Number of Transactions"
  ) +
  theme_minimal()

# ============================================================
# 9. Transaction Amount Analysis
# ============================================================

ggplot(data, aes(x = Fraud, y = Transaction_Amount)) +
  geom_boxplot() +
  labs(
    title = "Transaction Amount vs Fraud",
    x = "Transaction Type",
    y = "Transaction Amount"
  ) +
  theme_minimal()

# ============================================================
# 10. Fraud by Transaction Hour
# ============================================================

ggplot(data, aes(x = Transaction_Hour, fill = Fraud)) +
  geom_bar(position = "dodge") +
  labs(
    title = "Fraulent Transactions by Hour",
    x = "Transaction Hour",
    y = "Number of Transactions"
  ) +
  theme_minimal()

# ============================================================
# 11. Fraud by International Transaction
# ============================================================

ggplot(data, aes(x = factor(International), fill = Fraud)) +
  geom_bar(position = "dodge") +
  labs(
    title = "Fraud Analysis for International Transactions",
    x = "International Transaction (0 = No, 1 = Yes)",
    y = "Number of Transactions"
  ) +
  theme_minimal()

# ============================================================
# 12. Split Dataset into Training and Testing
# ============================================================

set.seed(123)

train_index <- createDataPartition(
  data$Fraud,
  p = 0.80,
  list = FALSE
)

train_data <- data[train_index, ]
test_data <- data[-train_index, ]

# ============================================================
# 13. Random Forest Model
# ============================================================

model <- randomForest(
  Fraud ~ Transaction_Amount +
    Transaction_Hour +
    Customer_Age +
    Previous_Transactions +
    International +
    Device_Change,
  data = train_data,
  ntree = 100
)

print(model)

# ============================================================
# 14. Feature Importance
# ============================================================

importance(model)

varImpPlot(
  model,
  main = "Important Factors for Fraud Detection"
)

# ============================================================
# 15. Predict Fraud on Test Data
# ============================================================

prediction <- predict(
  model,
  newdata = test_data
)

# Display predictions
head(prediction)

# ============================================================
# 16. Confusion Matrix
# ============================================================

confusion <- confusionMatrix(
  prediction,
  test_data$Fraud
)

print(confusion)

# ============================================================
# 17. Accuracy
# ============================================================

accuracy <- confusion$overall["Accuracy"]

print(paste(
  "Model Accuracy:",
  round(accuracy * 100, 2),
  "%"
))

# ============================================================
# 18. Real-Time Transaction Prediction
# ============================================================

real_time_transaction <- data.frame(
  Transaction_Amount = 45000,
  Transaction_Hour = 2,
  Customer_Age = 25,
  Previous_Transactions = 10,
  International = 1,
  Device_Change = 1
)

real_time_prediction <- predict(
  model,
  newdata = real_time_transaction
)

print("Real-Time Transaction Result:")
print(real_time_prediction)

# ============================================================
# 19. Fraud Detection Function
# ============================================================

detect_fraud <- function(
    amount,
    hour,
    age,
    previous_transactions,
    international,
    device_change
) {

  transaction <- data.frame(
    Transaction_Amount = amount,
    Transaction_Hour = hour,
    Customer_Age = age,
    Previous_Transactions = previous_transactions,
    International = international,
    Device_Change = device_change
  )

  result <- predict(
    model,
    newdata = transaction
  )

  if (result == "Fraud") {
    print("WARNING: FRAUDULENT TRANSACTION DETECTED!")
  } else {
    print("Transaction is NORMAL.")
  }

  return(result)
}

# ============================================================
# 20. Test Real-Time Fraud Detection
# ============================================================

detect_fraud(
  amount = 48000,
  hour = 3,
  age = 22,
  previous_transactions = 5,
  international = 1,
  device_change = 1
)

# ============================================================
# END OF PROJECT
# ============================================================
