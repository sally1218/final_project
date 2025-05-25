library(caret)
library(randomForest)

setwd("D:/sally_school")
data <- read.csv("data.csv")

# 移除不需要的欄位
if("ID" %in% colnames(data)) data <- subset(data, select = -ID)
if("No_Pation" %in% colnames(data)) data <- subset(data, select = -No_Pation)

# 確保類別為因子，且指定類別順序
data$CLASS <- factor(data$CLASS, levels = c("N", "P", "Y"))

# 轉換 Gender 為因子
data$Gender <- as.factor(data$Gender)

# 顯示缺失值狀況
print(colSums(is.na(data)))

# 使用中位數填補缺失值（只處理數值欄位）
numeric_cols <- c("AGE", "Urea", "Cr", "HbA1c", "Chol", "TG", "HDL", "LDL", "VLDL", "BMI")
for (col in numeric_cols) {
  data[[col]][is.na(data[[col]])] <- median(data[[col]], na.rm = TRUE)
}

# 刪除 CLASS 為 NA 的列
data <- data[!is.na(data$CLASS), ]

# 再次檢查是否還有缺失
print(colSums(is.na(data)))

# 資料分割
set.seed(123)
index <- createDataPartition(data$CLASS, p = 0.8, list = FALSE)
train <- data[index, ]
test <- data[-index, ]

# 轉換類別為 droplevels，避免空類別
train$CLASS <- droplevels(train$CLASS)
test$CLASS <- droplevels(test$CLASS)

# 建立 caret 訓練控制器
control <- trainControl(method = "cv", number = 5, classProbs = TRUE)

# 訓練 Random Forest 模型
set.seed(123)
rf_caret <- train(CLASS ~ Gender + AGE + Urea + Cr + HbA1c + Chol + TG + HDL + LDL + VLDL + BMI,
                  data = train,
                  method = "rf",
                  trControl = control,
                  tuneLength = 5,
                  importance = TRUE)

# 顯示最佳參數與準確率
print(rf_caret)

# 預測測試集
pred <- predict(rf_caret, newdata = test)

# 混淆矩陣與模型表現
confMat <- confusionMatrix(pred, test$CLASS)
print(confMat)

# 顯示變數重要性圖
varImpPlot(rf_caret$finalModel)

# 顯示準確率（百分比形式）
accuracy <- confMat$overall['Accuracy']
cat("模型準確率：", round(accuracy * 100, 2), "%\n")

# 儲存模型
save(rf_caret, file = "C:/xampp/htdocs/final_project/final_project.RData")
