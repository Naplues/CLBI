library(effsize)

# 1: release precision,recall,far,ce,d2h,mcc,ifa,recall_20,ER,RI,ratio
# 2: precision
# 3: recall    ==
# 4: far       ==
# 5: ce        ==
# 6: d2h       ==
# 7: mcc       ==
# 8: ifa       ==
# 9: recall_20 ==
# 12: ratio    ==

calc_p_value <- function(target_data, glance_data) {
  p_value <- c()
  p_value <- c(p_value, wilcox.test(target_data[, 3], glance_data[, 3], paired = TRUE)$p.value)
  p_value <- c(p_value, wilcox.test(target_data[, 4], glance_data[, 4], paired = TRUE)$p.value)
  p_value <- c(p_value, wilcox.test(target_data[, 5], glance_data[, 5], paired = TRUE)$p.value)
  p_value <- c(p_value, wilcox.test(target_data[, 6], glance_data[, 6], paired = TRUE)$p.value)
  p_value <- c(p_value, wilcox.test(target_data[, 7], glance_data[, 7], paired = TRUE)$p.value)
  p_value <- c(p_value, wilcox.test(target_data[, 8], glance_data[, 8], paired = TRUE)$p.value)
  p_value <- c(p_value, wilcox.test(target_data[, 9], glance_data[, 9], paired = TRUE)$p.value)
  p_value <- c(p_value, wilcox.test(target_data[, 12], glance_data[, 12], paired = TRUE)$p.value)
  p_value
}

calc_cliff <- function(target_data, glance_data) {
  cliff <- c()
  cliff <- c(cliff, cliff.delta(target_data[, 3], glance_data[, 3])$estimate)
  cliff <- c(cliff, cliff.delta(target_data[, 4], glance_data[, 4])$estimate)
  cliff <- c(cliff, cliff.delta(target_data[, 5], glance_data[, 5])$estimate)
  cliff <- c(cliff, cliff.delta(target_data[, 6], glance_data[, 6])$estimate)
  cliff <- c(cliff, cliff.delta(target_data[, 7], glance_data[, 7])$estimate)
  cliff <- c(cliff, cliff.delta(target_data[, 8], glance_data[, 8])$estimate)
  cliff <- c(cliff, cliff.delta(target_data[, 9], glance_data[, 9])$estimate)
  cliff <- c(cliff, cliff.delta(target_data[, 12], glance_data[, 12])$estimate)
  cliff
}

calc_power <- function(target_data, glance_data) {
  power <- c()
  power <- c(power, cliff.delta(target_data[, 3], glance_data[, 3])$magnitude)
  power <- c(power, cliff.delta(target_data[, 4], glance_data[, 4])$magnitude)
  power <- c(power, cliff.delta(target_data[, 5], glance_data[, 5])$magnitude)
  power <- c(power, cliff.delta(target_data[, 6], glance_data[, 6])$magnitude)
  power <- c(power, cliff.delta(target_data[, 7], glance_data[, 7])$magnitude)
  power <- c(power, cliff.delta(target_data[, 8], glance_data[, 8])$magnitude)
  power <- c(power, cliff.delta(target_data[, 9], glance_data[, 9])$magnitude)
  power <- c(power, cliff.delta(target_data[, 12], glance_data[, 12])$magnitude)
  power
}

# TMI_LR, TMI_SVM, TMI_MNB, TMI_DT, TMI_RF, LineDP, PMD, CheckStyle, NGram, NGram_C
# Glance_MD, Glance_EA, Glance_LR
name <- c('recall', 'far', 'ce', 'd2h', 'mcc', 'ifa', 'recall_20', 'ratio')

NGram_data <- read.csv('D:/CLDP_data/Result/NGram/line_result/evaluation.csv', header = TRUE)
NGram_C_data <- read.csv('D:/CLDP_data/Result/NGram-C/line_result/evaluation.csv', header = TRUE)
TMI_LR_data <- read.csv('D:/CLDP_data/Result/TMI-LR/line_result/evaluation.csv', header = TRUE)
TMI_SVM_data <- read.csv('D:/CLDP_data/Result/TMI-SVM/line_result/evaluation.csv', header = TRUE)
TMI_MNB_data <- read.csv('D:/CLDP_data/Result/TMI-MNB/line_result/evaluation.csv', header = TRUE)
TMI_DT_data <- read.csv('D:/CLDP_data/Result/TMI-DT/line_result/evaluation.csv', header = TRUE)
TMI_RF_data <- read.csv('D:/CLDP_data/Result/TMI-RF/line_result/evaluation.csv', header = TRUE)
LineDP_data <- read.csv('D:/CLDP_data/Result/LineDP/line_result/evaluation.csv', header = TRUE)

PMD_data <- read.csv('D:/CLDP_data/Result/PMD/line_result/evaluation.csv', header = TRUE)
CheckStyle_data <- read.csv('D:/CLDP_data/Result/CheckStyle/line_result/evaluation.csv', header = TRUE)

glance_data <- read.csv('D:/CLDP_data/Result/Glance-LR/line_result/evaluation.csv', header = TRUE)

# ====================================== Calculate p values =====================================
# ============== RQ2 ===============
NGram <- calc_p_value(glance_data, NGram_data)
NGram_C <- calc_p_value(glance_data, NGram_C_data)
TMI_LR <- calc_p_value(glance_data, TMI_LR_data)
TMI_SVM <- calc_p_value(glance_data, TMI_SVM_data)
TMI_MNB <- calc_p_value(glance_data, TMI_MNB_data)
TMI_DT <- calc_p_value(glance_data, TMI_DT_data)
TMI_RF <- calc_p_value(glance_data, TMI_RF_data)
LineDP <- calc_p_value(glance_data, LineDP_data)

result <- data.frame(name, NGram, NGram_C, TMI_LR, TMI_SVM, TMI_MNB, TMI_DT, TMI_RF, LineDP)
write.table(result, '../../result/RQ2/Statistics/p_values.csv', row.names = FALSE, sep = ',')

# ============== RQ3 ===============
PMD <- calc_p_value(glance_data, PMD_data)
CheckStyle <- calc_p_value(glance_data, CheckStyle_data)

result <- data.frame(name, PMD, CheckStyle)
write.table(result, '../../result/RQ3/Statistics/p_values.csv', row.names = FALSE, sep = ',')

# ====================================== Calculate p values =====================================
# ============== RQ2 ===============
NGram <- calc_cliff(glance_data, NGram_data)
NGram_C <- calc_cliff(glance_data, NGram_C_data)
TMI_LR <- calc_cliff(glance_data, TMI_LR_data)
TMI_SVM <- calc_cliff(glance_data, TMI_SVM_data)
TMI_MNB <- calc_cliff(glance_data, TMI_MNB_data)
TMI_DT <- calc_cliff(glance_data, TMI_DT_data)
TMI_RF <- calc_cliff(glance_data, TMI_RF_data)
LineDP <- calc_cliff(glance_data, LineDP_data)

result <- data.frame(name, NGram, NGram_C, TMI_LR, TMI_SVM, TMI_MNB, TMI_DT, TMI_RF, LineDP)
write.table(result, '../../result/RQ2/Statistics/cliff.csv', row.names = FALSE, sep = ',')

# ============== RQ3 ===============
PMD <- calc_cliff(glance_data, PMD_data)
CheckStyle <- calc_cliff(glance_data, CheckStyle_data)

result <- data.frame(name, PMD, CheckStyle)
write.table(result, '../../result/RQ3/Statistics/cliff.csv', row.names = FALSE, sep = ',')

# ====================================== Calculate p values =====================================
# ============== RQ2 ===============
NGram <- calc_power(glance_data, NGram_data)
NGram_C <- calc_power(glance_data, NGram_C_data)
TMI_LR <- calc_power(glance_data, TMI_LR_data)
TMI_SVM <- calc_power(glance_data, TMI_SVM_data)
TMI_MNB <- calc_power(glance_data, TMI_MNB_data)
TMI_DT <- calc_power(glance_data, TMI_DT_data)
TMI_RF <- calc_power(glance_data, TMI_RF_data)
LineDP <- calc_power(glance_data, LineDP_data)

result <- data.frame(name, NGram, NGram_C, TMI_LR, TMI_SVM, TMI_MNB, TMI_DT, TMI_RF, LineDP)
write.table(result, '../../result/RQ2/Statistics/power.csv', row.names = FALSE, sep = ',')

# ============== RQ3 ===============
PMD <- calc_power(glance_data, PMD_data)
CheckStyle <- calc_power(glance_data, CheckStyle_data)

result <- data.frame(name, PMD, CheckStyle)
write.table(result, '../../result/RQ3/Statistics/power.csv', row.names = FALSE, sep = ',')