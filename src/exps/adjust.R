# Title     : TODO
# Objective : TODO
# Created by: gzq-712
# Created on: 2021/12/27

data <- read.csv('../../result/RQ2/Statistics/p_values.csv', header = TRUE)

result <- c()
result <- c(result, data[, 2])
result <- c(result, data[, 3])
result <- c(result, data[, 4])
result <- c(result, data[, 5])
result <- c(result, data[, 6])
result <- c(result, data[, 7])
result <- c(result, data[, 8])
result <- c(result, data[, 9])

result <- p.adjust(result, method = "BH")

data[, 2] <- result[1:8]
data[, 3] <- result[9:16]
data[, 4] <- result[17:24]
data[, 5] <- result[25:32]
data[, 6] <- result[33:40]
data[, 7] <- result[41:48]
data[, 8] <- result[49:56]
data[, 9] <- result[57:64]

write.table(data, '../../result/RQ2/Statistics/BH_p_values.csv', row.names = FALSE, sep = ',')


data <- read.csv('../../result/RQ3/Statistics/p_values.csv', header = TRUE)

result <- c()
result <- c(result, data[, 2])
result <- c(result, data[, 3])

result <- p.adjust(result, method = "BH")

data[, 2] <- result[1:8]
data[, 3] <- result[9:16]

write.table(data, '../../result/RQ3/Statistics/BH_p_values.csv', row.names = FALSE, sep = ',')