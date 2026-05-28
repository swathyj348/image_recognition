wine_df <- read.csv('wine_quality.csv')
wine_df$high_quality <- ifelse(wine_df$quality >= 7, 1, 0)
summary_df <- data.frame(
  metric = c('rows', 'columns', 'high_quality_rate'),
  value = c(nrow(wine_df), ncol(wine_df), mean(wine_df$high_quality))
)
write.csv(summary_df, 'wine_agent/outputs/r_summary.csv', row.names = FALSE)
print(summary_df)
