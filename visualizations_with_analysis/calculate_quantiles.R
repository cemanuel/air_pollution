packages <- c("plyr", "RcppCNPy", "ggplot2")
lapply(packages, require, character.only = TRUE)


labels.all <- npyLoad("datasets/ALL_EXAMPLES_for_week6_presentation_dataset.npy")

labels.all.txt <- read.table("datasets/ALL_EXAMPLES_for_week6_presentation_dataset.txt", header = TRUE)
## yes, both .npy and .txt files are the same: 361156 rows and 9 columns


bins = seq(0, 1, by = 0.1)

bins <- labels.all.txt %>% group_by(webcamId) %>% 
  do(data.frame(t(quantile(.$pm, probs = bins))))

write.csv(bins, file = "Output/Bins_by_webcamID.csv", row.names = FALSE)
write.table(bins, file = "Output/Bins_by_webcamID.txt", row.names = FALSE, sep = ",")


bins2 = seq(0, 1, by = 0.2)
labels.all.txt.quant <- labels.all.txt %>% group_by(webcamId) %>% 
  mutate(quantile = findInterval(pm, quantile(pm, probs = bins2)))
         

         
ggplot(labels.all.txt.quant, aes(pm)) + geom_density(color = "dodgerblue2") +
  scale_fill_brewer(guide="none")  + facet_wrap(~webcamId) + theme_light() +
  xlab("PM label") + ylab("Density") 


ggsave("Distribution.pdf", width = 25, height = 18, units = "cm")

