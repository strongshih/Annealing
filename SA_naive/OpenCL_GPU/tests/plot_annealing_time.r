library(ggplot2)
library(tidyverse)
input1 <- read.csv("./dist1.txt", header=FALSE, sep=",")
input2 <- read.csv("./dist2.txt", header=FALSE, sep=",")
x <- as.numeric(input1[1:100,1])*1000 # optimized
y <- as.numeric(input2[1:100,1])*1000 # OpenCL simple
ggplot(, aes(x=x, y=y)) +
  geom_bin2d(bins = 40) +
  theme_light() + scale_fill_distiller(palette = "OrRd", direction = -1) +
  scale_x_log10(limits = c(0.001, 10000)) + 
  scale_y_log10(limits = c(0.001, 10000)) +
  xlab('Total time of Optimized (ms)') +
  ylab('Total time of OpenCL simple (ms)') +
  ggtitle('OpenCL simple vs Optimized') + geom_abline(intercept = 0, slope = 1, lwd=0.1)
