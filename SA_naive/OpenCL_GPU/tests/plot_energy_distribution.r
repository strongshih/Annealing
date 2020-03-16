library(plyr)
library(ggplot2)
# optimized
input1 <- read.csv("./energy1.csv", header=FALSE, sep=",") # optimized
a = rep(input1[1,1], input1[1,2])
for(i in 2:dim(input1)[1])
  a = c(rep(input1[i,1], input1[i,2]), a)
b <- as.numeric(unlist(read.csv("./output.txt", header=FALSE, sep=","))) # OpenCL simple
data <- data.frame(
  which = factor(rep(c("Optimized", "Simple"), each=1024), levels=c("Optimized", "Simple")),
  energy = c(a, b)
)

mu <- ddply(data, "which", summarise, grp.mean=mean(energy))
p <- ggplot(data, aes(x=energy, color=which)) + 
  geom_histogram(fill="white", position="dodge") +
  ggtitle("Energy distribution under different settings") +
  geom_vline(data=mu, aes(xintercept=grp.mean, color=which), linetype="dashed") +
  theme(legend.position="bottom")
p
