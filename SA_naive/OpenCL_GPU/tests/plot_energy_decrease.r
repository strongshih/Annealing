input <- read.csv("./decrease.csv", header=FALSE, sep=",") 
x <- as.numeric(unlist(input[2]))*1000
y <- as.numeric(unlist(input[1]))
plot(x, y, type="b", col="blue", lwd=2, pch=15, xlab="time (ms)", ylab="energy")
title("Energy decrease versus time")
legend(0,2.8,c(""), lwd=c(5,2), col=c("blue"), pch=c(15), y.intersp=1.5)
