par(mfrow=c(2,1))

t = seq(0.1,20,0.1)
speed <- read.csv("./speed.txt", header=FALSE, sep=",")
pos <- read.csv("./pos.txt", header=FALSE, sep=",")
speed <- as.numeric(speed[1:200,1])
pos <- as.numeric(pos[1:200,1])

plot(t, speed, type="l", xlab="time", ylab="speed", col="red")
plot(t, pos, type="l", xlab="time", ylab="position", col="blue")