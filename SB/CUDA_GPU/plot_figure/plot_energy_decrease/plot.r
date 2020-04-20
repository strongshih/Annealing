library(ggplot2)

opt <- read.csv("./m1_stats.csv", header=FALSE, sep=" ")
max_ = 0.0
which_max = -1
min_ = 10000.0
which_min = -1
opt[2] = opt[2]*1000

for (i in c(1:100)) {
	if (opt[i*40,2] > max_) {
		max_ = opt[i*40,2]
		which_max = i
	}
	if (opt[i*40,2] < min_) {
		min_ = opt[i*40,2]
		which_min = i
	}
}

max_line_m1 = opt[(which_max*40-39):(which_max*40),2:3]
min_line_m1 = opt[(which_min*40-39):(which_min*40),2:3]

opt <- read.csv("./m2_stats.csv", header=FALSE, sep=" ")
max_ = 0.0
which_max = -1
min_ = 10000.0
which_min = -1
opt[2] = opt[2]*1000

for (i in c(1:100)) {
	if (opt[i*20,2] > max_) {
		max_ = opt[i*20,2]
		which_max = i
	}
	if (opt[i*20,2] < min_) {
		min_ = opt[i*20,2]
		which_min = i
	}
}

max_line_m2 = opt[(which_max*20-19):(which_max*20),2:3]
min_line_m2 = opt[(which_min*20-19):(which_min*20),2:3]

opt <- read.csv("./opt_stats.csv", header=FALSE, sep=" ")
max_ = 0.0
which_max = -1
min_ = 10000.0
which_min = -1
opt[2] = opt[2]*1000

for (i in c(1:100)) {
	if (opt[i*20,2] > max_) {
		max_ = opt[i*20,2]
		which_max = i
	}
	if (opt[i*20,2] < min_) {
		min_ = opt[i*20,2]
		which_min = i
	}
}

max_line_opt = opt[(which_max*20-19):(which_max*20),2:3]
min_line_opt = opt[(which_min*20-19):(which_min*20),2:3]

plot(max_line_m1, type="l", col="green", xlab="ms", ylab="E", xlim=range(1e-3:10), ylim=range(-70000:0), las=1, log="x")
lines(min_line_m1, lty=2, col="green")
abline(h=-60278, col="black", lty=2)
abline(h=-62059, col="black", lty=4)

plot(max_line_m1, type="l", col="green", xlab="ms", ylab="E", xlim=range(1e-3:10), ylim=range(-70000:0), las=1, log="x")
lines(min_line_m1, lty=2, col="green")
lines(max_line_m2, col="orange")
lines(min_line_m2, lty=2, col="orange")
abline(h=-60278, col="black", lty=2)
abline(h=-62059, col="black", lty=4)

plot(max_line_m1, type="l", col="green", xlab="ms", ylab="E", xlim=range(1e-3:10), ylim=range(-70000:0), las=1, log="x")
lines(min_line_m1, lty=2, col="green")
lines(max_line_m2, col="orange")
lines(min_line_m2, lty=2, col="orange")
lines(max_line_opt, col="black")
lines(min_line_opt, lty=2, col="black")
abline(h=-60278, col="black", lty=2)
abline(h=-62059, col="black", lty=4)

### HNN
best_m1 = 10000.0
worst_m1 = 0.0
avg_m1 = 0.0
count = 0.0
opt <- read.csv("./m1_stats.csv", header=FALSE, sep=" ")
for (i in c(0:99)) {
	for (j in c(1:40)) {
		if (opt[i*40+j,3] < -62059) {
			if (opt[i*40+j,2] < best_m1) {
				best_m1 = opt[i*40+j,2]
			}
			if (opt[i*40+j,2] > worst_m1) {
				worst_m1 = opt[i*40+j,2]
			}
			avg_m1 = avg_m1 + opt[i*40+j,2]
			count = count + 1
			break
		}

	}
}
avg_m1 = avg_m1/count

best_m2 = 10000.0
worst_m2 = 0.0
avg_m2 = 0.0
count = 0.0
opt <- read.csv("./m2_stats.csv", header=FALSE, sep=" ")
for (i in c(0:99)) {
	for (j in c(1:20)) {
		if (opt[i*20+j,3] < -62059) {
			if (opt[i*20+j,2] < best_m2) {
				best_m2 = opt[i*20+j,2]
			}
			if (opt[i*20+j,2] > worst_m2) {
				worst_m2 = opt[i*20+j,2]
			}
			avg_m2 = avg_m2 + opt[i*20+j,2]
			count = count + 1
			break
		}

	}
}
avg_m2 = avg_m2/count

best_opt = 10000.0
worst_opt = 0.0
avg_opt = 0.0
count = 0.0
opt <- read.csv("./opt_stats.csv", header=FALSE, sep=" ")
for (i in c(0:99)) {
	for (j in c(1:20)) {
		if (opt[i*20+j,3] < -62059) {
			if (opt[i*20+j,2] < best_opt) {
				best_opt = opt[i*20+j,2]
			}
			if (opt[i*20+j,2] > worst_opt) {
				worst_opt = opt[i*20+j,2]
			}
			avg_opt = avg_opt + opt[i*20+j,2]
			count = count + 1
			break
		}

	}
}
avg_opt = avg_opt/count

"HNN"
"m1"
c(best_m1, avg_m1, worst_m1)*1000
"m2"
c(best_m2, avg_m2, worst_m2)*1000
"opt"
c(best_opt, avg_opt, worst_opt)*1000

### GW-SDP
best_m1 = 10000.0
worst_m1 = 0.0
avg_m1 = 0.0
count = 0.0
opt <- read.csv("./m1_stats.csv", header=FALSE, sep=" ")
for (i in c(0:99)) {
	for (j in c(1:40)) {
		if (opt[i*40+j,3] < -60278) {
			if (opt[i*40+j,2] < best_m1) {
				best_m1 = opt[i*40+j,2]
			}
			if (opt[i*40+j,2] > worst_m1) {
				worst_m1 = opt[i*40+j,2]
			}
			avg_m1 = avg_m1 + opt[i*40+j,2]
			count = count + 1
			break
		}

	}
}
avg_m1 = avg_m1/count

best_m2 = 10000.0
worst_m2 = 0.0
avg_m2 = 0.0
count = 0.0
opt <- read.csv("./m2_stats.csv", header=FALSE, sep=" ")
for (i in c(0:99)) {
	for (j in c(1:20)) {
		if (opt[i*20+j,3] < -60278) {
			if (opt[i*20+j,2] < best_m2) {
				best_m2 = opt[i*20+j,2]
			}
			if (opt[i*20+j,2] > worst_m2) {
				worst_m2 = opt[i*20+j,2]
			}
			avg_m2 = avg_m2 + opt[i*20+j,2]
			count = count + 1
			break
		}

	}
}
avg_m2 = avg_m2/count

best_opt = 10000.0
worst_opt = 0.0
avg_opt = 0.0
count = 0.0
opt <- read.csv("./opt_stats.csv", header=FALSE, sep=" ")
for (i in c(0:99)) {
	for (j in c(1:20)) {
		if (opt[i*20+j,3] < -60278) {
			if (opt[i*20+j,2] < best_opt) {
				best_opt = opt[i*20+j,2]
			}
			if (opt[i*20+j,2] > worst_opt) {
				worst_opt = opt[i*20+j,2]
			}
			avg_opt = avg_opt + opt[i*20+j,2]
			count = count + 1
			break
		}

	}
}
avg_opt = avg_opt/count

"GW-SDP"
"m1"
c(best_m1, avg_m1, worst_m1)*1000
"m2"
c(best_m2, avg_m2, worst_m2)*1000
"opt"
c(best_opt, avg_opt, worst_opt)*1000