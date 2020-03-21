import random

range_list = [-3, -2, -1, 0, 1, 2, 3]
count = {}

for num in range(10):
    for i in range(512):
        count[i] = 0
    f = open(str(num) +".txt", "a")
    f.write("512\n")
    for i in range(512):
        for j in range(i+1, 512):
            if random.random() > 0.5:
                r = random.choice(range_list)
                f.write(str(i) + " " + str(j) + " " + str(r) + "\n")
    f.close()
