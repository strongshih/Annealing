import random

range_list = [-1, 0, 1]
count = {}

for num in range(1):
    for i in range(1024):
        count[i] = 0
    f = open("1024.txt", "a")
    f.write("1024\n")
    for i in range(1024):
        for j in range(i+1, 1024):
            if random.random() > 0.5:
                r = random.choice(range_list)
                f.write(str(i) + " " + str(j) + " " + str(r) + "\n")
    f.close()
