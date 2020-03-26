import random

range_list = [-3, -2, -1, 0, 1, 2, 3]
count = {}

for num in range(1):
    for i in range(4096):
        count[i] = 0
    f = open(str(num) +".txt", "a")
    f.write("4096\n")
    for i in range(4096):
        for j in range(i+1, 4096):
            if random.random() > 0.5:
                r = random.choice(range_list)
                f.write(str(i) + " " + str(j) + " " + str(r) + "\n")
    f.close()
