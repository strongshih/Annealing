import random

range_list = [-3, -2, -1, 0, 1, 2, 3]

#    0  1  2  3  4  5  6  7     
#    8  9 10 11 12 13 14 15   
#   16 17 18 19 20 21 22 23     s1  s2  s3
#   24 25 26 27 28 29 30 31     s4   h  s5
#   32 33 34 35 36 37 38 39     s6  s7  s8
#   40 41 42 43 44 45 46 47
#   48 49 50 51 52 53 54 55
#   56 57 58 59 60 61 62 63

for num in [8, 16, 32, 64, 128]:
    EDGE = num
    N = EDGE*EDGE
    f = open(str(N) + ".txt", "a")
    f.write(str(N) + "\n")

    hasAlready = {}

    for row in range(EDGE):
        for column in range(EDGE):
            currentSpin = (column + row*EDGE)
            
            # generate external field
            r = random.choice(range_list) 
            f.write(str(currentSpin) + " " + str(currentSpin) + " " + str(r) + "\n")
                        
            # generate interactions
            if not (currentSpin%EDGE == 0 or currentSpin < EDGE):                  #1
                e = currentSpin-EDGE-1
                if not (e*N+currentSpin in hasAlready):
                    if random.random() > 0.5:
                        r = random.choice(range_list) 
                        f.write(str(currentSpin) + " " + str(e) + " " + str(r) + "\n")
                        hasAlready[e*N+currentSpin] = 1
                        hasAlready[currentSpin*N+e] = 1
            if not (currentSpin < EDGE):                                           #2
                e = currentSpin-EDGE
                if not (e*N+currentSpin in hasAlready):
                    if random.random() > 0.5:
                        r = random.choice(range_list) 
                        f.write(str(currentSpin) + " " + str(e) + " " + str(r) + "\n")
                        hasAlready[e*N+currentSpin] = 1
                        hasAlready[currentSpin*N+e] = 1
            if not ((currentSpin+1)%EDGE == 0 or currentSpin < EDGE):              #3
                e = currentSpin-EDGE+1
                if not (e*N+currentSpin in hasAlready):
                    if random.random() > 0.5:
                        r = random.choice(range_list) 
                        f.write(str(currentSpin) + " " + str(e) + " " + str(r) + "\n")
                        hasAlready[e*N+currentSpin] = 1
                        hasAlready[currentSpin*N+e] = 1
            if not (currentSpin%EDGE == 0):                                        #4
                e = currentSpin-1
                if not (e*N+currentSpin in hasAlready):
                    if random.random() > 0.5:
                        r = random.choice(range_list) 
                        f.write(str(currentSpin) + " " + str(e) + " " + str(r) + "\n")
                        hasAlready[e*N+currentSpin] = 1
                        hasAlready[currentSpin*N+e] = 1
            if not ((currentSpin+1)%EDGE == 0):                                    #5
                e = currentSpin+1
                if not (e*N+currentSpin in hasAlready):
                    if random.random() > 0.5:
                        r = random.choice(range_list) 
                        f.write(str(currentSpin) + " " + str(e) + " " + str(r) + "\n")
                        hasAlready[e*N+currentSpin] = 1
                        hasAlready[currentSpin*N+e] = 1
            if not (currentSpin%EDGE == 0 or currentSpin >= EDGE*(EDGE-1)):        #6
                e = currentSpin+EDGE-1
                if not (e*N+currentSpin in hasAlready):
                    if random.random() > 0.5:
                        r = random.choice(range_list) 
                        f.write(str(currentSpin) + " " + str(e) + " " + str(r) + "\n")
                        hasAlready[e*N+currentSpin] = 1
                        hasAlready[currentSpin*N+e] = 1
            if not (currentSpin >= EDGE*(EDGE-1)):                                 #7
                e = currentSpin+EDGE
                if not (e*N+currentSpin in hasAlready):
                    if random.random() > 0.5:
                        r = random.choice(range_list) 
                        f.write(str(currentSpin) + " " + str(e) + " " + str(r) + "\n")
                        hasAlready[e*N+currentSpin] = 1
                        hasAlready[currentSpin*N+e] = 1
            if not ((currentSpin+1)%EDGE == 0 or currentSpin >= EDGE*(EDGE-1)):    #8
                e = currentSpin+EDGE+1
                if not (e*N+currentSpin in hasAlready):
                    if random.random() > 0.5:
                        r = random.choice(range_list) 
                        f.write(str(currentSpin) + " " + str(e) + " " + str(r) + "\n")
                        hasAlready[e*N+currentSpin] = 1
                        hasAlready[currentSpin*N+e] = 1
    f.close()
