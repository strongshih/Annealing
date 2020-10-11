import math

P = 128
cnt = 0
for i in range(P):
    print("                int temp"+str(cnt)+" = localJ[m]["+str(i)+"] * lspin[m][j+"+str(i)+"];")
    cnt += 1

print("")

layer = int(math.log(P,2))
for i in range(layer):
    off = int(2**(layer - i - 1))
    start = cnt - 2*off
    for j in range(off):
        print("                int temp"+str(cnt)+" = temp"+str(start+j)+" + temp"+str(start+j+off) + ";")
        cnt += 1
    print("")
print("                lfield[m] += temp" + str(cnt-1))
