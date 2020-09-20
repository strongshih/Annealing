for j in $(seq 10 1 14);
do
	var2=$(echo "2^$j" | bc)
	echo $var2;
	sed -i "s/^#define N .*$/#define N $var2/g" SCA.cu
	nvcc SCA3.cu
	./a.out tests/$var2.txt
done
