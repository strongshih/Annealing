for i in $(seq 2 1 6);
do
	for j in $(seq 10 1 14);
	do
		var1=$(echo "2^$i" | bc)
		var2=$(echo "2^$j" | bc)
		echo $var1 $var2;
		sed -i "s/^#define M .*$/#define M $var1/g" pre_calculate.cu
		sed -i "s/^#define N .*$/#define N $var2/g" pre_calculate.cu
		make
		./pre_calculate-cuda tests/$var2.txt
	done
done
