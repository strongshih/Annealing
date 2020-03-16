#!/bin/bash

### get optimized SA code
if [ ! -f "optimized_SA.tar.gz" ]; then
	wget http://csie.ntu.edu.tw/~b05902108/optimized_SA.tar.gz
	tar xvf optimized_SA.tar.gz
fi
cd anc/
make
cd ../

### generate test instances
if [ ! -f "99.txt" ]; then
	python fully.py
fi

### generate data for heatmap
sweep=200
repeat=1024
thread=32
P=0.99
for i in {0..99}; do
	# first
    t=$(./anc/an_ss_ge_fi_vdeg_omp -v -l "./$i.txt" -sched lin -s $sweep -r $repeat -t $thread | grep '#work' | awk '{print $4}')
    time=$(echo "scale=10 ; $t / $((repeat))" | bc)
    big=$(./anc/an_ss_ge_fi_vdeg_omp -v -l "./$i.txt" -sched lin -s $sweep -r $repeat -t $thread | grep '^   ' | awk '{print $1}' | tail -n 1)
    small=$(./anc/an_ss_ge_fi_vdeg_omp -v -l "./$i.txt" -sched lin -s $sweep -r $repeat -t $thread | grep '^   ' | awk '{print $1}' | head -n 1)
    difference=$(( big - small ))
    partial=$(( small + difference/5 )) # 80% --> success
    num_of_success=$(./anc/an_ss_ge_fi_vdeg_omp -v -l "./$i.txt" -sched lin -s $sweep -r $repeat -t $thread  | grep '^   ' | awk -v par="$partial" '$1 < par {sum += $2} END {print sum}')
    success_rate=$( echo "scale=10; $num_of_success / $((repeat))" | bc)
    annealing_time=$( echo "$time*l(1-$P)/l(1-$success_rate)" | bc -l)
    echo "$annealing_time" >> dist1.txt

	# second
	cal_time="$(time ( ../Ising-cuda "$i".txt ) 2>&1 1>/dev/null )"
	time=$(echo $cal_time | awk '{print $2}' | sed 's/^.*m//g' | sed 's/s.*$//g')
	time=$(echo "scale=10; $time / 1024" | bc)
    num_of_success=$(cat output.txt | awk -v par="$partial" '$1 < par {sum += 1} END {print sum}')
    success_rate=$( echo "scale=10; $num_of_success / $((repeat))" | bc)
    annealing_time=$( echo "$time*l(1-$P)/l(1-$success_rate)" | bc -l)
    echo "$annealing_time" >> dist2.txt
done
# Install R and packages before plotting
  # sudo apt install r-base
  # Rscript -e 'install.packages("tidyverse")'
  # Rscript -e 'install.packages("ggplot2")'
  # Rscript -e 'install.packages("plyr")'
Rscript plot_annealing_time.r
mv Rplots.pdf annealing_time.pdf

### Plot distribution
./anc/an_ss_ge_fi_vdeg_omp -v -l "./99.txt" -sched lin -s 200 -r 1024 -t 32 | grep "^   " | awk '{print $1 "," $2}' > energy1.csv
Rscript plot_energy_distribution.r
mv Rplots.pdf energy_distribution.pdf

### Plot energy descreasing
for i in $(seq 1 1 20);
do
	echo $i
    sed -i 's/^#define SWEEP.*$/#define SWEEP '"$i"'00/g' ../Ising.cu
	cd .. && make && cd tests
	cal_time="$(time ( ../Ising-cuda 1.txt ) 2>&1 1>/dev/null )"
    time=$(echo $cal_time | awk '{print $2}' | sed 's/^.*m//g' | sed 's/s.*$//g')
    time=$(echo "scale=10; $time / 1024" | bc)
    smallest=$(cat output.txt | sort | uniq | tail -n 1)
    echo "$smallest,$time" >> decrease.csv
done
sed -i 's/^#define SWEEP.*$/#define SWEEP 200/g' ../Ising.cu
Rscript plot_energy_decrease.r
mv Rplots.pdf plot_energy_decrease.pdf
	
