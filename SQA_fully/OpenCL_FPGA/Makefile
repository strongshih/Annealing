AOCL_COMPILE_CONFIG=$(shell aocl compile-config)
AOCL_LINK_CONFIG=$(shell aocl link-config)

all: Ising Ising.o Ising-kernel.aocx

Ising : Ising.o
    gcc -o Ising Ising.o $(AOCL_LINK_CONFIG) -lm

Ising.o : Ising.c
    gcc -c Ising.c $(AOCL_COMPILE_CONFIG)

Ising-kernel.aocx : Ising-kernel.cl
ifdef emu
    aoc -march=emulator -v Ising-kernel.cl -o Ising-kernel.aocx
else
    aoc Ising-kernel.cl -o Ising-kernel.aocx -board=c5gt -report -v -profile
endif

test :
ifdef emu
    CL_CONTEXT_EMULATOR_DEVICE_INTELFPGA=1 ./Ising ./Ising-kernel.aocx tests/1.txt
else
    aocl program acl0 Ising-kernel.aocx
    ./Ising ./Ising-kernel.aocx tests/1.txt
endif

clean: 
    rm -rf Ising.o Ising Ising-kernel.aoco Ising-kernel.aocx Ising-kernel/
