AOCL_COMPILE_CONFIG=$(shell aocl compile-config)
AOCL_LINK_CONFIG=$(shell aocl link-config)

all: king king.o king-kernel.aocx

king : king.o
    gcc -o king king.o $(AOCL_LINK_CONFIG) -lm

king.o : king.c
    gcc -c king.c $(AOCL_COMPILE_CONFIG)

king-kernel.aocx : king-kernel.cl
ifdef emu
    aoc -march=emulator -v king-kernel.cl -o king-kernel.aocx
else
    aoc king-kernel.cl -o king-kernel.aocx -board=c5gt -report -v -profile
endif

test :
ifdef emu
    CL_CONTEXT_EMULATOR_DEVICE_INTELFPGA=1 ./king ./king-kernel.aocx tests/1024.txt
else
    aocl program acl0 king-kernel.aocx
    ./king ./king-kernel.aocx tests/1024.txt
endif

clean: 
    rm -rf king.o king king-kernel.aoco king-kernel.aocx king-kernel/
