# B. Computing, Chips, Compilers

**Contents**

- [B.1 Chips: Physical Evaluators]()
    - [Instruction Set Architecture: RISC-V]()
    - [von Neumann Architecture]()
        - [Microprocessor's Microarchitecture: Single-cycle, Multi-cycle, Pipelined, Out of Order]()
        - [Memory:]()
    - [Performance Analysis and Tuning]()
<!-- - [B.3 Compilers: Virtual Translators]() -->

zip the stack from the bottom up.


# B.1 Chips: Physical Evaluators

## Instruction Set Architecture: RISC-V

## von Neumann Architecture
(preliminary discusion of logical design of electronic computing instrument 1946)
- processor: control + data
- memory (instructions + data)
- i/o

PROCESSOR: evaluates the instruction (computes)
--------------------------------------------------------------------------------



MEMORY
--------------------------------------------------------------------------------
most von neumann architectures are byte adressable.
changing with machine learning. fp4. fp2?? energy requirements tradeoff.

word addressable (32/64bits at a time) 4hex/8hex digits
byte addressable (8bits=1 byte at a time)

how are the bytes in a word ordered? little endian (MSB <- LSB)/big endian (LSB -> MSB).



key(address) value(word)
------------------------
000            u8
001            u8
010             .
011             .
100             .
101
110            u8


### Microprocessor's Microarchitecture: Single-cycle, Multi-cycle, Pipelined

#### Single-cycle

state: sequential logic
processor: combinational logic (ALU)
- datapath
- control unit

core (processor):
4 stateful elements
core: 1. processor counter 2.registers (FETCH/load.DECODE)
memory: 1. instruction memory 2. data memory (STORE)
- (instr. mem is ROM for now)
- the multicycle uarch will have more realistic unified memory (instr + data)


design process
*start with stateful elements (LOAD/STORE)
add blocks of combinational (functional) logic (EXEC)
to evaluate state_i -> p(instr_i, data_i) -> state_i+1


---------------------------------------
instruction processing loop.
takes single machine clock cycle.
in single-cycle machine.
    - FETCH
    - DECODE
    - EXEC
    - STORE

 s1 --P(i0)-> s0
 |             |
 ---------------
single-cycle machine:  1 instr/clock
(no intermediate programmer invisible states)
cons: slowest instr. determines cycle time.

contra multi-cycle machine: 1 instr/N clocks
- updates made *during* instruction execution
- slowest "stage". determines cycle time.
-----------------------------------------





#### Multi-cycle

#### Pipelined

### Memory

### Performance Analysis and Tuning
- lab(micro)benchmarks: eliminate nondeterminism and noise
- prod benchmaks: include nondeter. and noise. execution/wall clock time (latency)
noise variance with clock freq (DFS: dynamic freq scalignpower scaling)
    ==> variance with execution time (wall clock time)
    - tools like temci??? exist to reduce variance
- hotspots

"active benchmarking": *guilty* until proven innocent

https://rcs.uwaterloo.ca/~ali/cs854-f23/papers/onelevel.pdf
criterion.rs

# B.2 Compiler: Virtual Translators