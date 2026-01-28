## Lab 5 Part 1 Discussion
Please answer the following questions.

### Your Name
Aditya Mavle

### Briefly Explain What is MSCCLang-DSL. [0.5 points]
MSCCLang-DSL is a domain specific language embedded in Python for specifying collective communication algorithms through chunk oriented operations like copy and reduce. It offers a high level of abstraction for routing data chunks across GPUs, without dealing with low level details like data races or deadlocks. It still achieves performance of hand written kernels.

### How Many Types of Buffers Each GPU Have in MSCCLang? Which are They? [0.5 points]
- Number of Buffer Types: 3
- They are: As per the paper, MSCCLang exposes GPU memory as named buffers on the following ranks : 1. Input Buffer 2. Output Buffer 3. Scratch : Used for temporary storage

### Brifely Explain What Each Core Operations in MSCCLang-DSL Denotes. [0.5 points]
- `chunk`: References contiguous chunks ina specific GPU buffer (types as above) by rank, buffer name and index
- `copy`: Represents an operation to transfer chunks between different buffers or GPUs.
- `reduce`: Represents an in place reduction (eg : sum) between two chunk references, overwriting first operand.

### What is RecvReduceCopy Operation? [0.5 points]
It is a fused GPU instruction that combines 3 operations in a single atomic step. 1. Recieve a data chunk from remote GPU 2. Reduce it with a local chunk using a predefined operation 3. Copy the result to a specified destination buffer

### Which Collective Algorithm the First Code Snippet Captures? Why Do you think So? [0.5 points]
- Algorithm: Direct All-to-All All-Gather
- Reason: The code implements a direct all to all pattern for All-Gather by iterating through all possible source destination NPU pairs. For each pair (except when source equals destination), it fetches a chunk from the source NPU's input buffer and copies it directly to the destination NPU's output buffer at an idx corresponding to the src's rank. Ensures that each NPU recieves data from every other NPU and places it in the correct position in output buffer.

### Which Collective Algorithm the Second Code Snippet Captures? Why Do you think So? [0.5 points]
- Algorithm: Ring All Gather 
- Reason: The code implements a Ring All-Gather algorithm by creating a virtual ring topology where each NPU's data travels around the complete circle. For each GPU, it grabs that GPU's input chunk and then systematically copies it to every other GPU in sequence, following a ring pattern. The modulo operation (npu + 1) % npus_count creates the circular topology, ensuring data travels to each NPU exactly once before stopping when it reaches the original sender (while next != npu). This matches the definition of the Ring algorithm described in the MSCCLang paper.
