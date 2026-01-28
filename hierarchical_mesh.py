import argparse

from msccl.language import *
from msccl.topologies import *
from msccl.collectives import *
from msccl.language.collectives import AllReduce
from typing import List

def hierarchical_mesh(width: int, height: int) -> None:   
    # number of NPUs: width * height
    # Assumption: width and height are even numbers
    npus_count = width * height
    topology = fully_connected(npus_count)
    
    # Note: this is All-Reduce.
    # Each NPU starts with: N (=npus_count) number of chunks, and ends with N number of chunks.
    collective = AllReduce(num_ranks=npus_count, chunk_factor=npus_count, inplace=True)
    
    ### ===============================================
    # Lab 5.5.2 and 5.5.3 Helper function
    # TODO: You may copy-paste your Lab 5.4.1 implementation
    def coord_to_id(x: int, y: int) -> int:
        return y * width + x
    ### ===============================================

    def uni_ring_all_reduce(npus: List[int]) -> None:
        ### ===============================================
        # Lab 5.5.1
        # TODO: Implement this function, which executes the unidirectional Ring All-Reduce algorithm
        # TODO: When the list of NPUs is given.
        # e.g., npus = [0, 1, 2, 3, 7, 6, 5, 4] runs the Ring All-Reduce algorithm among 8 NPUs.
        
        # Hint: Although the #NPUs of the above example is 8, the #chunks of the collective is still 16 (=N).
        # Hint: This means each NPU should process 16 / 8 = 2 chunks.
        
        # Hint: Modify your implementation of uni_ring_allreduce_updated.py
        # Hint: But note that the chunks to be processed per NPU is (#N / len(npus))
        # Hint: think of the chunk_id each NPU will process.
        
        # Hint: Also, the next npu is determined by npus[i + 1], not simply (next + 1).
        chunks_per_processor = npus_count // len(npus)
        for processor_idx in range(len(npus)):
            for local_chunk_idx in range(chunks_per_processor):
                chunk_id = processor_idx * chunks_per_processor + local_chunk_idx
                current_chunk = chunk(rank=npus[processor_idx], buffer=Buffer.input, index=chunk_id)
                
                next_processor_idx = (processor_idx + 1) % len(npus)
                for step in range(len(npus) - 1):
                    next_chunk = chunk(rank=npus[next_processor_idx], buffer=Buffer.input, index=chunk_id)
                    current_chunk = next_chunk.reduce(current_chunk)
                    next_processor_idx = (next_processor_idx + 1) % len(npus)
                for step in range(len(npus) - 1):
                    current_chunk = current_chunk.copy(dst=npus[next_processor_idx], buffer=Buffer.input, index=chunk_id)
                    next_processor_idx = (next_processor_idx + 1) % len(npus)
        ### ===============================================

    def phase1() -> None:
        ### ===============================================
        # Lab 5.5.2
        #: Implement phase 1 of the hierarchical mesh All-Reduce algorithm.
        
        # Hint: for 4x4 mesh, phase 1 will have two ring All-Reduce operations:
        #   1. Ring All-Reduce among: [0, 1, 2, 3, 7, 6, 5, 4]
        #   2. Ring All-Reduce among: [8, 9, 10, 11, 15, 14, 13, 12]
        
        # Hint: construct the npus list accordingly, and invoke the uni_ring_all_reduce function.
        for row_start in range(0, height, 2):
            ring_npus = []
            for row_offset in range(2):
                if row_offset % 2 == 0:
                    ring_npus.extend(coord_to_id(col, row_start+row_offset) for col in range(width))
                else:
                    ring_npus.extend(coord_to_id(col, row_start+row_offset) for col in range(width-1, -1, -1))
            uni_ring_all_reduce(ring_npus)
        ### ===============================================
    
    def phase2() -> None:
        ### ===============================================
        # Lab 5.5.3
        # TODO: Implement phase 2 of the hierarchical mesh All-Reduce algorithm.
        
        # Hint: like phase 2, think of the npus list and invoke the uni_ring_all_reduce function.
        # Hint: for example, for 4x4 mesh, the npus list will be:
        #      [0, 8], [4, 12], [1, 9], [5, 13], [2, 10], [6, 14], [3, 11], [7, 15]
        for column in range(width):
            even_row_npus = []
            odd_row_npus = []
            for row_start in range(0, height, 2):
                for row_offset in range(2):
                    if row_offset % 2 == 0:
                        even_row_npus.append(coord_to_id(column, row_start+row_offset))
                    else:
                        odd_row_npus.append(coord_to_id(column, row_start+row_offset))
            uni_ring_all_reduce(even_row_npus)
            uni_ring_all_reduce(odd_row_npus)
        ### ===============================================

    with MSCCLProgram("hierarchical_mesh", topology, collective, 1):
        # run phase 1
        phase1()
        
        # run phase 2
        phase2()
        
        Check()
        XML()

def main():
    # parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--width', type=int, help ='width')
    parser.add_argument('--height', type=int, help ='height')
    args = parser.parse_args()

    # run MSCCLang-DSL to generate MSCCL-IR
    assert args.width % 2 == 0, "width must be even"
    assert args.height % 2 == 0, "width must be even"
        
    hierarchical_mesh(args.width, args.height)


if __name__ == '__main__':
    main()
