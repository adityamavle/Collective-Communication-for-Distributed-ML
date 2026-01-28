import argparse

from msccl.language import *
from msccl.topologies import *
from msccl.collectives import *
from msccl.language.collectives import AllGather


def bi_ring_allgather(npus_count: int, chunks_per_npu: int) -> None:
    topology = fully_connected(npus_count)
    
    # Note: now each NPU starts with:
    # C (=chunks_per_npu) number of chunks
    # and ends with C * N (=chunks_per_npu * npus_count) number of chunks
    # Assumption: C is even, so that each ring can process C/2 number of chunks.
    collective = AllGather(num_ranks=npus_count, chunk_factor=chunks_per_npu, inplace=True)

    with MSCCLProgram("bi_ring_allgather", topology, collective, 1):
        ### ===============================================
        # Lab 5.2.2
        # TODO: Implement bidirectional Ring All-Gather algorithm
        # TODO: Each NPU starts with C (=chunks_per_npu) number of chunks
        for npu in range(npus_count):
            chunks_per_ring = chunks_per_npu // 2
            
            for chunk_idx in range(chunks_per_ring):
                c = chunk(rank=npu, buffer=Buffer.input, index=chunk_idx)
                output_offset = npu * chunks_per_npu + chunk_idx
                next = (npu + 1) % npus_count
                for step in range(npus_count - 1):
                    dst_idx = output_offset
                    c = c.copy(dst=next, buffer=Buffer.output, index=dst_idx)
                    next = (next + 1) % npus_count
            
            for chunk_idx in range(chunks_per_ring, chunks_per_npu):
                c = chunk(rank=npu, buffer=Buffer.input, index=chunk_idx)
                output_offset = npu * chunks_per_npu + chunk_idx
                next = (npu - 1) % npus_count
                for step in range(npus_count - 1):
                    dst_idx = output_offset
                    c = c.copy(dst=next, buffer=Buffer.output, index=dst_idx)
                    next = (next - 1) % npus_count
        
        # Hint: modify your implementation of uni_ring_allgather_updated.py
        # Hint: C/2 chunks follow the original Ring
        # Hint: and the other C/2 chunks follow the Ring in the opposite direction
        ### ===============================================
            
        Check()
        XML()


def main():
    # parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--npus_count', type=int, help ='number of NPUs')
    parser.add_argument('--chunks_per_npu', type=int, help ='initial number of chunks per NPU')
    args = parser.parse_args()
    
    # assure chunks_per_npu is even
    assert args.chunks_per_npu % 2 == 0, "chunks_per_npu must be even"

    # run MSCCLang-DSL to generate MSCCL-IR
    bi_ring_allgather(args.npus_count, args.chunks_per_npu)


if __name__ == '__main__':
    main()
