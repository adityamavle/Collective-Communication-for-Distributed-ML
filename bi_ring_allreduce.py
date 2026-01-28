import argparse

from msccl.language import *
from msccl.topologies import *
from msccl.collectives import *
from msccl.language.collectives import AllReduce


def bi_ring_allreduce(npus_count: int, chunks_per_npu: int) -> None:
    topology = fully_connected(npus_count)
    
    # Note: now each NPU starts with:
    # C * N (=chunks_per_npu * npus_count) number of chunks
    # and ends with C * N number of chunks.
    # Assumption: C is even, so that each ring can process C/2 number of chunks.
    collective = AllReduce(num_ranks=npus_count, chunk_factor=chunks_per_npu * npus_count, inplace=True)
    half = chunks_per_npu // 2
    with MSCCLProgram("bi_ring_allreduce", topology, collective, 1):
        ### ===============================================
        # Lab 5.3.2
        # TODO: Implement bidirectional Ring All-Reduce algorithm
        for npu in range(npus_count):
            for local_idx in range(half):
                chunk_id = npu * chunks_per_npu + local_idx
                c = chunk(rank=npu, buffer=Buffer.input, index=chunk_id)
                next_npu = (npu + 1) % npus_count
                for _ in range(npus_count - 1):
                    next_c = chunk(rank=next_npu, buffer=Buffer.input, index=chunk_id)
                    c = next_c.reduce(c)
                    next_npu = (next_npu + 1) % npus_count
                for _ in range(npus_count - 1):
                    c = c.copy(dst=next_npu, buffer=Buffer.input, index=chunk_id)
                    next_npu = (next_npu + 1) % npus_count

            for local_idx in range(half, chunks_per_npu):
                chunk_id = npu * chunks_per_npu + local_idx
                c = chunk(rank=npu, buffer=Buffer.input, index=chunk_id)
                next_npu = (npu - 1 + npus_count) % npus_count
                for _ in range(npus_count - 1):
                    next_c = chunk(rank=next_npu, buffer=Buffer.input, index=chunk_id)
                    c = next_c.reduce(c)
                    next_npu = (next_npu - 1 + npus_count) % npus_count
                for _ in range(npus_count - 1):
                    c = c.copy(dst=next_npu, buffer=Buffer.input, index=chunk_id)
                    next_npu = (next_npu - 1 + npus_count) % npus_count

        # Hint: modify your implementation of uni_ring_allreduce_updated.py
        
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
    
    assert args.chunks_per_npu % 2 == 0, "chunks_per_npu must be even"

    # run MSCCLang-DSL to generate MSCCL-IR
    bi_ring_allreduce(args.npus_count, args.chunks_per_npu)


if __name__ == '__main__':
    main()
