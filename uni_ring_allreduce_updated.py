import argparse

from msccl.language import *
from msccl.topologies import *
from msccl.collectives import *
from msccl.language.collectives import AllReduce


def uni_ring_allreduce_updated(npus_count: int, chunks_per_npu: int) -> None:
    topology = fully_connected(npus_count)
    
    # Note: now each NPU starts with:
    # C * N (=chunks_per_npu * npus_count) number of chunks
    # and ends with C * N number of chunks.
    collective = AllReduce(num_ranks=npus_count, chunk_factor=chunks_per_npu * npus_count, inplace=True)

    with MSCCLProgram("uni_ring_allreduce_updated", topology, collective, 1):
        ### ===============================================
        # Lab 5.3.1
        # TODO: Implement unidirectional Ring All-Reduce algorithm
        for npu in range(npus_count):
            for local_idx in range(chunks_per_npu):
      
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

        # Hint: each NPU should start the reduction of C chunks, not just 1.
        # Hint: modify the original uni_ring_allreduce.py code
        ### ===============================================
            
        Check()
        XML()


def main():
    # parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--npus_count', type=int, help ='number of NPUs')
    parser.add_argument('--chunks_per_npu', type=int, help ='initial number of chunks per NPU')
    args = parser.parse_args()

    # run MSCCLang-DSL to generate MSCCL-IR
    uni_ring_allreduce_updated(args.npus_count, args.chunks_per_npu)


if __name__ == '__main__':
    main()
