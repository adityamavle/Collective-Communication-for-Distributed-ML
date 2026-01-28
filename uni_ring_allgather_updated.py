import argparse

from msccl.language import *
from msccl.topologies import *
from msccl.collectives import *
from msccl.language.collectives import AllGather


def uni_ring_allgather_updated(npus_count: int, chunks_per_npu: int) -> None:
    topology = fully_connected(npus_count)
    
    # Note: now each NPU starts with:
    # C (=chunks_per_npu) number of chunks
    # and ends with C * N (=chunks_per_npu * npus_count) number of chunks
    collective = AllGather(num_ranks=npus_count, chunk_factor=chunks_per_npu, inplace=True)

    with MSCCLProgram("uni_ring_allgather_updated", topology, collective, 1):
        ### ===============================================
        # Lab 5.2.1
        # TODO: Implement unidirectional Ring All-Gather algorithm
        # TODO: Each NPU starts with C (=chunks_per_npu) number of chunks, not 1
        for npu in range(npus_count):
            # Iterate over each local chunk on this NPU
            for local_idx in range(chunks_per_npu):
                # Handle for the initial chunk in Buffer.input
                c = chunk(
                    rank=npu,
                    buffer=Buffer.input,
                    index=local_idx
                )
                next_npu = (npu + 1) % npus_count
                for _ in range(npus_count - 1):
                    global_idx = npu * chunks_per_npu + local_idx

                    c = c.copy(
                        dst=next_npu,
                        buffer=Buffer.output,
                        index=global_idx
                    )
                    #Advance to the next hop
                    next_npu = (next_npu + 1) % npus_count
        # Hint: modify the original uni_ring_allgather.py code
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
    uni_ring_allgather_updated(args.npus_count, args.chunks_per_npu)


if __name__ == '__main__':
    main()
