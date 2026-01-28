import argparse

from msccl.language import *
from msccl.topologies import *
from msccl.collectives import *
from msccl.language.collectives import AllGather
from typing import List, Tuple

def allgather_bi_ring_mesh(width: int, height: int) -> None:
    # number of NPUs: width * height
    # Assumption: width and height are even
    npus_count = width * height
    topology = fully_connected(npus_count)
    
    # Note: this is All-Gather, with 2 initial chunks per NPU.
    # i.e., for each NPU, chunk 0 will be processed in the original direction
    # ahd chunk 1 will be processed in the opposite direction
    collective = AllGather(num_ranks=npus_count, chunk_factor=2, inplace=True)

    ### ===============================================
    # Lab 5.4.2
    # TODO: You may copy-paste your Lab 5.4.1 implementation
    def coord_to_id(x: int, y: int) -> int:
        return y * width + x
    
    def get_ring() -> List[int]:
        ring = []
        x, y = 0, 0  
        direction = 1  
        while True:
            ring.append(coord_to_id(x, y))
            x += direction
            if x >= width or x <= 0: 
                x -= direction 
                y += 1  
                direction *= -1
                
                if y >= height: 
                    break
        y = height - 1
        while y > 0:
            ring.append(coord_to_id(0, y))  
            y -= 1   
        return ring
    ### ===============================================


    with MSCCLProgram("allgather_bi_ring_mesh", topology, collective, 1):
        # get ring
        ring = get_ring()
        
        ### ===============================================
        # Lab 5.4.2
        # TODO: Finish implementing the bidirectional Ring All-Gather algorithm.
        
        # Hint: Modify the implementation of your uni_ring_mesh.py
        # Hint: chunk 0 will be processed in the original direction
        # Hint: whereas chunk 1 will be processed in the opposite direction
        ### ===============================================

        for current_index in range(npus_count):
            current_rank = ring[current_index]
            
  
            chunk0 = chunk(rank=current_rank, buffer=Buffer.input, index=0)
            for hop in range(npus_count - 1):
                target_index = (current_index + 1 + hop) % npus_count
                target_rank = ring[target_index]
                chunk0 = chunk0.copy(dst=target_rank, buffer=Buffer.output, index=current_rank * 2)
            
            chunk1 = chunk(rank=current_rank, buffer=Buffer.input, index=1)
            reverse_index = current_index
            for hop in range(npus_count - 1):
                target_index = (reverse_index - 1 - hop) % npus_count
                target_rank = ring[target_index]
                chunk1 = chunk1.copy(dst=target_rank, buffer=Buffer.output, index=current_rank * 2 + 1)                   
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
        
    allgather_bi_ring_mesh(args.width, args.height)


if __name__ == '__main__':
    main()
