import argparse

from msccl.language import *
from msccl.topologies import *
from msccl.collectives import *
from msccl.language.collectives import AllGather
from typing import List, Tuple

def allgather_uni_ring_mesh(width: int, height: int) -> None:
    # Number of NPUs: width * height
    npus_count = width * height
    topology = fully_connected(npus_count)
    
    # Note: this is All-Gather, with 1 chunk per NPU.
    collective = AllGather(num_ranks=npus_count, chunk_factor=1, inplace=True)
    
    def coord_to_id(x: int, y: int) -> int:
        ### ===============================================
        # Lab 5.4.1 helper function
        # TODO: Implement this helper function
        # TODO: which converts 2D coordinates to NPU id
        return y * width + x
        ### ===============================================
    
    def get_ring() -> List[int]:        
        ### ===============================================
        # Lab 5.4.1
        # TODO: Implement this helper function which returns the NPUs of the ring.
        
        # Hint: for a 2x2 mesh, the ring is
        #       [0, 1, 2, 3, 7, 6, 5, 9, 10, 11, 15, 14, 13, 12, 8, 4]
        
        # Hint: Start from 0
        # Hint: then, as necessary, repeat the following steps:
        #   go right until the end of the row
        #   go down to the next row
        #   go left until the end of the row
        # Hint: then, move up to finish the ring
        ### ===============================================
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
    with MSCCLProgram("allgather_uni_ring_mesh", topology, collective, 1):
        ring = get_ring()
        ### ===============================================
        # Lab 5.4.1
        # TODO: Finish implementing the All-Gather algorithm.
        
        # Hint: You may modify the implementation of uni_ring_allgather.sh,
        # Hint: but instead of simple next = next + 1, you'll to use the `ring`.
        ### ===============================================
        for current_index in range(npus_count):
            current_rank = ring[current_index]
            c = chunk(rank=current_rank, buffer=Buffer.input, index=0)
            
            for hop in range(npus_count - 1):
                target_index = (current_index + 1 + hop) % npus_count
                target_rank = ring[target_index]
                c = c.copy(dst=target_rank, buffer=Buffer.output, index=current_rank)
                                        
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
        
    allgather_uni_ring_mesh(args.width, args.height)


if __name__ == '__main__':
    main()
