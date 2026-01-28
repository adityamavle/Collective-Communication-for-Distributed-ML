# Collective-Communication-for-Distributed-ML
Code for topology-aware collective communication algorithms (All-Gather, All-Reduce) implemented using MSCCLang-DSL, exploring ring, mesh, and hierarchical designs for distributed ML systems. Lab 5 for CS 8803: HW/SW Co-Design for ML Systems (Georgia Tech).

## 1. Lab Goal

Implement and validate **topology-aware collective communication algorithms** (All-Gather, All-Reduce) using **MSCCLang-DSL**, comparing ring, mesh-aware, and hierarchical designs for distributed ML systems. This work corresponds to **Lab 5 for CS 8803: HW/SW Co-Design for ML Systems (Georgia Tech)**.

---

## 2. Methodology

The lab uses **MSCCLang-DSL** to express collective communication at the chunk level using `chunk`, `copy`, and `reduce` primitives. Each implementation is compiled into **MSCCL-IR (XML)** and validated for correctness using `Check()`.

Implemented algorithms and files:
- `uni_ring_allgather_updated.py`: Unidirectional ring All-Gather with multiple chunks per NPU
- `bi_ring_allgather.py`: Bidirectional ring All-Gather
- `uni_ring_allreduce_updated.py`: Unidirectional ring All-Reduce with multiple chunks per NPU
- `bi_ring_allreduce.py`: Bidirectional ring All-Reduce
- `uni_ring_mesh.py`: Topology-aware ring All-Gather on a 2D mesh (Hamiltonian-cycle traversal)
- `bi_ring_mesh.py`: Bidirectional topology-aware mesh ring All-Gather
- `hierarchical_mesh.py`: Hierarchical All-Reduce on a 2D mesh (two-phase reduction)
- `discussion.md`: Written discussion answers for the lab

---

## 3. Experiments
### How to Run and Test

Each execution:
1. Generates an MSCCLang program
2. Runs `Check()` for correctness
3. Emits **MSCCL-IR (XML)** using `XML()`

Example commands:

Unidirectional Ring All-Gather:
```
python uni_ring_allgather_updated.py --npus_count N --chunks_per_npu C
```

Unidirectional Ring All-Reduce:
```
python uni_ring_allreduce_updated.py --npus_count N --chunks_per_npu C
```

Bidirectional Ring All-Gather / All-Reduce:
```
python bi_ring_allgather.py --npus_count N --chunks_per_npu C
python bi_ring_allreduce.py --npus_count N --chunks_per_npu C
```

Mesh-aware Ring All-Gather (even dimensions required):
```
python uni_ring_mesh.py --width W --height H
python bi_ring_mesh.py --width W --height H
```

Hierarchical Mesh All-Reduce (even dimensions required):
```
python hierarchical_mesh.py --width W --height H
```

A run is considered successful if the program completes without errors, `Check()` passes, and valid MSCCL-IR XML is produced.

---

## 4. Results and Discussion

All implementations generate valid MSCCL-IR and pass correctness checks.

Observations:
- Bidirectional ring variants reduce the critical path by utilizing parallel communication in both directions.
- Mesh-aware ring construction aligns communication with physical topology, avoiding inefficient logical mappings.
- Hierarchical All-Reduce decomposes global reduction into structured phases, illustrating how topology-aware designs can improve scalability.

Overall, the lab demonstrates how collective communication performance and structure depend on hardware topology and how MSCCLang-DSL enables clear, correct expression of these algorithms.
