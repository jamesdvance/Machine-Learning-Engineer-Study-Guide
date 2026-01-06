# MPI (Message Passing Interface)

## Summary

MPI is the standard interface for parallel computing, widely used in high-performance computing (HPC) environments. While NCCL has become the preferred choice for GPU deep learning, MPI remains important for HPC integration, CPU-based training, and environments with existing MPI infrastructure. MPI provides a rich set of communication primitives with decades of optimization.

Key points to remember:

- Standard interface for parallel computing with multiple implementations
- Common implementations: OpenMPI, MPICH, Intel MPI
- Excellent for HPC cluster integration
- Lower latency than alternatives for small messages
- Rich feature set beyond basic collectives
- Used by Horovod as communication layer
- Requires separate installation and configuration
- Not as optimized for GPU as NCCL

## MPI Implementations

### OpenMPI

Most popular open-source implementation:

```bash
# Ubuntu/Debian
sudo apt-get install openmpi-bin libopenmpi-dev

# Conda
conda install -c conda-forge openmpi
```

### MPICH

Alternative open-source implementation:

```bash
# Ubuntu/Debian
sudo apt-get install mpich

# From source
wget https://www.mpich.org/static/downloads/4.1/mpich-4.1.tar.gz
tar -xzf mpich-4.1.tar.gz
cd mpich-4.1
./configure --prefix=/usr/local
make && sudo make install
```

### Intel MPI

Optimized for Intel hardware:

```bash
# Using oneAPI
source /opt/intel/oneapi/setvars.sh
```

## PyTorch MPI Backend

### Building PyTorch with MPI

```bash
# Ensure MPI is in PATH
which mpirun

# Build PyTorch with MPI support
USE_MPI=1 python setup.py develop
```

### Using MPI Backend

```python
import torch.distributed as dist

# Initialize with MPI backend
dist.init_process_group(backend="mpi")

rank = dist.get_rank()
world_size = dist.get_world_size()

# Collective operations
tensor = torch.randn(1000)
dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
```

### Launching

```bash
mpirun -np 4 python train.py
```

## MPI4Py

Python bindings for MPI:

```bash
pip install mpi4py
```

### Basic Usage

```python
from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Send/Receive
if rank == 0:
    data = np.array([1.0, 2.0, 3.0])
    comm.Send(data, dest=1, tag=11)
elif rank == 1:
    data = np.empty(3, dtype=np.float64)
    comm.Recv(data, source=0, tag=11)

# Broadcast
data = np.zeros(100)
if rank == 0:
    data = np.arange(100, dtype=np.float64)
comm.Bcast(data, root=0)

# All-reduce
sendbuf = np.array([rank], dtype=np.float64)
recvbuf = np.zeros(1, dtype=np.float64)
comm.Allreduce(sendbuf, recvbuf, op=MPI.SUM)
```

### With PyTorch

```python
from mpi4py import MPI
import torch
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

# Convert between PyTorch and numpy for MPI operations
def allreduce_tensor(tensor):
    # To numpy
    np_tensor = tensor.numpy()
    result = np.zeros_like(np_tensor)

    # MPI all-reduce
    comm.Allreduce(np_tensor, result, op=MPI.SUM)

    # Back to PyTorch
    return torch.from_numpy(result)
```

## Horovod with MPI

Horovod uses MPI for coordination:

```bash
# Launch with mpirun
mpirun -np 4 python train_horovod.py

# Or with horovodrun (wraps mpirun)
horovodrun -np 4 python train_horovod.py
```

### MPI Configuration for Horovod

```bash
# OpenMPI settings
mpirun -np 4 \
    --mca btl_tcp_if_include eth0 \
    --mca btl ^openib \
    python train.py
```

## Multi-Node Configuration

### Hostfile

```
# hostfile.txt
node1 slots=4
node2 slots=4
node3 slots=4
node4 slots=4
```

### Launch Command

```bash
mpirun -np 16 --hostfile hostfile.txt python train.py
```

### SSH Setup

MPI requires passwordless SSH:

```bash
# Generate key
ssh-keygen -t rsa -N ""

# Copy to nodes
ssh-copy-id user@node1
ssh-copy-id user@node2
```

## Slurm Integration

### Slurm Script

```bash
#!/bin/bash
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --time=24:00:00

# Load MPI module
module load openmpi

# Run with srun (Slurm's MPI launcher)
srun python train.py
```

### Environment Variables

Slurm sets:
- `SLURM_PROCID`: Global rank
- `SLURM_LOCALID`: Local rank
- `SLURM_NTASKS`: World size

## Performance Tuning

### OpenMPI Tuning

```bash
# Use specific network
mpirun --mca btl_tcp_if_include eth0

# Disable specific transport
mpirun --mca btl ^openib

# Enable CUDA-aware MPI
mpirun --mca mpi_cuda_support 1

# Binding
mpirun --bind-to core --map-by socket
```

### Process Binding

```bash
# Bind to cores
mpirun -np 4 --bind-to core python train.py

# Bind to sockets
mpirun -np 4 --bind-to socket python train.py

# Report binding
mpirun -np 4 --report-bindings python train.py
```

### CUDA-Aware MPI

For GPU tensors without host copy:

```bash
# Check if MPI is CUDA-aware
ompi_info --parsable --all | grep mpi_built_with_cuda_support

# Enable CUDA-aware operations
export OMPI_MCA_mpi_cuda_support=1
```

## Collective Operations

### Standard Collectives

```python
from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD

# Broadcast
data = np.zeros(100) if rank != 0 else np.arange(100)
comm.Bcast(data, root=0)

# Scatter
sendbuf = np.arange(size * 10) if rank == 0 else None
recvbuf = np.zeros(10)
comm.Scatter(sendbuf, recvbuf, root=0)

# Gather
sendbuf = np.array([rank] * 10)
recvbuf = np.zeros(size * 10) if rank == 0 else None
comm.Gather(sendbuf, recvbuf, root=0)

# Allgather
sendbuf = np.array([rank])
recvbuf = np.zeros(size)
comm.Allgather(sendbuf, recvbuf)

# Reduce
sendbuf = np.array([rank])
recvbuf = np.zeros(1) if rank == 0 else None
comm.Reduce(sendbuf, recvbuf, op=MPI.SUM, root=0)

# Allreduce
sendbuf = np.array([rank])
recvbuf = np.zeros(1)
comm.Allreduce(sendbuf, recvbuf, op=MPI.SUM)
```

### Non-blocking Operations

```python
# Non-blocking send/receive
if rank == 0:
    req = comm.Isend(data, dest=1)
    # Do other work
    req.Wait()
elif rank == 1:
    req = comm.Irecv(data, source=0)
    req.Wait()

# Non-blocking collective
req = comm.Iallreduce(sendbuf, recvbuf, op=MPI.SUM)
# Do other work
req.Wait()
```

## Debugging

### MPI Debugging Tools

```bash
# Verbose output
mpirun -np 4 --mca mpi_show_mca_params all python train.py

# Debug specific component
mpirun -np 4 --mca btl_base_verbose 100 python train.py
```

### Common Issues

**Rank mismatch**:
```python
# Always check rank/size
print(f"MPI Rank: {comm.Get_rank()}, Size: {comm.Get_size()}")
```

**Deadlock**:
```python
# Ensure matching send/recv
if rank == 0:
    comm.Send(data, dest=1)
    comm.Recv(data2, source=1)  # Potential deadlock
elif rank == 1:
    comm.Send(data, dest=0)    # Both blocking on send
    comm.Recv(data2, source=0)

# Fix with non-blocking
if rank == 0:
    req = comm.Isend(data, dest=1)
    comm.Recv(data2, source=1)
    req.Wait()
```

## Comparison with NCCL/Gloo

| Aspect | MPI | NCCL | Gloo |
|--------|-----|------|------|
| GPU optimization | Limited | Excellent | Moderate |
| CPU support | Excellent | No | Good |
| HPC integration | Excellent | Limited | Limited |
| Latency (small msg) | Best | Good | Moderate |
| Setup complexity | High | Low | Low |
| Feature richness | Highest | Focused | Moderate |

### When to Use MPI

1. HPC clusters with existing MPI infrastructure
2. CPU-intensive distributed workloads
3. Need for advanced MPI features
4. Integration with other MPI-based code
5. Low-latency small message communication

### When to Use Alternatives

1. GPU deep learning (use NCCL)
2. Simple distributed PyTorch (use built-in DDP)
3. No MPI installation available (use Gloo)
