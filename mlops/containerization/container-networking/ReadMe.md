# Container Networking

## Summary

Container networking enables communication between containers, hosts, and external networks. For ML systems, networking configuration affects model serving latency, distributed training performance, and service discovery. Understanding container networking helps optimize inference endpoints and troubleshoot connectivity issues.

Key points to remember:

- Containers use Linux network namespaces for isolation
- Bridge networks provide container-to-container communication on a single host
- Overlay networks enable multi-host container communication
- Host networking eliminates network overhead for performance-critical workloads
- CNI plugins provide networking in Kubernetes environments

## Network Fundamentals

### Network Namespaces

Containers use Linux network namespaces for isolation:

```bash
# Each container gets its own network namespace with:
# - Network interfaces
# - IP addresses
# - Routing tables
# - Firewall rules

# View container's network namespace
docker inspect --format '{{.NetworkSettings.SandboxKey}}' mycontainer
# /var/run/docker/netns/abc123

# Execute in container's network namespace
nsenter --net=/var/run/docker/netns/abc123 ip addr
```

### Virtual Ethernet Pairs

Containers connect to the host via veth pairs:

```
Container Network Namespace          Host Network Namespace
┌─────────────────────────┐         ┌─────────────────────────┐
│                         │         │                         │
│   eth0 ←───────────────────────────→ vethXYZ               │
│   172.17.0.2            │  veth   │         │              │
│                         │  pair   │         ↓              │
└─────────────────────────┘         │   docker0 (bridge)     │
                                    │   172.17.0.1           │
                                    │         │              │
                                    │         ↓              │
                                    │   eth0 (host)          │
                                    │   192.168.1.10         │
                                    └─────────────────────────┘
```

## Docker Network Types

### Bridge Network (Default)

Containers on same bridge can communicate:

```bash
# Create bridge network
docker network create ml-network

# Run containers on network
docker run -d --name model-server --network ml-network mymodel:v1
docker run -d --name feature-store --network ml-network redis:7

# Containers can reach each other by name
docker exec model-server ping feature-store
```

Bridge network configuration:

```bash
# Create with custom subnet
docker network create \
    --driver bridge \
    --subnet 172.20.0.0/16 \
    --gateway 172.20.0.1 \
    ml-network

# Inspect network
docker network inspect ml-network
```

### Host Network

Container shares host's network namespace:

```bash
# Use host networking
docker run --network host mymodel:v1
```

Benefits:
- No NAT overhead
- Best network performance
- Direct port access

Tradeoffs:
- No network isolation
- Port conflicts possible
- Container sees all host interfaces

Use for:
- Performance-critical inference
- Distributed training with NCCL
- Debugging network issues

### None Network

Container has no network connectivity:

```bash
docker run --network none mymodel:v1
```

Use for:
- Completely isolated processing
- Security-sensitive workloads
- Batch jobs not needing network

### Overlay Network

Multi-host container communication:

```bash
# Initialize swarm (required for overlay)
docker swarm init

# Create overlay network
docker network create \
    --driver overlay \
    --attachable \
    ml-overlay

# Run containers on different hosts
docker run -d --name model-1 --network ml-overlay mymodel:v1  # Host A
docker run -d --name model-2 --network ml-overlay mymodel:v1  # Host B

# Containers can communicate across hosts
```

Overlay uses VXLAN encapsulation:

```
Container A (Host 1)              Container B (Host 2)
     │                                  ↑
     ↓ Packet to 10.0.0.3              │
┌─────────────────┐            ┌─────────────────┐
│ Overlay Network │            │ Overlay Network │
│ VXLAN Tunnel    │───────────→│ VXLAN Tunnel    │
│ Encapsulation   │  Underlay  │ Decapsulation   │
└─────────────────┘  Network   └─────────────────┘
```

### Macvlan Network

Containers get MAC addresses on physical network:

```bash
docker network create \
    --driver macvlan \
    --subnet 192.168.1.0/24 \
    --gateway 192.168.1.1 \
    -o parent=eth0 \
    ml-macvlan
```

Use for:
- Direct physical network access
- VLAN integration
- Legacy network requirements

## Port Publishing

### Port Mapping

Expose container ports to host:

```bash
# Map container port 8080 to host port 80
docker run -p 80:8080 mymodel:v1

# Map to specific interface
docker run -p 127.0.0.1:80:8080 mymodel:v1

# Map to random host port
docker run -p 8080 mymodel:v1
docker port mycontainer  # Shows assigned port

# Map multiple ports
docker run -p 8080:8080 -p 9090:9090 mymodel:v1
```

### Port Ranges

```bash
# Map port range
docker run -p 8080-8090:8080-8090 mymodel:v1
```

### UDP Ports

```bash
# Explicitly specify UDP
docker run -p 5000:5000/udp mymodel:v1
```

## DNS and Service Discovery

### Docker DNS

Containers on user-defined networks get DNS resolution:

```bash
# Create network
docker network create ml-net

# Start containers
docker run -d --name redis --network ml-net redis:7
docker run -d --name model --network ml-net mymodel:v1

# DNS resolution works
docker exec model ping redis
# PING redis (172.18.0.2): 56 data bytes
```

### DNS Configuration

```bash
# Custom DNS servers
docker run --dns 8.8.8.8 --dns 8.8.4.4 mymodel:v1

# Custom search domains
docker run --dns-search example.com mymodel:v1
```

### Aliases

```bash
# Network aliases
docker run -d \
    --name model-v1 \
    --network ml-net \
    --network-alias model \
    mymodel:v1

docker run -d \
    --name model-v2 \
    --network ml-net \
    --network-alias model \
    mymodel:v2

# Both respond to "model" (round-robin)
```

## Container-to-Container Communication

### Same Host

```yaml
# docker-compose.yml
version: '3.8'
services:
  model:
    image: mymodel:v1
    networks:
      - ml-net
    depends_on:
      - feature-store
      - redis

  feature-store:
    image: feast-server:v1
    networks:
      - ml-net

  redis:
    image: redis:7
    networks:
      - ml-net

networks:
  ml-net:
    driver: bridge
```

### Cross-Host (Docker Swarm)

```yaml
# docker-compose.yml for swarm
version: '3.8'
services:
  model:
    image: mymodel:v1
    deploy:
      replicas: 3
    networks:
      - ml-overlay

  load-balancer:
    image: nginx:latest
    ports:
      - "80:80"
    networks:
      - ml-overlay

networks:
  ml-overlay:
    driver: overlay
```

## Network Performance

### Measuring Network Performance

```bash
# Install iperf in containers
docker run -d --name server --network ml-net networkstatic/iperf3 -s
docker run --rm --network ml-net networkstatic/iperf3 -c server

# Results show bandwidth and latency
```

### Network Overhead Comparison

| Network Mode | Latency Overhead | Throughput |
|--------------|------------------|------------|
| Host | None | Native |
| Bridge | ~50-100μs | ~95% native |
| Overlay | ~100-200μs | ~90% native |
| Macvlan | ~10-20μs | ~98% native |

### Optimizing for ML Workloads

For inference:
```bash
# Use host networking for lowest latency
docker run --network host mymodel:v1
```

For distributed training:
```bash
# Host networking for NCCL performance
docker run --network host \
    --gpus all \
    --ipc=host \
    training-image:v1
```

## Kubernetes Networking

### Pod Networking

Every pod gets a unique IP address:

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: model-server
spec:
  containers:
    - name: model
      image: mymodel:v1
      ports:
        - containerPort: 8080
```

```bash
# Get pod IP
kubectl get pod model-server -o wide
# NAME           READY   STATUS    IP           NODE
# model-server   1/1     Running   10.244.1.5   node-1
```

### CNI Plugins

Container Network Interface plugins provide Kubernetes networking:

| CNI Plugin | Features |
|------------|----------|
| Calico | Network policies, BGP routing |
| Cilium | eBPF-based, advanced policies |
| Flannel | Simple overlay networking |
| Weave | Mesh networking, encryption |
| AWS VPC CNI | Native AWS VPC networking |

### Network Policies

Control traffic between pods:

```yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: model-server-policy
spec:
  podSelector:
    matchLabels:
      app: model-server
  policyTypes:
    - Ingress
    - Egress
  ingress:
    - from:
        - podSelector:
            matchLabels:
              role: api-gateway
      ports:
        - protocol: TCP
          port: 8080
  egress:
    - to:
        - podSelector:
            matchLabels:
              app: feature-store
      ports:
        - protocol: TCP
          port: 6566
```

### Service Networking

Services provide stable endpoints:

```yaml
apiVersion: v1
kind: Service
metadata:
  name: model-service
spec:
  selector:
    app: model-server
  ports:
    - port: 80
      targetPort: 8080
  type: ClusterIP
```

Traffic flow:
```
Client → Service IP (10.96.0.10:80) → Pod IP (10.244.1.5:8080)
              ↓
         kube-proxy (iptables/IPVS)
```

## ML-Specific Networking

### Distributed Training Networks

NCCL requires specific network configuration:

```yaml
# Kubernetes pod for distributed training
apiVersion: v1
kind: Pod
metadata:
  name: trainer
spec:
  hostNetwork: true  # Best NCCL performance
  containers:
    - name: trainer
      image: training:v1
      env:
        - name: NCCL_DEBUG
          value: INFO
        - name: NCCL_IB_DISABLE
          value: "0"  # Enable InfiniBand if available
      resources:
        limits:
          nvidia.com/gpu: 8
```

### High-Bandwidth Networking

For large model training:

```bash
# Use RDMA/InfiniBand when available
docker run --network host \
    --device=/dev/infiniband \
    --ulimit memlock=-1 \
    training:v1
```

### Feature Store Connectivity

```yaml
# docker-compose.yml
services:
  model:
    image: mymodel:v1
    environment:
      - FEAST_REDIS_HOST=redis
      - FEAST_REDIS_PORT=6379
    networks:
      - ml-net
    depends_on:
      - redis

  redis:
    image: redis:7
    networks:
      - ml-net
    volumes:
      - redis-data:/data

networks:
  ml-net:
    driver: bridge
    ipam:
      config:
        - subnet: 172.20.0.0/16
```

### gRPC for Model Serving

```yaml
# Expose gRPC port
services:
  triton:
    image: nvcr.io/nvidia/tritonserver:24.01-py3
    ports:
      - "8000:8000"  # HTTP
      - "8001:8001"  # gRPC
      - "8002:8002"  # Metrics
    networks:
      - ml-net
```

## Debugging Network Issues

### Inspect Container Network

```bash
# View container network settings
docker inspect --format '{{json .NetworkSettings}}' mycontainer | jq .

# Check IP address
docker inspect --format '{{.NetworkSettings.IPAddress}}' mycontainer

# List networks container is connected to
docker inspect --format '{{json .NetworkSettings.Networks}}' mycontainer | jq .
```

### Test Connectivity

```bash
# Exec into container
docker exec -it mycontainer sh

# Test DNS resolution
nslookup feature-store

# Test connectivity
ping feature-store
curl http://feature-store:8080/health

# Check listening ports
netstat -tlnp
```

### Network Debugging Tools

```bash
# Run debugging container on same network
docker run -it --rm --network ml-net nicolaka/netshoot

# Inside netshoot:
dig feature-store
tcpdump -i any port 8080
iperf3 -c model-server
```

### Common Issues

DNS not resolving:
```bash
# Check if on user-defined network (not default bridge)
docker network inspect bridge  # No DNS on default bridge
docker network create my-net   # Create user-defined network
```

Port not accessible:
```bash
# Check port mapping
docker port mycontainer

# Check firewall
iptables -L -n | grep 8080

# Check if service is listening
docker exec mycontainer netstat -tlnp
```

## Best Practices

### Use User-Defined Networks

```bash
# Don't use default bridge
docker run mymodel:v1  # Bad: uses default bridge, no DNS

# Use user-defined network
docker network create ml-net
docker run --network ml-net mymodel:v1  # Good: has DNS
```

### Isolate Networks by Purpose

```yaml
networks:
  frontend:    # Public-facing services
  backend:     # Internal services
  monitoring:  # Metrics and logging

services:
  api:
    networks:
      - frontend
      - backend

  model:
    networks:
      - backend

  prometheus:
    networks:
      - monitoring
      - backend
```

### Use Host Networking for Performance

```bash
# When latency matters
docker run --network host mymodel:v1
```

### Implement Network Policies

```yaml
# Deny all, then allow specific traffic
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: default-deny
spec:
  podSelector: {}
  policyTypes:
    - Ingress
    - Egress
```

## Common Pitfalls

### Exposing Ports Unnecessarily

```yaml
# Bad: exposes to all interfaces
ports:
  - "8080:8080"

# Better: localhost only for development
ports:
  - "127.0.0.1:8080:8080"

# Best: don't expose, use internal network
# (remove ports section, access via service mesh)
```

### Ignoring Network Namespaces

```bash
# Process inside container can't see host ports
docker exec mycontainer curl localhost:9090  # Fails if 9090 is on host

# Must use host IP or host networking
docker exec mycontainer curl host.docker.internal:9090
```

### Overlay Network Overhead

```bash
# Don't use overlay for latency-sensitive local communication
# Use bridge for single-host, overlay only for multi-host
```

### MTU Mismatches

```bash
# Check MTU
docker exec mycontainer cat /sys/class/net/eth0/mtu

# Set MTU explicitly if needed
docker network create --opt com.docker.network.driver.mtu=1400 ml-net
```
