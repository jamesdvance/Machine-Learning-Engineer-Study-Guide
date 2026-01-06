# Auto-Scaling for ML Training

## Summary

Auto-scaling dynamically adjusts compute resources based on workload demands, optimizing cost and utilization. For ML training, this includes scaling training clusters, managing GPU allocation, and handling job queues efficiently. Proper auto-scaling reduces idle resources while ensuring capacity when needed.

Key points to remember:

- Scale based on job queue depth and utilization
- Set minimum instances for quick start
- Use warm pools to reduce cold start time
- Consider spot instance availability
- Implement graceful shutdown for training jobs
- Monitor and alert on scaling events
- Balance responsiveness vs stability
- Different strategies for training vs inference

## Scaling Strategies

### Queue-Based Scaling

```python
class QueueBasedScaler:
    def __init__(
        self,
        min_instances=0,
        max_instances=10,
        scale_up_threshold=5,  # Queue depth
        scale_down_threshold=1,
        cooldown_seconds=300
    ):
        self.min_instances = min_instances
        self.max_instances = max_instances
        self.scale_up_threshold = scale_up_threshold
        self.scale_down_threshold = scale_down_threshold
        self.cooldown_seconds = cooldown_seconds
        self.last_scale_time = 0
        self.current_instances = min_instances

    def get_desired_instances(self, queue_depth, current_running):
        """Determine desired instance count."""
        import time

        # Cooldown check
        if time.time() - self.last_scale_time < self.cooldown_seconds:
            return self.current_instances

        # Calculate desired
        if queue_depth > self.scale_up_threshold:
            desired = min(self.max_instances, self.current_instances + 1)
        elif queue_depth < self.scale_down_threshold and current_running == 0:
            desired = max(self.min_instances, self.current_instances - 1)
        else:
            desired = self.current_instances

        if desired != self.current_instances:
            self.last_scale_time = time.time()
            self.current_instances = desired

        return desired
```

### Utilization-Based Scaling

```python
class UtilizationScaler:
    def __init__(
        self,
        target_utilization=0.7,
        min_instances=1,
        max_instances=10
    ):
        self.target_utilization = target_utilization
        self.min_instances = min_instances
        self.max_instances = max_instances

    def get_desired_instances(self, current_instances, current_utilization):
        """Scale based on GPU utilization."""
        if current_utilization > 0.9:
            # Scale up
            desired = min(self.max_instances, current_instances + 2)
        elif current_utilization > self.target_utilization:
            # Scale up slowly
            desired = min(self.max_instances, current_instances + 1)
        elif current_utilization < 0.3:
            # Scale down
            desired = max(self.min_instances, current_instances - 1)
        else:
            desired = current_instances

        return desired
```

## Cloud Provider Auto-Scaling

### AWS SageMaker

```python
import boto3

client = boto3.client('sagemaker')

# Create auto-scaling policy
client.register_scalable_target(
    ServiceNamespace='sagemaker',
    ResourceId='endpoint/my-endpoint/variant/AllTraffic',
    ScalableDimension='sagemaker:variant:DesiredInstanceCount',
    MinCapacity=1,
    MaxCapacity=10
)

# Target tracking policy
client.put_scaling_policy(
    PolicyName='GPUUtilization',
    ServiceNamespace='sagemaker',
    ResourceId='endpoint/my-endpoint/variant/AllTraffic',
    ScalableDimension='sagemaker:variant:DesiredInstanceCount',
    PolicyType='TargetTrackingScaling',
    TargetTrackingScalingPolicyConfiguration={
        'TargetValue': 70.0,
        'PredefinedMetricSpecification': {
            'PredefinedMetricType': 'SageMakerVariantInvocationsPerInstance'
        },
        'ScaleInCooldown': 300,
        'ScaleOutCooldown': 60
    }
)
```

### GCP Vertex AI

```yaml
# Vertex AI custom job with auto-scaling
apiVersion: vertex.googleapis.com/v1
kind: CustomJob
spec:
  displayName: "training-job"
  jobSpec:
    workerPoolSpecs:
      - machineSpec:
          machineType: n1-standard-8
          acceleratorType: NVIDIA_TESLA_T4
          acceleratorCount: 1
        replicaCount: 1
        containerSpec:
          imageUri: gcr.io/project/image
    scheduling:
      timeout: 86400s
```

### Kubernetes HPA

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: training-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: training-workers
  minReplicas: 1
  maxReplicas: 10
  metrics:
    - type: External
      external:
        metric:
          name: job_queue_depth
        target:
          type: AverageValue
          averageValue: 5
    - type: Resource
      resource:
        name: nvidia.com/gpu
        target:
          type: Utilization
          averageUtilization: 70
```

## Training-Specific Considerations

### Graceful Shutdown

```python
import signal
import torch

class GracefulShutdown:
    def __init__(self, model, optimizer, checkpoint_path):
        self.model = model
        self.optimizer = optimizer
        self.checkpoint_path = checkpoint_path
        self.shutdown_requested = False

        signal.signal(signal.SIGTERM, self.handle_signal)
        signal.signal(signal.SIGINT, self.handle_signal)

    def handle_signal(self, signum, frame):
        print("Shutdown requested, saving checkpoint...")
        self.shutdown_requested = True

    def should_stop(self):
        return self.shutdown_requested

    def save_checkpoint(self, epoch, step):
        torch.save({
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epoch': epoch,
            'step': step,
        }, self.checkpoint_path)

# Usage in training loop
shutdown_handler = GracefulShutdown(model, optimizer, 'checkpoint.pt')

for epoch in range(num_epochs):
    for step, batch in enumerate(dataloader):
        if shutdown_handler.should_stop():
            shutdown_handler.save_checkpoint(epoch, step)
            exit(0)

        train_step(batch)
```

### Elastic Training

```python
from torch.distributed.elastic.multiprocessing.errors import record

@record
def main():
    """Training function that supports elastic scaling."""
    import torch.distributed as dist

    # Initialize distributed (handles node changes)
    dist.init_process_group(backend='nccl')

    world_size = dist.get_world_size()
    rank = dist.get_rank()

    # Adjust batch size based on world size
    per_gpu_batch = 32
    global_batch = per_gpu_batch * world_size

    # Training loop
    for epoch in range(num_epochs):
        # Resample data for new world size
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
        dataloader = DataLoader(dataset, sampler=sampler)

        train_epoch(model, dataloader)

# Launch with torchrun
# torchrun --nproc_per_node=8 --nnodes=1:4 --rdzv_backend=c10d train.py
```

## Warm Pools

### Pre-Warmed Instances

```python
class WarmPoolManager:
    def __init__(self, pool_size=2):
        self.pool_size = pool_size
        self.warm_instances = []

    def maintain_pool(self):
        """Keep pool of ready instances."""
        current_warm = len(self.warm_instances)
        if current_warm < self.pool_size:
            # Launch new instances
            for _ in range(self.pool_size - current_warm):
                instance = self.launch_instance()
                self.prepare_instance(instance)  # Install deps, load model
                self.warm_instances.append(instance)

    def get_instance(self):
        """Get a warm instance (fast) or launch new (slow)."""
        if self.warm_instances:
            return self.warm_instances.pop()
        else:
            instance = self.launch_instance()
            self.prepare_instance(instance)
            return instance

    def return_instance(self, instance):
        """Return instance to pool."""
        if len(self.warm_instances) < self.pool_size:
            self.warm_instances.append(instance)
        else:
            self.terminate_instance(instance)
```

## Cost Optimization

### Spot Instance Integration

```python
class SpotAwareScaler:
    def __init__(self, on_demand_ratio=0.2):
        self.on_demand_ratio = on_demand_ratio  # Baseline

    def get_instance_mix(self, desired_instances):
        """Mix spot and on-demand for reliability."""
        on_demand = max(1, int(desired_instances * self.on_demand_ratio))
        spot = desired_instances - on_demand

        return {
            'on_demand': on_demand,
            'spot': spot
        }

    def handle_spot_interruption(self, interrupted_instance):
        """Handle spot instance termination."""
        # Save checkpoint immediately
        # Request replacement instance
        pass
```

### Scheduled Scaling

```python
class ScheduledScaler:
    def __init__(self):
        self.schedules = {
            # Work hours: higher capacity
            (9, 18): {'min': 5, 'max': 20},
            # Night: lower capacity
            (18, 9): {'min': 1, 'max': 5},
        }

    def get_limits(self):
        """Get scaling limits based on time."""
        import datetime
        hour = datetime.datetime.now().hour

        for (start, end), limits in self.schedules.items():
            if start <= end:
                if start <= hour < end:
                    return limits
            else:  # Wraps around midnight
                if hour >= start or hour < end:
                    return limits

        return {'min': 1, 'max': 10}  # Default
```

## Monitoring and Alerts

### Scaling Metrics

```python
def log_scaling_metrics(
    queue_depth,
    running_jobs,
    instance_count,
    gpu_utilization,
    cost_per_hour
):
    """Log metrics for scaling decisions."""
    import wandb

    wandb.log({
        'scaling/queue_depth': queue_depth,
        'scaling/running_jobs': running_jobs,
        'scaling/instance_count': instance_count,
        'scaling/gpu_utilization': gpu_utilization,
        'scaling/cost_per_hour': cost_per_hour,
        'scaling/efficiency': running_jobs / instance_count if instance_count > 0 else 0
    })
```

### Alerts

```python
def check_scaling_health(metrics):
    """Alert on scaling issues."""
    alerts = []

    # Queue growing too fast
    if metrics['queue_depth'] > 20:
        alerts.append("High queue depth - may need more capacity")

    # Low utilization
    if metrics['gpu_utilization'] < 0.3 and metrics['instance_count'] > 1:
        alerts.append("Low GPU utilization - consider scaling down")

    # Scaling failures
    if metrics['scaling_failures'] > 0:
        alerts.append("Scaling failures detected")

    return alerts
```

## Best Practices

1. **Set appropriate cooldowns**: Prevent thrashing
2. **Use minimum instances**: For quick job starts
3. **Implement graceful shutdown**: Save training state
4. **Monitor scaling events**: Track efficiency
5. **Use warm pools**: Reduce cold start time
6. **Mix spot and on-demand**: Balance cost and reliability
7. **Set maximum limits**: Prevent runaway costs
8. **Test scaling policies**: Verify behavior under load
