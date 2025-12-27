# [Understanding Data Storage and Ingestion For Large-Scale Deep Recommendation Model Training](https://arxiv.org/pdf/2108.09373.pdf)

## Setup and Scale
* Exabyte-scale data throughput and storage
* 10's of terrabytes QPS

## DLRM Training
* Meta's DLRM have > 12 *trillion* params 
* Meta uses hundreds of distinct recommendation models in production
* Replicate and shard the model and across multiple ZionEx nodes
* Mini-batches are distributed to different nodes
* Embeddings, activations, and gradients are synchronized across nodes over the backend network
* Synchronization is done once an normalized entropy threshold is reached (TODO- learn more)
* Each training job requires a DSI pipeline to generate samples, store in datasets and preprocess into tensors
* Training set is PB szed and stored as structured samples in distributed file systems. New features are constantly added
	* Training jobs read one epoch with feature-wise and row-wise filtering

## ZionEx hardware
* Meta's custom hardware platform
* Each node contains:
	* 8 NVIDIA A100 GPU's, each with a 200 Gbps Remote Direct Memory Access NIC for inter-node communicatoin
	* 4 CPU's with a 100Gpbs NIC connected to datacenter for data ingestion

## Data Storage and Training Pipeline
* Scribe - Meta's streaming platform which can stream PB's per hour
* LogDevice - Meta's append-only distributed store built on RocksDB
	* RocksDB - key-value store optimized for low-latency storage on high-speed disk drives
* ML engineer's run queries against Spark or Presto for feature engineering
	* Presto - SQL query engine for scale-efficient fast queries against Data Lakehouse
* Jobs are launched with FBLearner Flow
	* DPP Control Plane - DPP Master node breaks down preprocessing workload into self-contained work items called splits
		* serves splits to DPP workers upon requests and tracks their progress
		* Responsible for fault tolerance and auto-scaling
		* Is itself replicated to avoid SPOF
	* DPP Data Plan - DPP Workers effortlessly scale out to avoid data stalls
		* Are stateless. Communicate only with the DPP Master for instructions and a limited number of DPP Clients (serving them tensors)
		* All transformations within a mini-batch are performed locally
		* Recieves a set of transformation instructions from the Master, represented by a serialized and pre-compiled Pytorch module
		* Workers continuously query for new splits from the DPP master when finieshed
		* Each split requires workers to Extract, Transform and Load training data. 
		* Transformations are applied using high-performance (pre-compiled Pytorch) C++ binaries 
		* Once transformed, samples are batched together into tensors to be loaded into GPU trainers
	* Trainers- DPP clients run on the training nodes (Trainers). They expose a hook that the Pytorch Runtime can call to obtain preprocessed trainers
		* A simple RPC request returns a batch of tensors from the Worker buffer
		* Offloading the expensive ETL opp's to DPP Workers, along with Client multithreading, means DPP Clients do not bottleneck the data ingestion pipeline
		* Each client uses partitioned round robin routing, capping the # of connectiosn Clients and Workers need to maintain

## Coordinated Training At Scale
* ML Engineers start with small (<5%) subsets of data to run expirments
* The most promising of those get put into a combo job of 10s to 100s of training jobs
* The best of these become Release Candidates (RC's) which are then trained on the freshest data
* The most accurate of that retrain is deployed to production
* Engineers are encouraged by this process to batch many *different* architectures, temporal locality and features into combo jobs, which makes the process overall effiient
* Jobs can take 10 days to run, but many will be killed early due to low performance. Then an engineer will submit another job to maximize the combo job 

## Global Training Utilization
* Training resources (spanning datacenters) have to be calibrated to meet peak demand for combo jobs
* Each model reads a distinct dataset, so datacenter architects have to co-locate DSI resources with traininers themselves
* Each region contains a copy of all model's datasets
	* This storage is optimized with bin-packing techniques

## Feature Engineering
* Beta (exploratory) features are not logged, but can be backfilled or dynamical joined into each exploratory training job
* If a release candidate becomes the next production model, its features become active
* A feature review process is ongoing to deprecate less useful features
* Hundreds of new features are added and deprecated every month, which has implications for the data storage

##  Data Storage and Reading
* Benchmark datasets are re-read multiple times, so work is being done to cache benchmark data across epochs

## Production Training Jobs
* In production (as compared to offline) training, model size and compute capacity constrain the # of samples that can be used per epoch
* Each training job specifies a table and filters to select rows and columns
* Production jobs are trained for only a single epoch and are able to reach desired accuracy

## Online Preprocessing
* Data stalls occur when online preprocessing throughput is less than the throughput of the trainers themselves
	* results in underutilized GPU resources
* Online preprocessing throughput req expected to 3.5x in next two years
* Production-scale training is approaching NIC saturation. Bandwidth is a bottleneck, often resulting in data stalls
* Given expected advances in CPUs, memory bandwidth is expected to be a future bottleneck
* Preprocessing reduces data size, so extract tasks require ~ 2x bandwidth compared to loading processed into tensors
* Online preprocessing focuses on continuous throughput, not computing as fast as possible like other distributed processing
* Preprocessing should be right-sized to each model

## TorchArrow
* Many of the preprocessing operations are being open-sourced in TorchArrow
* DLRM transformations split into
	1. feature generation - Bucketize, NGram, MapId
	2. sparse feature normalization - SigridHash, FirstX
	3. dense feature normalization - Logic, boxcox, onehot
* Of these, feature generation is most expensive - often 75% of transformation cycle
* In the future these tasks will be done on GPI to speed up 
	* Will require specialized GPUs as current GPUs not optimized for multiple small, diverse patterns

## Hardware Bottlenecks 
* IOPS constrains storage by 8x
	* Therefore more storage is provisioned than needed to accomodate IOPS needs
* Use HDD for storage. Just using SSD's would provide inadequate storage relative to IOPS
* Better custom hardware needed to balance IO and storage
* Memory expected to be a larger bottleneck in the future

## Hetergeneous Hardware
* IOPS to storage incongruency offset by heterogenous storage media
* Complex architectures seek to balance IOPS and storage by taking features from both SSD and HDD
* SmartNICs and accelerators for microservers will help accelerate network IOPS

## Datacenters
* Accurate benchmark of of datasets and preprocessing reqs are crucial for setting up DC's correctly
* Future opportunities to intelligently route and bin-pack training jobs to specific regions to reduce storage duplications
* As datasets grow beyond DC-scale, model-hopper parallelism to be used to move TB-scale models across data centers (instead of PB-scale data)

## Benchmarks
* MLPerf is imperfect due to its old click log dataset
* Meta is working on more representative datasets in the future

## Multi-dimensional System Co-Design
* Small optimizations in the DSI systems translate to large payoffs at scale
* Optimizations should be thought off as end-to-end across hardware and software
* Feature flattening - the custom DWRF file format was optimize with *feature flattening*
	* organizes maps such that values for a given feature ID are stored as seperate streams on disk
	* Allows selective reading of features, found to double DPP worker throughput
* Coalesced reads - combines group-selected feature streams within 1 MiB in one I/O
	* Addresses small read sizes due to filtering at the storage layer (enabled by feature flattening)
	* small read sizes led to very inefficient and excessive disk seeks
* Feature reordering - Coelesced reads result in over-reads of unused features
	* To address, features are written to disk in ordered based on feature's popularity in training jobs w/in recent 7 day window
	* Leverages the insight that a small # of popular features contribute to a large % of storage throughput
* Large stripes - increases number of rows in each file stripe to increase average I/O size of each read
* In-memory flatmaps - way DPP workers can represent samples to match the DWRF (for disk) and tensor formats
* These optimizations considered ML application characteristics, like filtering on evolving feature sets, 

