# Key Value Stores

## [The Secret Sauce Behind NoSQL: The LSM Tree](https://www.youtube.com/watch?v=I6jB0nM9SKU&t=336s)

### Demand for NoSQL
Write-heavy, massive sized datasets have createad demand for NoSQL

### Relational DB's
* Structures data with a B-Tree which is optimized for reads
* Updates involve random I/O and sometimes multiple updates on disk

### NoSQL
* Structures data with LSM tree
* Writes are batched in memory in a structure called a Memtable 
* A memtable is ordered by object key, implemented with balanced binary tree
* Once the memtable's size limits are reached it sorts its data and flushes them to disk
* The data are written in sequential IO in an immutable structure called SSTable

### SSTable
* The SSTable is the most recent segment of the LSM tree
* SSTables are immutable and store key-value pairs in a sorted sequence
* Fast writes due to sequential IO
* Each SSTable represents a small chronological subset of the incoming changes
* An update to an existing object key, does not update an old SSTable, it just appends to the current SSTable
* Deleting an object requires adding a marker called a tombstone for the most recent SSTable for the object key of deleted item

### Read Requests
1. Look for the key in the memtable
2. Look in the most recent SSTable
3. Look in the next most recent SSTable, etc

### Compaction and Merging
* As number of SSTables grows, it takes longer to look up individual keys
* Additionally plenty of disk space is wasted on out of date entries where there is a newer value or tombstone
* To fix this, there is a periodic merging and compaction process running in background
* It merges SStables and discards deleted values
* Since sorted, can do this pretty efficiently. Algorithm similar to merge sort.

### Compaction Strategies
* Tiered - write optimized
* Level - read optimized
* SSTables in the LSM Tree are seperated into levels
* Merges result in the SSTables going to the level below

### Read Optimization
* Most NoSQL systems keep a summary table in memory with the min/max range of each disk block for faster searching
* Checking for a key that doens't exist would mean checking every single table at each level
	* A bloom filter is used for this to return a boolean if a key does not exist

***