# Real-Time Payment Fraud Detector

## Prompt
Design a real-time fraud detection system for a payment processor handling 10,000 transactions per second, with sub-100ms latency requirements and the ability to adapt to new fraud patterns.

## Design

### Requirements

#### Clarifying Questions
* Do we expect to to need to support multiple types of fraud models?
* Do we need to include an experimentation and training platform for data scientists?
* Will the data be available from the client or do we need to setup a separate real-time data pipeline? 

#### Problem Framing
* Fraud is a rare event - expect < 1 % of transactions 
* Binary classification
* Threshold-dependent - we can experiment with different probability thresholds and will want to ensure probabilities are stored

#### Functional
1. Label payments as 'fraud' or 'not fraud' 
2. Experiment with new models
3. Capture ground truth (fraud ) reporting data pipeline

#### Non-functional
1. Response latency: < 100ms
2. Throughput: 10,000 transactions / second

#### High-level
Based on the throughput and streaming requirements, the two potential inference patterns are a streaming pipeline or a rest api. If the business requires the main payment workflow to stop a transaction, then the rest-api will make sense. If the the business simply wants to immediately flag potential fraud cases in a dataset then the system can operate as an inference streaming processing where the system merely sends a stream out to a downstream processor. 

We'll assume the system wants to immediately stop the payment and thus design a rest api. 

A high-level diagram might look like:


