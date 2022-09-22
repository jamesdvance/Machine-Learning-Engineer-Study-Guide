# API Gateways

## [API Gateway Microservice Architecture Pattern] https://microservices.io/patterns/apigateway.html

### Motivations
* Different clients need different data
* Client doesn't need to deal with the granularity of many different API's
* Network perfromance is different for different clients
* Partitioning of services can change over time and should be hidden from clients
* Services might use many protocols which may not be web friendly 

### Solution
* An API gateway is a single entry point for all clients
* It handles requests by either
	1. Routing requests to the appropriate service
	2. Fanning out requests to multiple services
* API gateway can perform other functinos like implement security

### Variation
* The 'Backend For Frontends' microservice pattern defines a seperate API gateway for each kind of client

### Benefits
* Insulates clients from how an application is partitioned into microservices
* Insulates clients from determining locations of service instances
* Reduces number of requests by enabling clients to retrieve data from multiple services in a single round-trip
* Translates from a standard public API protocol to whatever protocols are used internally

### Drawbacks
* Adds complexity
* Adds an additional network hop through the API gateway

### Load Balancing
* API Gateway is not a load balancer. Its primary function is to route requests to different services
***

[Amazon API Gateway](https://aws.amazon.com/api-gateway/)