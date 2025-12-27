# Object Oriented Programming

## Method Depth

## Solid Principles 
* https://towardsdatascience.com/solid-coding-in-python-1281392a6a94
*** 

### Single-Responsibility Principle
* Functions should have a single responsibility (i.e. be atomic)
* A seperate function or class (generically named 'main') can be used to call all each atomic function in a step by step process
* Results in
	* easier localization of errors
	* function reuseability
	* easier testing (also, tests should be written before main)

### Open-Closed Principle
* "Software entities...should be open to extension but closed for modification"
* To accomodate new functionality, should only need to add more code, not modify existing
* To accomplish, can use subclassing. E.g. instead of functions above, can just subclass a single Abstract Base Class 
* Can iterate over the Abstract Base Classes's __subclassess__ attribute, which will enumerate all subclasses

### Liskov Substitution Principle
* "Derived classes must be substitutable for their base classes"
* If a subclass redefines a function also present in the parent class, a client-user shouldn't notice a different between its behavior and that in the parent class
* Base classes should be reasonably abstract to allow for this

### Interface Segregation Principle
* A class should only have the interface needed, and avoid methods that aren't relevant to the  class
* This arises when a subclass inherits methods from a base class that it doesn't need
* Is followed by creating more client-specific interfaces and avoiding one general-purpose one
* Multiple inheritance can help make it so certain subclasses don't get unncessary attributes

### Dependency-Inversion Principle
* Abstractions should not depend on details. Details should depend on abstractions
* High level modules should not depend on low level modules. Both should depend on abstractions. 
* Abstractions should not depend on details. Details should depend on abstractions
* [This](https://stackoverflow.com/questions/61358683/dependency-inversion-in-python) S.O answer gives a good example in Python

***