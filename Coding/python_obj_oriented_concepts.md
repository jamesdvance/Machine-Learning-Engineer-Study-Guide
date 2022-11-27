# Object Oriented In Python

## OOP Fundemental Principles
### Encapsulation
* bundle the data and methods that work on one unit
* Allows one to hide specific attributes/methods and/or control access
* In Java, Getter/Setter methods used
* In Python, `_` prefix makes an attribute private by convention
* In Python, `__` provides name manging
	* `__attribute` will be inaccessible outside class
	* same attribute will be accessible instead at `_class__attribute`

### Polymorphism
* 

## Design Patterns
### Singleton
* Is a single class which is responsible for creating an object that makes sure only a single object gets created
* Use cases - DB connection object shared by multiple objects
### Builder
* Is responsible for creating another type of object

## Encapsulate Private Names In a Class
* Both _ and __ can be used to encapsulate private names in a class in Python. 
* With double underscores, attributes cannot be overridden with inheritance
* To avoid defining a variable that clashes with a reserved word (like next or lambda) use a trailing underscore, aka next_

## Built-In Base Classes 
### Enum
* Enums have names and values associated with them
* from enum import Enum
* Can be displayed as type string or repr (string representation of object)
* Often hold constants
* Members can be accessed in two ways - by value and by name
* Are iterable - can be used in loops. (Iterate over class)
* Support hashing.Hashing can be used for getting and setting method
* Supports "==" and "is" equalities

### Abstract Base Class (ABC)
* Lets one avoid hasattr and isinstance methods to check whether an input conforms to a certain specification
* Also prevents instantiating subclass *without* over-riding a particular method in the superclass
* from abc import ABCMeta
* provides a method called register where any abstract base class can become an ancestor of any abritrary concrete class
* subclass with (metaclass=abc.ABCMeta)
* To register a given class as a subclass of an AbstractClass, can use the <AbstractClass>.register method or @<AbstractClass>.register decorator
* To perform automatic registraction of every subclass (instead of the manual methods above), you can use __subclasshook__, a magic method. 
* def __subclasshook__ must be defined with the @classmethod decorator
* @abc.abstractmethod prevents any attempt to instantiate a given class that doesn't override a particular method in the superclass
* Abstract classes cannot be instantiated. Must be inherited

## Decorators
### @staticmethod
* A static method doesn't receive any reference argument whether it is called by the class itself or an instance of the clss
* No self reference
* Can't access class attributes or instance attributes
* Can be called with ClassName.method_name() or object.method_name()

### @classmethod
* Can be used to declare a factor method that returns objects of a class
* Can access class attributes, but not instance attributes
* Can be called with ClassName.method_name() or object.method_name()
* First argument to method is `cls` not `self`

### @dataclass


## Build In Methods
### __repr__
* Used to represent a class's objects as a string
* Conventionally over-ridden to print ClassName(obj1, obj2) as a string
