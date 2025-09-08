# Exercise 6: A transform Pass

Now you have all the basics list operation. So you can begin to manipulate the IR in passes.

There are two types of passes: Transformations that operate at a giving dialect, and Conversion/lowering passes that convert from a dialect to anothers.

For this exercise, you will write a (dummy) transformation that removes `list.foo` ops only if it contains an atttribute named `useless`. In this case, you have to replace the result by the input. 

**Clues**:
- Documentation on class [Operation](https://mlir.llvm.org/doxygen/classmlir_1_1Operation.html))

*Note: [to Understand the walker used in the pass](https://mlir.llvm.org/docs/Tutorials/UnderstandingTheIRStructure/)*

*Note: Don't forget to remove the TODO and REMOVE_ME ;)*
