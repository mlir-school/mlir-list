# Exercise 4: A verifier for `list.map`

Huston, we have a problem. It seems that you can write a `list.map` that returns a `!list.list<i32>` but yield a `i1` for each element. You need to prevent this!

Operation definition allows you to add a custom verifier in C++. This may solve the issue.

Follow the errors to fix the verifier of the `list.map`.

**Clues**:
- Documentation on class [Value](https://mlir.llvm.org/doxygen/classmlir_1_1Value.html)
- Given the ODS Definition of Types and ops you did before, getters are automatically generated for params/arguments/results:
```cpp
// Example: Access $tensor with .getTensor() 
let parameters = (ins "MemRefType":$tensor);
// Example: Access $input with .getInput()
let arguments = (ins I32:$input);
```
- (not used normally: Documentation on class [Operation](https://mlir.llvm.org/doxygen/classmlir_1_1Operation.html))

*Note: Don't forget to remove the TODO and REMOVE_ME ;)*
