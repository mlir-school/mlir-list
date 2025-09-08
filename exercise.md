# Exercise 5: Add `list.length` with a canonicalization pattern

Let's take this example:
```mlir
%l = list.range %c0 to %c10 : !list.list<i32>
%length = list.length %l : !list.list<i32> -> i32
```
Maybe there is a way to simplify this structure, a length of a range, to something like this:
```mlir
%length = arith.subi %c10, %c0 : i32
```

To do this in MLIR, you can implement a canonicalization pattern on the `list.length` operation, in C++. The `--canonicalize` pass is the responsible to call all the patterns.

Follow the error and fix the canonicalization pattern for `list.length`.

**Clues**:
- Documentation on class [Value](https://mlir.llvm.org/doxygen/classmlir_1_1Value.html)
- Documentation on class [Operation](https://mlir.llvm.org/doxygen/classmlir_1_1Operation.html))
- llvm [`dyn_cast`](https://llvm.org/docs/ProgrammersManual.html#the-isa-cast-and-dyn-cast-templates)

*Note: Don't forget to remove the TODO and REMOVE_ME ;)*
