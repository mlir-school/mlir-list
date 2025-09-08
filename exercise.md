# Exercise 7: A lowering Pass

*[many transformations later]*

You are now ready to lower your super optimized list IR into standard MLIR dialects, to in fine generate an executable.

A lowering pass can be seen as a Conversion from a dialect to one or many others. Conversion passes have some nice properties that you can exlore [here](https://mlir.llvm.org/docs/DialectConversion/). It is usually implemented with a special [`ConversionPattern`](https://mlir.llvm.org/docs/DialectConversion/#conversion-patterns). Unlike more general `RewritePattern`, it can handle [type conversion](https://mlir.llvm.org/docs/DialectConversion/#type-conversion).

Fix the conversion pass from list to SCF/Arith/Tensor dialects.

**Clues**:
- Documentation on class [ConversionPatternRewriter](https://mlir.llvm.org/doxygen/classmlir_1_1ConversionPatternRewriter.html). (Don't forget to also explore inherited methods)
- [`tensor.dim`](https://mlir.llvm.org/docs/Dialects/TensorOps/#tensordim-tensordimop)

*Note: The list conversion pass does not require a special [`TypeConverter`](https://mlir.llvm.org/docs/DialectConversion/#type-converter), because there is not remaining operation operating on `!list.list<>` once all conversion pattern have been applied.*

*Note: Don't forget to remove the TODO and REMOVE_ME ;)*
