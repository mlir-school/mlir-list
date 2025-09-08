# Exercise 2: A type: `!list.list<>`

Before you create operations operating on lists, you will need to define a list type. Type definition can be done in `ODS`. Follow the error and fix the definition of the `ListType`.

**Clues**:

- ElementType is built by constraints: https://mlir.llvm.org/docs/DefiningDialects/Constraints/*
```td
// Example: Element type constraint for VectorType
def Builtin_VectorTypeElementType : AnyTypeOf<[AnyInteger, Index, AnyFloat]>;
```

- ListType is full type defined in ODS: (https://mlir.llvm.org/docs/DefiningDialects/AttributesAndTypes/#adding-a-new-attribute-or-type-definition)
```td
// Example: nvgpu.warpgroup.descriptor
def NVGPU_WarpgroupMatrixDescriptor : NVGPU_Type<"WarpgroupMatrixDescriptor", "warpgroup.descriptor", []> {
  let summary = "Warpgroup matrix descriptor type";
  let description = [{...}];
  let parameters = (ins "MemRefType":$tensor);
  let assemblyFormat = "`<` struct(params) `>`";
}
```

*Note: Don't forget to remove the TODO ;)*
