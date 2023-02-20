# chap5 部分Lower 到低等级dialect

## Dialect Conversion

- A Conversion Target 转换目标

           判断dialect/operation 是否能合法转换

- A set of Rewrite Patterns 重写模式
转换合法的操作
- Optionally, Type Conversion 类型转换

### Conversion Target

```cpp
void ToyToAffineLoweringPass::runOnOperation() {
// The first thing to define is the conversion target. This will define the
// final target for this lowering.
  mlir::ConversionTarget target(getContext());

// We define the specific operations, or dialects, that are legal targets for
// this lowering. In our case, we are lowering to a combination of the
// `Affine`, `Arith`, `Func`, and `MemRef` dialects.
  target.addLegalDialect<AffineDialect, arith::ArithDialect,
                         func::FuncDialect, memref::MemRefDialect>();

// We also define the Toy dialect as Illegal so that the conversion will fail
// if any of these operations are *not* converted. Given that we actually want
// a partial lowering, we explicitly mark the Toy operations that don't want
// to lower, `toy.print`, as *legal*. `toy.print` will still need its operands
// to be updated though (as we convert from TensorType to MemRefType), so we
// only treat it as `legal` if its operands are legal.
  target.addIllegalDialect<ToyDialect>();
  target.addDynamicallyLegalOp<toy::PrintOp>([](toy::PrintOp op) {
return llvm::none_of(op->getOperandTypes(),
                         [](Type type) {return type.isa<TensorType>(); });
  });
  ...
}
```

将`ToyDialect`为非法，转换目标`AffineDialect, arith::ArithDialect,func::FuncDialect, memref::MemRefDialect`为合法，操作printOp为合法

### Conversion Patterns

以transposeOp为例

```cpp
/// Lower the `toy.transpose` operation to an affine loop nest.
structTransposeOpLowering :public mlir::ConversionPattern {
  TransposeOpLowering(mlir::MLIRContext *ctx)
      : mlir::ConversionPattern(TransposeOp::getOperationName(), 1, ctx) {}
///tensor<...>` 转换到 `memref<...>
/// Match and rewrite the given `toy.transpose` operation, with the given
/// operands that have been remapped from `tensor<...>` to `memref<...>`.
  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op, ArrayRef<mlir::Value> operands,
                  mlir::ConversionPatternRewriter &rewriter)constfinal {
auto loc = op->getLoc();

// Call to a helper function that will lower the current operation to a set
// of affine loops. We provide a functor that operates on the remapped
// operands, as well as the loop induction variables for the inner most
// loop body.
    lowerOpToLoops(
        op, operands, rewriter,
        [loc](mlir::PatternRewriter &rewriter,
              ArrayRef<mlir::Value> memRefOperands,
              ArrayRef<mlir::Value> loopIvs) {
// Generate an adaptor for the remapped operands of the TransposeOp.
// This allows for using the nice named accessors that are generated
// by the ODS. This adaptor is automatically provided by the ODS
// framework.
          TransposeOpAdaptor transposeAdaptor(memRefOperands);
          mlir::Value input = transposeAdaptor.input();

// Transpose the elements by generating a load from the reverse
// indices.
          SmallVector<mlir::Value, 2> reverseIvs(llvm::reverse(loopIvs));
return rewriter.create<mlir::AffineLoadOp>(loc, input, reverseIvs);
        });
return success();
  }
```

注册转换模式

```cpp
void ToyToAffineLoweringPass::runOnOperation() {
  ...

// Now that the conversion target has been defined, we just need to provide
// the set of patterns that will lower the Toy operations.
  mlir::RewritePatternSet patterns(&getContext());
  patterns.add<..., TransposeOpLowering>(&getContext());

  ...
```

### 局部下降

一旦定义了模式，我们就可以执行实际的下降。`DialectConversion`框架提供了几种不同的下调模式，但考虑到我们的目的，我们将执行部分下降，因为我们此时不会转换`toy.print`(Affine不能转换print）

```cpp
void ToyToAffineLoweringPass::runOnOperation() {
  ...

// With the target and rewrite patterns defined, we can now attempt the
// conversion. The conversion will signal failure if any of our *illegal*
// operations were not converted successfully.
if (mlir::failed(mlir::applyPartialConversion(getOperation(), target, patterns)))
    signalPassFailure();
}
```

更新`toy.print`的当前定义，以允许在降低的类型上操作。这种方法的好处是它很简单，不会引入额外的隐藏副本，也不需要另一个操作定义。此选项的缺点是，它需要混合`Toy`方言的抽象层。

为简单起见，我们将使用第三个选项来降低。这涉及更新操作定义文件中PrintOp的类型约束

```cpp
def PrintOp : Toy_Op<"print"> {
  ...

// The print operation takes an input tensor to print.
// We also allow a F64MemRef to enable interop during partial lowering.
let arguments = (ins AnyTypeOf<[F64Tensor, F64MemRef]>:$input);
}
```

![Untitled](chap5%20%E9%83%A8%E5%88%86Lower%20%E5%88%B0%E4%BD%8E%E7%AD%89%E7%BA%A7dialect%2058c9085d97d64bbb8c0ef624682b5d64/Untitled.png)

![Untitled](chap5%20%E9%83%A8%E5%88%86Lower%20%E5%88%B0%E4%BD%8E%E7%AD%89%E7%BA%A7dialect%2058c9085d97d64bbb8c0ef624682b5d64/Untitled%201.png)

![Untitled](chap5%20%E9%83%A8%E5%88%86Lower%20%E5%88%B0%E4%BD%8E%E7%AD%89%E7%BA%A7dialect%2058c9085d97d64bbb8c0ef624682b5d64/Untitled%202.png)