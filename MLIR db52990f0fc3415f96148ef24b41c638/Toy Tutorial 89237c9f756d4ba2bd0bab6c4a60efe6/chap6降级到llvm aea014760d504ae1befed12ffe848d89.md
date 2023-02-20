# chap6降级到llvm

除了printop外，std，affine，Arith等都有内置的convert

print 声明

```cpp
/// Return a symbol reference to the printf function, inserting it into the
/// module if necessary.
static FlatSymbolRefAttr getOrInsertPrintf(PatternRewriter &rewriter,
                                           ModuleOp module,
                                           LLVM::LLVMDialect *llvmDialect) {
auto *context = module.getContext();
if (module.lookupSymbol<LLVM::LLVMFuncOp>("printf"))
return SymbolRefAttr::get("printf", context);

// Create a function declaration for printf, the signature is:
//   * `i32 (i8*, ...)`
auto llvmI32Ty = IntegerType::get(context, 32);
auto llvmI8PtrTy =
      LLVM::LLVMPointerType::get(IntegerType::get(context, 8));
auto llvmFnType = LLVM::LLVMFunctionType::get(llvmI32Ty, llvmI8PtrTy,
/*isVarArg=*/true);

// Insert the printf function into the body of the parent module.
  PatternRewriter::InsertionGuard insertGuard(rewriter);
  rewriter.setInsertionPointToStart(module.getBody());
  rewriter.create<LLVM::LLVMFuncOp>(module.getLoc(), "printf", llvmFnType);
return SymbolRefAttr::get("printf", context);
}
```

### 转换模式

```cpp
  mlir::RewritePatternSet patterns(&getContext());
  mlir::populateAffineToStdConversionPatterns(patterns, &getContext());
  mlir::cf::populateSCFToControlFlowConversionPatterns(patterns, &getContext());
  mlir::arith::populateArithToLLVMConversionPatterns(typeConverter,
                                                          patterns);
  mlir::populateFuncToLLVMConversionPatterns(typeConverter, patterns);
  mlir::cf::populateControlFlowToLLVMConversionPatterns(patterns, &getContext());

// The only remaining operation, to lower from the `toy` dialect, is the
// PrintOp.
  patterns.add<PrintOpLowering>(&getContext());

```

### dump llvm ir

```
  llvm::LLVMContext llvmContext;
auto llvmModule = mlir::translateModuleToLLVMIR(module, llvmContext);
```

```
  llvm::LLVMContext llvmContext;
auto llvmModule = mlir::translateModuleToLLVMIR(module, llvmContext);
```

### 设置JIT

```cpp
auto maybeEngine = mlir::ExecutionEngine::create(module,
/*llvmModuleBuilder=*/nullptr, optPipeline);
  assert(maybeEngine && "failed to construct an execution engine");
auto &engine = maybeEngine.get();

// Invoke the JIT-compiled function.
auto invocationResult = engine->invoke("main");
```