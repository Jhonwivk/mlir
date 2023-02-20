# chap3 分析和转换

目标：

```cpp
def main() {
  var a<2,1> = [1, 2];
  var b<2,1> = a;
  var c<2,1> = b;
  print(c);
}
```

```llvm
module {
  toy.func @main() {
    %0 = toy.constant dense<[1.000000e+00, 2.000000e+00]> : tensor<2xf64>
    %1 = toy.reshape(%0 : tensor<2xf64>) to tensor<2x1xf64>
    %2 = toy.reshape(%1 : tensor<2x1xf64>) to tensor<2x1xf64>
    %3 = toy.reshape(%2 : tensor<2x1xf64>) to tensor<2x1xf64>
    toy.print %3 : tensor<2x1xf64>
    toy.return
  }
}
```

消除三个reshape，再使用常量折叠

## 模式匹配转换

1.命令式，C++模式匹配和重写.2.声明性的、基于规则的模式匹配和重写，使用声明性重写规则Declarative Rewrite Rules(DRR)

```cpp
class Pattern<
dag sourcePattern,list<dag> resultPatterns,//源模式，目标模式
list<dag> additionalConstraints = [],//额外约束
dag benefitsAdded = (addBenefit 0)>;//优先级
```

冗余转换模式

```cpp
// Reshape(Reshape(x)) = Reshape(x)
def ReshapeReshapeOptPattern : Pat<(ReshapeOp(ReshapeOp $arg)),
                                   (ReshapeOp $arg)>;
```

消除reshape

```cpp
//input ouput type一样约束
def TypesAreIdentical : Constraint<CPred<"$0.getType() == $1.getType()">>;
def RedundantReshapeOptPattern : Pat<
  (ReshapeOp:$res $arg), (replaceWithValue $arg),
  [(TypesAreIdentical $res, $arg)]>; //用参数arg代替reshape

```

常量折叠

```cpp
def ReshapeConstant : NativeCodeCall<"$0.reshape(($1.getType()).cast<ShapedType>())">;
def FoldConstantReshapeOptPattern : Pat<
  (ReshapeOp:$res (ConstantOp $arg)),
  (ConstantOp (ReshapeConstant $arg, $res))>;

```

`toyc-ch3 test/Examples/Toy/ch3/trivial_reshape.toy -emit=mlir -opt` 优化后

```cpp
module {
  func @main() {
    %0 = toy.constant dense<[[1.000000e+00], [2.000000e+00]]> : tensor<2x1xf64>
    toy.print %0 : tensor<2x1xf64>
    toy.return
  }
}
```