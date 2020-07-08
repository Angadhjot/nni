# 搜索空间

## 概述

在 NNI 中，Tuner 会根据搜索空间来取样生成参数和网络架构。搜索空间通过 JSON 文件来定义。

To define a search space, users should define the name of the variable, the type of sampling strategy and its parameters.

* An example of a search space definition is as follow:

```yaml
{
    "dropout_rate": {"_type": "uniform", "_value": [0.1, 0.5]},
    "conv_size": {"_type": "choice", "_value": [2, 3, 5, 7]},
    "hidden_size": {"_type": "choice", "_value": [124, 512, 1024]},
    "batch_size": {"_type": "choice", "_value": [50, 250, 500]},
    "learning_rate": {"_type": "uniform", "_value": [0.0001, 0.1]}
}

```

将第一行作为示例。 `dropout_rate` is defined as a variable whose priori distribution is a uniform distribution with a range from `0.1` to `0.5`.

Note that the available sampling strategies within a search space depend on the tuner you want to use. We list the supported types for each builtin tuner below. 对于自定义的 Tuner，不必遵循鞋标，可使用任何的类型。

## 类型

所有采样策略和参数如下：

* `{"_type": "choice", "_value": options}`
  
  * The variable's value is one of the options. 这里的 `options` 应该是字符串或数值的列表。 可将任意对象（如子数组，数字与字符串的混合值或者空值）存入此数组中，但可能会产生不可预料的行为。
  * `options` can also be a nested sub-search-space, this sub-search-space takes effect only when the corresponding element is chosen. The variables in this sub-search-space can be seen as conditional variables. Here is an simple [example of nested search space definition](https://github.com/microsoft/nni/tree/master/examples/trials/mnist-nested-search-space/search_space.json). If an element in the options list is a dict, it is a sub-search-space, and for our built-in tuners you have to add a `_name` key in this dict, which helps you to identify which element is chosen. Accordingly, here is a [sample](https://github.com/microsoft/nni/tree/master/examples/trials/mnist-nested-search-space/sample.json) which users can get from nni with nested search space definition. See the table below for the tuners which support nested search spaces.

* `{"_type": "randint", "_value": [lower, upper]}`
  
  * Choosing a random integer between `lower` (inclusive) and `upper` (exclusive).
  * 注意：不同 Tuner 可能对 `randint` 有不同的实现。 Some (e.g., TPE, GridSearch) treat integers from lower to upper as unordered ones, while others respect the ordering (e.g., SMAC). If you want all the tuners to respect the ordering, please use `quniform` with `q=1`.

* `{"_type": "uniform", "_value": [low, high]}`
  
  * The variable value is uniformly sampled between low and high.
  * 当优化时，此变量值会在两侧区间内。

* `{"_type": "quniform", "_value": [low, high, q]}`
  
  * The variable value is determined using `clip(round(uniform(low, high) / q) * q, low, high)`, where the clip operation is used to constrain the generated value within the bounds. 例如，`_value` 为 [0, 10, 2.5]，可取的值为 [0, 2.5, 5.0, 7.5, 10.0]; `_value` 为 [2, 10, 5]，可取的值为 [2, 5, 10]。
  * 适用于离散，同时反映了某种"平滑"的数值，但上下限都有限制。 If you want to uniformly choose an integer from a range [low, high], you can write `_value` like this: `[low, high, 1]`.

* `{"_type": "loguniform", "_value": [low, high]}`
  
  * The variable value is drawn from a range [low, high] according to a loguniform distribution like exp(uniform(log(low), log(high))), so that the logarithm of the return value is uniformly distributed.
  * 当优化时，此变量必须是正数。

* `{"_type": "qloguniform", "_value": [low, high, q]}`
  
  * The variable value is determined using `clip(round(loguniform(low, high) / q) * q, low, high)`, where the clip operation is used to constrain the generated value within the bounds.
  * 适用于值是“平滑”的离散变量，但上下限均有限制。

* `{"_type": "normal", "_value": [mu, sigma]}`
  
  * The variable value is a real value that's normally-distributed with mean mu and standard deviation sigma. 优化时，此变量不受约束。

* `{"_type": "qnormal", "_value": [mu, sigma, q]}`
  
  * The variable value is determined using `round(normal(mu, sigma) / q) * q`
  * 适用于在 mu 周围的离散变量，且没有上下限限制。

* `{"_type": "lognormal", "_value": [mu, sigma]}`
  
  * The variable value is drawn according to `exp(normal(mu, sigma))` so that the logarithm of the return value is normally distributed. 当优化时，此变量必须是正数。

* `{"_type": "qlognormal", "_value": [mu, sigma, q]}`
  
  * The variable value is determined using `round(exp(normal(mu, sigma)) / q) * q`
  * 适用于值是“平滑”的离散变量，但某一边有界。

## 每种 Tuner 支持的搜索空间类型

|                     |  choice  | choice(nested) | randint  | uniform  | quniform | loguniform | qloguniform |  normal  | qnormal  | lognormal | qlognormal |
|:-------------------:|:--------:|:--------------:|:--------:|:--------:|:--------:|:----------:|:-----------:|:--------:|:--------:|:---------:|:----------:|
|      TPE Tuner      | &#10003; |    &#10003;    | &#10003; | &#10003; | &#10003; |  &#10003;  |  &#10003;   | &#10003; | &#10003; | &#10003;  |  &#10003;  |
| Random Search Tuner | &#10003; |    &#10003;    | &#10003; | &#10003; | &#10003; |  &#10003;  |  &#10003;   | &#10003; | &#10003; | &#10003;  |  &#10003;  |
|    Anneal Tuner     | &#10003; |    &#10003;    | &#10003; | &#10003; | &#10003; |  &#10003;  |  &#10003;   | &#10003; | &#10003; | &#10003;  |  &#10003;  |
|   Evolution Tuner   | &#10003; |    &#10003;    | &#10003; | &#10003; | &#10003; |  &#10003;  |  &#10003;   | &#10003; | &#10003; | &#10003;  |  &#10003;  |
|     SMAC Tuner      | &#10003; |                | &#10003; | &#10003; | &#10003; |  &#10003;  |             |          |          |           |            |
|     Batch Tuner     | &#10003; |                |          |          |          |            |             |          |          |           |            |
|  Grid Search Tuner  | &#10003; |                | &#10003; |          | &#10003; |            |             |          |          |           |            |
|  Hyperband Advisor  | &#10003; |                | &#10003; | &#10003; | &#10003; |  &#10003;  |  &#10003;   | &#10003; | &#10003; | &#10003;  |  &#10003;  |
|     Metis Tuner     | &#10003; |                | &#10003; | &#10003; | &#10003; |            |             |          |          |           |            |
|      GP Tuner       | &#10003; |                | &#10003; | &#10003; | &#10003; |  &#10003;  |  &#10003;   |          |          |           |            |

已知的局限：

* GP Tuner and Metis Tuner support only **numerical values** in search space (`choice` type values can be no-numerical with other tuners, e.g. string values). GP Tuner 和 Metis Tuner 都使用了高斯过程的回归（Gaussian Process Regressor, GPR）。 GPR 基于计算不同点距离的和函数来进行预测，其无法计算非数值值的距离。

* 请注意，对于嵌套搜索空间：
  
      * Only Random Search/TPE/Anneal/Evolution tuner supports nested search space