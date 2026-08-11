# TASK

学习并抄写：<https://github.com/FEniCS/basix/tree/main/cpp/basix> 的 cpp 全部内容（作为有限元基底定义子模块），<https://github.com/FEniCS/dolfinx/tree/main/cpp/dolfinx> 中的 common, geometry, graph, mesh, io模块（需要为mphtxt网格格式进行扩展），但是将所有对 boost 等外部的依赖替换掉，命名空间替换到 hellofem，消除项目元信息等对我们的库没用的内容。剪裁注释，但是内容和原始文件中的license内容应该予以保留。然后，尝试进行 <https://github.com/catyugu/mpfem> 的复现。
要求：

1. 同时支持稀疏矩阵的预条件迭代求解（包括 AMG、DDM 等较高级算法）以及无矩阵法的装配迭代求解
2. 不用 MPI 多进程，所有 MPI 相关代码转化为多线程，建议是引入 oneTBB 实现并行化
3. 非线性求解支持：（无矩阵和有矩阵的）基于差分近似的 JFNK 迭代，以及 Anderson 加速的 Picard 迭代，Anderson 的实现参考<https://github.com/catyugu/MetaHotspot/blob/main/src/solver/nonlinear_solver.cpp>
4. 库中应支持所有 basix 支持的基底定义
5. 应用层的内容搞到应用层app/mpfem文件夹中，不要污染库，底层只是离散化和数值工具
6. 日志库用 spdlog，线性代数库用 Eigen 5.0，内部库的测试用Catch2配合ctest进行，表达式解析用muparser，应用层的测试只用pytest做结果验证
7. 全部依赖利用 CPM 引入，可学习 catyugu/mpfem 中引入依赖的范式
8. 案例的模型解析不应另写 XML，而是直接利用 COMSOL 的 Java 建模脚本
9. 请自己新造案例并跑出 COMSOL 结果，可以利用 <https://github.com/catyugu/autocomsol>，首先写一个 Java 建模脚本，产生 .mph 文件，然后从 .mph 计算结果，再导出供解析用的更干净的 Java 脚本以及 .mphtxt 格式的网格。我们的程序应该吃进倒两手后的干净的 Java 脚本和网格，输出格式和 COMSOL 一致的文件供比对之用。这个处理工序应该用单个脚本封装
10. 进行结果对比的脚本可能需要重新写过（为了更好的通用性）
