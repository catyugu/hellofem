# hellofem 任务：抄写 basix + dolfinx 模块并复现 mpfem

## 1. 背景

当前 `hellofem` 仓库是空骨架（`hellofem/foo.cpp` + 基础 CMake，无功能、无依赖）。本任务把它变成一个有限元库 + 一个可对标 COMSOL 的应用：

1. **抄写 basix cpp 全部内容**（有限元基底定义子模块），**抄写 dolfinx 的 common/geometry/graph/mesh/io 五模块**，io 扩展 mphtxt。
2. **替换掉 boost 等外部依赖**（std / Eigen / 自研实现替代），命名空间统一 `hellofem::`；剪裁注释但**保留每个文件原有的 license 头**。
3. 在库之上**复现 catyugu/mpfem**（电-热-力耦合 FEM，**通用化**：不钉死固定场），应用层放 `app/mpfem/`，不污染库。
4. **有限元装配模块作为库模块 `hellofem::fem`**（转录 dolfinx fem，PETSc→Eigen），不做成应用层私有实现。
5. 求解器支持：稀疏矩阵预条件迭代（AMG、DDM 等）+ **无矩阵装配迭代**；非线性支持 JFNK（矩阵/无矩阵差分近似）与 **Anderson 加速 Picard**（参考 MetaHotspot）。
6. 案例解析**不另写 XML**，直接用 COMSOL Java 建模脚本；用 autocomsol 产出 COMSOL 参考结果，单脚本封装整条工序；结果对比脚本重写为通用版。

**许可**：basix/dolfinx 均为 MIT，道义起见，保留其文件开头的声明（如果需要）。

## 2. 约束与已定决策（实现时不得偏离）

| 项目       | 决策                                                                                                                         |
| ---------- | ---------------------------------------------------------------------------------------------------------------------------- |
| 命名空间   | 统一 `hellofem::`（basix→`hellofem::basis`，dolfinx 模块→`hellofem::<mod>`，fem→`hellofem::fem`，新增求解器 `hellofem::la`） |
| MPI        | **必需依赖**，`find_package(MPI REQUIRED)`，**不做单进程回退**。                                                             |
| io 范围    | VTK(vtu) + 新增 mphtxt 读写 + COMSOL Data 文本导出；**丢弃** XDMF/ADIOS2/HDF5                                                |
| AMG/DDM    | **自研** Ruge-Stüben 经典 AMG + 加性 Schwarz(DDM)，零新增依赖                                                                |
| Eigen      | **GitLab Eigen 5.0**（`gitlab.com/libeigen/eigen`）经 CPM；具体 tag 实现时解析，拉取失败回退最新稳定并记录                   |
| 案例       | **自行新建** busbar_steady/order2/transient + 新造案例（走 COMSOL 流水线），放 **`app/mpfem/cases/`**                        |
| 装配归属   | 库模块 `hellofem::fem`                                                                                                       |
| 表达式     | muparser（CPM）                                                                                                              |
| 物理层     | **通用化**：通用场 / 弱式抽象组织物理，不在固定场（V/T/disp）或固定物理类上钉死；新增场仅需声明即可扩展                      |
| 应用层包装 | **去 `core/` 等纯薄包装**：直接用 `hellofem::common`（types/log）与 Eigen，不重复包装 types/logger/tensor/exception          |
| 依赖管理   | 全部经 CPM（spdlog / Eigen / muparser / Catch2），学 mpfem 的 `Dependencies.cmake` + `Targets.cmake` 范式                    |

## 3. 库模块布局与依赖

```text
hellofem/
├── basis/        # ★抄写 basix cpp 全量（hellofem::basis）
├── common/       # ★抄写 dolfinx common
├── graph/        # ★抄写 dolfinx graph
├── mesh/         # ★抄写 dolfinx mesh
├── geometry/     # ★抄写 dolfinx geometry（纯头文件）
├── io/           # ★抄写 dolfinx io（裁剪）+ 新增 mphtxt/COMSOL 文本
├── fem/          # ★抄写 dolfinx fem 装配（PETSc→Eigen，矩阵+无矩阵）
└── la/           # ★新增：线性代数与求解器（Eigen 5.0）
```

依赖方向（单向、无环）：`common ← graph ← mesh ← geometry`；`basis`(Eigen) 独立；`la`(common, Eigen)；`io`(common, mesh)；`fem`(common, graph, mesh, geometry, basis)；`app/mpfem` → 全部 + muparser + spdlog。

构建：每模块静态库 `hellofem_<mod>`（alias `hellofem::<mod>`）+ 汇总目标 `hellofem`；复用 mpfem 的 `mpfem_add_library` 函数模式（改名 `hellofem_add_library`）。

## 4. 任务清单（分阶段，每阶段验收通过后再进入下一阶段）

### P0 脚手架：CMake + CPM + MPI

- [ ] 根 CMake 重组：`cmake/CPM.cmake(最新)`、`Dependencies.cmake`（CPM 拉 spdlog / Eigen 5.0(GitLab) / muparser / Catch2；`find_package(MPI REQUIRED)`）、`Targets.cmake`（`hellofem_add_library`）
- [ ] 建立模块空库骨架 `hellofem/{basis,common,graph,mesh,geometry,io,fem,la}/`，删 `foo.cpp` 骨架
- **验收**：`cmake --build build` 绿；ctest 空跑绿；`tools/mpi_probe` 并入 MPI 检查

### P1 hellofem::basis：抄写 basix

- [ ] 全量抄写：`cell, dof-transformations, element-families, finite-element, indexing, interpolation, lattice, maps, math, moments, polynomials, polyset, precompute, quadrature, sobolev-spaces, types, mdspan.hpp(保留)`
- [ ] 14 个元素族：`e-lagrange, e-nce-rtc, e-brezzi-douglas-marini, e-nedelec, e-raviart-thomas, e-regge, e-hhj, e-crouzeix-raviart, e-bubble, e-serendipity, e-hermite`
- [ ] BLAS/LAPACK（`math::matvec` 等小规模稠密运算）→ Eigen 稠密 / 手写循环
- [ ] 命名空间 `basix`→`hellofem::basis`；include 路径 `basix/`→`hellofem/basis/`；保留 SPDX 头
- [ ] Catch2 测试：各元素族 tabulation 与 basix 已知值一致
- **验收**：`tests/basis` 数值测试全绿

### P2 common/graph/mesh/geometry：抄写 dolfinx + 替换 boost

- [ ] **common**：`defines, IndexMap, log(spdlog), math, sort, types, local_range, MPI, Scatterer, Table, Timer, TimeLogger, timing, utils, version.h.in`
  - `utils.h` `boost::functional::hash` → 自研 hash_combine；`boost::lexical_cast` → `std::from_chars/to_chars/stod`
- [ ] **graph**：`AdjacencyList, ordering(RCM/AMD), partitioners, partition, utils`
  - `boost::sort`（`sort_by_perm`）→ `std::sort`/自研基数排序；partitioners 的 MPI+ParMETIS/PT-SCOTCH/KaHIP 分支删除，**新增自研串行分区器**（坐标剖分 + 图多级/贪婪 k 剖分）供 mesh 与 DDM
- [ ] **mesh**：`Mesh, Geometry, Topology, MeshTags, EntityMap, cell_types, graphbuild, permutationcomputation, topologycomputation, generation, utils`
  - `boost::sort` → `std::sort`；`boost/unordered_map` → `std::unordered_map`；`boost::hash` → 自研；Topology/IndexMap 走 MPI 分布式路径
- [ ] **geometry**（纯头文件）：`BoundingBoxTree.h, gjk.h, utils.h`
  - `gjk.h` `boost::multiprecision/cpp_bin_float` → 基于 double 的稳健实现
- [ ] Catch2 测试：网格/topology/BBT/ordering
- **验收**：从 mphtxt 建 Mesh、topology 正确、BBT 查询正确

### P3 hellofem::io：VTK + mphtxt + COMSOL 文本

- [ ] 抄写 `cells.h/cpp, vtk_utils.h/cpp, utils.h`；丢弃 ADIOS2/HDF5/VTKFile/VTKHDF/XDMF/xdmf_*
- [ ] 新增 mphtxt 读写器（读 COMSOL 导出 `.mphtxt`：顶点/元素/几何实体编号/子域与边界标记）
- [ ] 新增 COMSOL Data 文本导出（`%` 头 + 行列，匹配 result.txt 格式）
- [ ] Catch2 测试：mphtxt 往返；读入 busbar `mesh.mphtxt` 校验 7 domain / 43 boundary
- **验收**：io 测试全绿

### P4 hellofem::la：线性代数与求解器

- [ ] 类型 `Real=double`；`LinearOperator`(matvec) + `Preconditioner`(apply) 接口
- [ ] 迭代求解器（算子级）：CG、BiCGStab、GMRES(m) —— 天然支持无矩阵
- [ ] 预条件：Diagonal(Jacobi)、SSOR、ILU(0)；**自研 AMG**（Ruge-Stüben：strength-of-connection、C/F 分裂、插值、粗化、V/W 循环、GS/Jacobi smoother）；**自研 DDM**（加性 Schwarz：partition 向量 + 稀疏矩阵 → 块 Jacobi）
- [ ] 直接法兜底：Eigen::SparseLU / SparseQR / SimplicialLDLT
- [ ] 非线性：矩阵基牛顿；**JFNK**（无矩阵有限差分 `Jv≈(F(x+hv)-F(x))/h` + GMRES 内迭代，另有矩阵版）；**Picard+Anderson**（移植 MetaHotspot `AndersonMixer`：depth=5, warmup=2, dampening, divergence guard，泛化模板/算子版）
- [ ] 统一 `SolverConfig` + Factory
- [ ] Catch2 测试：泊松收敛、AMG 收敛、JFNK / Anderson-Picard 收敛
- **验收**：la 测试全绿

### P5 hellofem::fem：装配模块

- [ ] 转录 dolfinx fem 核心：`FiniteElement`(包 basis)、`DofMap`(+Builder)、`FunctionSpace`、`Function`、`Form`、`dirichletbc`、`coord_element`、`element`、`utils`；剔除 petsc 部分
- [ ] **通用装配内核**（不照搬 dolfinx 代码生成的逐元素特化）：`assemble_matrix`（稀疏）+ **无矩阵算子版**（cell 级 matvec → `hellofem::la::LinearOperator`）；`assemble_vector`、`assemble_scalar`；DirichletBC 施加（罚/消除）
- [ ] 剔 MPI 分布式路径；保留 MPI 通信原语
- [ ] Catch2 测试：patch test、DofMap、无矩阵 matvec 与稀疏矩阵一致
- **验收**：fem 测试全绿

### P6 app/mpfem：复现 mpfem（通用化，去纯包装层）

> 不用纯包装层：app 直接使用 `hellofem::common`（types/log/…）与 Eigen，不另建 `core/` 薄包装。
> 通用化：用**通用场 / 弱式抽象**组织物理，不在固定场（V/T/disp）或固定物理类上钉死，便于扩展新场/新物理。

- [ ] `expr/`：muparser 替换自研 Pratt parser；unit 处理；variable_graph（表达式依赖）
- [ ] **通用场抽象**：`Field`（未知量名 + FE 元素 + 取值网格函数），字段按名注册，不硬编码 V/T/disp
- [ ] **通用弱式抽象**：`EquationTerm`（积分式：系数 × 形状函数组合，如 `a∇u·∇v`、`c·u·v`、`f·v`）；material 属性为按名系数表达式（muparser，可依赖坐标/时间/其他场）
- [ ] `assembly/`：由 `hellofem::fem` 按通用弱式装配（稀疏/算子版）+ 通用 DirichletBC（boundary tag + 表达式）
- [ ] `solver/`：全部走 `hellofem::la`
- [ ] `io/`：case 解析用 COMSOL Java（替代 XML）→ 通用 `ModelSpec`；material 内嵌 Java；result 导出（COMSOL txt + VTU）
- [ ] `physics/`：用通用弱式**声明** electrostatics / heat_transfer / structural（焦耳热、热膨胀耦合）为若干 `EquationTerm` 集合；新增场仅需声明，不改求解器
- [ ] `problem/`：steady + transient（BDF1→BDF2/CN），时间步进
- [ ] `examples/busbar_example.cpp`；跑通 `app/mpfem/cases/busbar_steady`
- [ ] **扩展性验证**：用同一套通用抽象另定义一个新标量场（如扩散/传质），不新增物理类即可跑通
- **验收**：busbar_steady 的 V/T/disp 对拍达标（L2_rel < 1e-5 / 1e-4 / 1e-2）；扩展性验证通过

### P7 COMSOL 流水线 + 自建案例 + 对比脚本 + pytest

- [ ] `scripts/comsol_pipeline.py` 单脚本封装：Java 建模 → comsolcompile → comsolbatch 计算 .mph → 导出 mphtxt + 结果文本 → 跑 app 二进制 → 对拍报告
- [ ] 干净 Java 解析：app 端解析惯例化编写的 Java（`model.param().set / physics().create+feature().set / material / study`）→ `ModelSpec`
- [ ] **自建** `busbar_steady / busbar_steady_order2 / busbar_transient` + 新造案例，跑出 COMSOL 参考结果
- [ ] `scripts/compare_results.py` 通用重写：自动读 Expressions、坐标对齐、逐字段 L2/Linf/max_rel/L2_rel、瞬态多时间块、CLI 可配置、JSON 报告
- [ ] pytest：`app/mpfem/tests/*.py` 解析 Java、跑二进制、对拍 COMSOL
- **验收**：端到端新案例对拍通过；`pytest` 全绿

## 5. 测试策略

- 内部：Catch2 + ctest，`tests/<module>/` 每模块一目录。
- 应用：pytest only（`pytest.ini` 已设 `testpaths=app/mpfem/tests`）。
- 每个行为变更遵循 **TDD**：先加失败测试 → 实现 → 绿。

## 6. 验证命令

- 构建：`cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build --parallel`
- 内部单测：`ctest --test-dir build --output-on-failure`
- 应用验证：`python -m pytest -c pytest.ini`
- 端到端：`python scripts/comsol_pipeline.py app/mpfem/cases/<new_case>` → 对拍报告

## 7. 风险与注意事项

- COMSOL mesh→mphtxt 精确 Java API：实现时用 probe 模型验证（autocomsol lab-notebook 只验证过 `result().export() Data→txt`）。
- Eigen 5.0(GitLab) 具体 tag：实现时解析；拉取失败回退最新稳定并记录。
- MPI 必需依赖：多机/集群运行需完整 MS-MPI 发行版。
- AMG 鲁棒性：先 1D/2D 泊松验证再上 3D 多物理。
- basix 的 LAPACK 依赖点需精确定位并逐点替换为 Eigen。

## 8. 参考链接

- basix：<https://github.com/FEniCS/basix/tree/main/cpp/basix>
- dolfinx（common/geometry/graph/mesh/io/fem）：<https://github.com/FEniCS/dolfinx/tree/main/cpp/dolfinx>
- mpfem（复现目标）：<https://github.com/catyugu/mpfem>
- MetaHotspot（Anderson 参考）：<https://github.com/catyugu/MetaHotspot/blob/main/src/solver/nonlinear_solver.cpp>
- autocomsol（COMSOL Java 建模）：<https://github.com/catyugu/autocomsol>
- Eigen 5.0：<https://gitlab.com/libeigen/eigen>
