# AI Study Record

这是我的大模型学习记录仓库，重点记录三类内容：

- 学习路线和阶段计划
- 每日/阶段性总结与概念整理
- 配套的 PyTorch 基础与大模型方向 demo

## 当前结构

```text
.
├─ references/
├─ docs/
│  ├─ course-notes/
│  ├─ daily-checkins/
│  ├─ guides/
│  ├─ roadmap/
│  └─ summaries/
├─ code/
│  ├─ fundamentals/
│  └─ llm/
├─ scripts/
├─ d2l-zh-master/          # 本地参考教材，默认不上传
└─ README.md
```

## 重点内容

- 路线图：
  - [15天数学基础到主线衔接学习计划](docs/roadmap/10天大模型数学基础学习计划.md)
  - [数学基础之后的大模型主线学习路线](docs/roadmap/数学基础之后的大模型主线学习路线.md)
  - [15天之后的大模型项目学习路线](docs/roadmap/15天之后的大模型项目学习路线.md)
- 总结：
  - [大模型数学基础学习总结](docs/summaries/大模型数学基础学习总结.md)
- 课程笔记：
  - [Dive Into LLMs 课程学习笔记索引](docs/course-notes/dive-into-llms/README.md)
- 入门指引：
  - [D2L-ZH Start Here](docs/guides/d2l_zh_start_here.md)

## 当前主线与历史资料的区分

- `docs/`、`code/`、`scripts/`
  - 当前这条线是我自己整理和持续更新的学习记录、计划、总结与 demo。
  - `docs/course-notes/` 中也包含我自己写的课程学习笔记与知识总结。
- `references/`
  - 这里存放我之前学过的外部教程或专题学习包，作为历史参考资料归档。
  - 当前已归档：
    - [dive-into-llms-main reference note](references/dive-into-llms-main/README.md)

## 代码分区

- `code/fundamentals/`
  - 张量、线代、autograd、softmax、MLP 等基础 demo
- `code/llm/`
  - attention、Transformer、LoRA、RAG 等大模型方向 demo
- `scripts/`
  - 环境检查和辅助脚本

## 快速开始

在仓库根目录运行：

```powershell
python scripts/d2l_zh_smoke_test.py
python code/fundamentals/day01_tensor_demo.py
```

## 说明

- `d2l-zh-master/` 是本地参考教材源码，作为学习辅助材料保留在本地。
- 大体积 PDF 和临时产物默认不纳入 GitHub 记录。
- `references/dive-into-llms-main/` 现在只保留来源说明，不再镜像原始课件、图片、notebook 或上游文本。
- 对于没有明确许可证的外部项目，我只在公开仓库里保留引用说明和我自己的学习记录，不公开再分发原始资料。
