# SI507 期末项目说明

这个仓库是 **Stock and News Network Explorer** 的课程提交版。

目标很明确：

- 满足 SI507 期末作业要求
- 能直接部署到 Streamlit Cloud
- 保留 CLI、测试、学习文档和本地数据快照

## 项目核心

这个项目不是股价预测器，也不是投资建议工具。

它的核心是一个图结构：

- `stock` 节点
- `sector` 节点
- `topic` 节点

核心边包括：

- `stock_stock`
- `stock_sector`
- `stock_topic`
- `sector_sector`

项目用图来组织和解释：

- 股票之间的结构关系
- 行业之间的联动
- 新闻主题暴露
- 新输入新闻对当前图结构可能带来的影响

## 这个仓库里有什么

- `main.py`
  - CLI 主入口
- `query_app.py`
  - 统一网页入口
  - 里面已经整合了查询工作台和交互图
- `interactive_graph_app.py`
  - 交互图的独立入口
- `streamlit_app.py`
  - 查询工作台主体
- `data_loader.py` / `seed_data.py`
  - 数据下载与本地缓存
- `news_processor.py`
  - 新闻结构化
- `network_builder.py`
  - 图构建
- `network_analyzer.py`
  - 图分析
- `llm_news_impact_analyzer.py`
  - 新新闻影响评估
- `news_graph_augmenter.py`
  - 新新闻接回图结构
- `tests/`
  - 核心测试
- `study_docs/`
  - 按顺序阅读的学习文档

## 本地运行

安装依赖：

```bash
pip install -r requirements.txt
```

运行 CLI：

```bash
python main.py
```

运行网页：

```bash
streamlit run query_app.py
```

## 网页说明

网页入口是 `query_app.py`。

它现在不是单一页面，而是一个统一站点：

- `Analysis Dashboard`
  - 股票查询
  - 行业浏览
  - 股票比较
  - 路径分析
  - 中心节点
  - 历史新闻 LLM impact
  - 新新闻分析
- `Interactive Graph`
  - 可拖拽网络图
  - 过滤节点类型和边类型
  - 查看图结构和桥接节点

## Streamlit Cloud 部署

部署时填写：

- `Branch`: `main`
- `Main file path`: `query_app.py`

如果你想启用 `New News Analysis` 里的 OpenAI 功能，在 Streamlit secrets 里加：

```toml
OPENAI_API_KEY = "你的_openai_api_key"
```

## 当前数据快照

当前仓库已经包含课程演示需要的本地数据快照：

- `50` 只股票价格
- `50` 条行业元数据
- `50` 个单 ticker 新闻文件
- `41,913` 条去重后的新闻文章

所以即使不重新下载数据，也可以直接运行主要功能。

## 建议学习顺序

不要直接扎进代码。

先读：

1. `study_docs/01_项目全景与最终目标.md`
2. `study_docs/02_数据源与本地数据快照.md`
3. `study_docs/04_图是怎么构建出来的.md`
4. `study_docs/05_图分析层到底在回答什么问题.md`

然后实际运行：

```bash
python main.py
streamlit run query_app.py
```

最后再看：

- `study_docs/03_新闻处理层是怎么工作的.md`
- `study_docs/06_交互层为什么分成CLI和网页.md`
- `study_docs/07_LLM新闻影响分析层.md`
- `study_docs/08_测试_验证与可信度.md`

## 一句话总结

这个项目做的事情是：

**用真实股票、行业、新闻主题数据构建一个图，然后在这个图上做查询、比较、路径分析、新闻影响解释和新新闻结构预览。**
