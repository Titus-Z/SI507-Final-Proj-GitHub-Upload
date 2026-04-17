# 前端设计风格导出文档

这份文档是当前 **SI507 Final Project** 网页前端的设计规范导出版本。

它不是泛泛的设计说明，而是对当前项目里已经落地的 Streamlit 风格进行整理，方便后续：

- 继续改 UI
- 保持不同页面视觉一致
- 向老师解释“为什么这个网页看起来是这样”
- 避免后面越改越乱

---

## 1. 设计目标

这套前端不是做成“交易终端”，也不是做成“聊天机器人页面”。

目标非常明确：

1. 看起来像一个 **图驱动的市场分析工作台**
2. 强调 **结构、关系、浏览、钻取**
3. 比普通课程项目更成熟，但仍然保持可解释、不过度炫技

所以前端风格的核心关键词是：

- graph-first
- research dashboard
- market intelligence
- clean but not flat
- information density with hierarchy

一句话概括：

**这是一个偏研究型、偏分析型、偏图结构探索的金融数据工作台。**

---

## 2. 整体视觉方向

整体视觉方向不是暗黑极客风，也不是五颜六色的信息图风。

当前项目采用的是：

- **浅色主画布**
- **深色侧边栏**
- **高对比的 hero 区**
- **卡片化信息组织**
- **图节点 / 图边使用稳定编码**

这样做的原因：

1. 浅色背景更适合表格、指标和解释型文本
2. 深色侧边栏能把配置区和分析区分开
3. hero 区可以快速建立页面语义
4. 卡片结构让信息密度高，但不显得乱

---

## 3. 颜色系统

### 3.0 当前支持的三套交互图风格

交互图页面现在不是固定一套皮肤，而是提供三套风格切换：

- `AWS + Bloom`
  - 更像成熟的 graph explorer 产品
  - 适合课程展示时强调“分析工作台”感
- `AWS + Sigma`
  - 更强调干净画布和现代图浏览器气质
  - 适合突出图本身
- `Kumu + Bloom`
  - 更像系统关系图或网络地图
  - 适合强调结构、桥接和关系网络

这三套风格共享同一套数据、分析逻辑和图构建方式，变化的是：

- 页面底色和 hero 渐变
- 侧边栏语气和整体视觉重量
- 节点 / 边颜色编码
- legend accent
- 图画布背景与字体对比

设计原则是：

**允许切换视觉语气，但不允许破坏语义一致性。**

也就是说：

- `stock / sector / topic` 仍然必须保持稳定、可读的颜色区分
- 交互方式不能因为换风格而变
- 查询逻辑和图结构逻辑不能因为换风格而变

### 3.1 查询工作台主色

查询页主要使用：

- 主青绿色：`#0F766E`
- 主蓝色：`#2563EB`
- 辅助橙色：`#D97706`
- 辅助紫色：`#9333EA`
- 强调红色：`#BE123C`
- 辅助靛蓝：`#4F46E5`

这些颜色主要用于：

- tab 激活态
- snapshot card 顶部边框
- badge
- 局部强调块

查询页底色不是纯白，而是带轻微渐变：

- `#F6F8FC -> #EEF3F8`

并叠加两层 radial gradient：

- 青绿色光晕
- 蓝色光晕

这样页面不会显得平，能稍微接近现代 dashboard 的质感。

### 3.2 交互图页主色

交互图页的视觉方向稍微更硬一点：

- 深海军蓝 hero：`rgba(15, 23, 42, 0.96)`
- 强蓝辅助：`rgba(37, 99, 235, 0.94)`

这是为了让它更像“结构浏览器”，和查询工作台做出区分。

### 3.3 图节点颜色

图节点必须用固定语义颜色，不能随机。

- `stock`: `#2F6BFF`
- `sector`: `#1E9E63`
- `topic`: `#F08C2E`
- `unknown`: `#7F8C8D`

原因：

- 蓝色适合资产实体
- 绿色适合行业层
- 橙色适合新闻/主题层

### 3.4 图边颜色

图边按边类型编码：

- `stock_stock`: `#2563EB`
- `stock_sector`: `#0F766E`
- `stock_topic`: `#D97706`
- `sector_sector`: `#9333EA`
- fallback: `#B0B8C5`

这一步很重要，因为它让图不只是“节点分色”，而是边也有语义。

---

## 4. 布局原则

### 4.1 查询页布局

查询页采用的是：

1. hero 区
2. 核心 metrics
3. market snapshot
4. tabs 钻取

这背后的阅读顺序是：

**先扫描市场，再进入具体查询。**

所以查询页不应该一上来就给用户大量表格或输入框。

### 4.2 图页布局

交互图页现在不是只放一张图，而是拆成：

- `Canvas`
- `Inspector`
- `Structure`

这是参考成熟图浏览器的设计方法：

- 画布负责“看整体”
- Inspector 负责“看一个点”
- Structure 负责“看整体拓扑摘要”

这样用户不会被一个大图淹没。

### 4.3 侧边栏

侧边栏统一承担：

- 输入配置
- 模式切换
- preset 切换
- graph filter
- focus 配置

原则是：

**侧边栏放控制，主区域放结果。**

不要把配置项散落到页面中间。

---

## 5. 组件风格

### 5.1 Hero

Hero 的作用不是装饰，而是定义页面语义。

当前 hero 的标准：

- 大圆角
- 强渐变
- 白字
- 一句标题
- 一段说明

Hero 要短，不能写成长文。

### 5.2 Metric 卡片

Metric 用于：

- stocks / sectors / topics / articles
- filtered nodes / filtered edges / components / avg degree

风格要求：

- 白色半透明背景
- 细边框
- 轻阴影
- 圆角

它们应该看起来是“分析指标块”，不是广告 tile。

### 5.3 Section Card

Section card 用于承接说明性文本。

特点：

- 比 metric 更宽
- 文本多一些
- 用于解释这个区域在干什么

### 5.4 Snapshot Card

Snapshot card 是查询页的核心视觉单元之一。

它负责快速展示某个 sector 的：

- 名称
- stock count
- graph degree
- linked sectors

设计原则：

- 顶部彩色边
- 中间大值
- 下方元信息

这种结构很适合课程项目，因为一眼就能看懂。

### 5.5 Legend Card

Legend card 不只是“告诉用户颜色是什么”，也是在告诉用户：

- 这个图怎么读
- 视觉编码是否稳定

所以 legend 区必须简洁，不能塞太多文案。

### 5.6 Badge

Badge 只用于少量概念提醒，例如：

- overview before query
- graph-grounded analysis
- LLM used for impact assessment, not prediction

Badge 不能滥用，否则会让页面像运营后台。

---

## 6. 图浏览器交互模式

这一部分是当前交互图页面最重要的升级点。

### 6.1 Preset 模式

交互图页现在有这些 preset：

- `Market Structure`
- `Sector Bridges`
- `Topic Exposure`
- `Topic Map`
- `Custom`

这一步的目的不是“多一个下拉框”，而是：

**把常见分析视角包装成可直接切换的视图。**

这是比“用户自己勾选几十个过滤条件”更高级的设计。

### 6.2 Focus Node

Focus Node 的作用是：

- 从全图切到局部邻域
- 限制 1 到 3 hop

这是图产品里非常常见的设计，因为大图天然会密。

原则：

- 大图负责背景
- focus view 负责真实探索

### 6.3 Inspector

Inspector 面板应该始终回答两个问题：

1. 这个节点本身是什么
2. 它连着谁

所以现在拆成：

- node detail table
- neighbor table

这是正确的，因为图浏览器的核心不是“只看图”，而是“图 + 结构化解释”。

### 6.4 Structure 面板

Structure 面板现在放：

- node type summary
- edge type summary
- top degree nodes
- connected components
- legend
- display controls

它的作用是：

**把图从“可视化对象”变成“可分析对象”。**

---

## 7. 信息表达风格

当前前端文案风格应该保持：

- 简洁
- 分析型
- 不夸张
- 不预测股价
- 不用营销语气

网页里所有说明都应围绕：

- exploration
- structure
- exposure
- impact assessment

不要用这些词：

- alpha machine
- stock predictor
- smart recommendation
- AI trading engine

这些词会把项目带偏。

---

## 8. 这套风格不应该变成什么

后续改 UI 时，最容易出问题的是下面这些方向。

### 不要变成纯“科技炫酷风”

如果全页面都是：

- 纯黑背景
- 霓虹渐变
- 夸张光效

那它会更像 demo，而不是课程项目。

### 不要变成金融终端仿制品

你这个项目不是 Bloomberg 终端，也不是量化交易台。

所以不要硬塞：

- K 线大屏
- ticker 跑马灯
- 交易台风格红绿数字墙

### 不要变成纯表格后台

如果只有：

- 表格
- dropdown
- dataframe

没有视觉层次，那又会太普通。

正确方向是：

**分析型卡片 + 图浏览器 + 结构化结果。**

---

## 9. 后续继续升级时的准则

如果以后继续改前端，建议优先做这些：

1. 节点搜索高亮
2. 选中节点后的路径高亮
3. 更明确的 edge legend
4. graph canvas 上方的当前视图摘要
5. 更强的 node inspector

如果以后要做“真正更高级”的交互图，推荐升级方向是：

- 用 Cytoscape 风格前端替代 pyvis
- 支持 click-to-highlight
- 支持 path highlighting
- 支持固定节点和局部展开

但在当前课程项目阶段，现有方案已经足够：

- 比普通 Streamlit dataframe 项目更成熟
- 比纯 pyvis 默认图更可用
- 仍然保持可解释和可维护

---

## 10. 这套风格的一句话定义

如果要用一句话定义当前前端风格，可以这样说：

**A graph-first market research dashboard with a clean light workspace, a dark control rail, stable visual encoding, and a browser-style interaction model for structural exploration.**

中文表达：

**一个以图为中心的市场研究工作台，采用浅色分析画布、深色控制侧栏、稳定视觉编码，以及图浏览器式的结构探索交互。**

---

## 11. 实际落地文件

当前这套风格主要落在这些文件里：

- `streamlit_app.py`
  - 查询工作台的视觉和交互
- `interactive_graph_app.py`
  - 图浏览器的视觉和交互
- `query_app.py`
  - 统一网站入口

如果后续要改风格，优先从这三个文件入手。
