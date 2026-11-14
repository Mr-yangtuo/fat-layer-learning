# LayerLab · 脂肪层分割学习手册

中文静态学习网站：从原项目数据、模型、损失、训练、评估到端侧导出，建立理论与源码之间的对应。

- 12 节项目报告，包含重要公式、结构图、交互式 Soft-argmax 演示。
- 19 个完整核心源码文件，2,572 行，114 个讲解片段。
- 原理与语法分开解释；手写练习、参考答案和掌握进度。
- 全文搜索、文件下载、深链接、响应式布局、打印。
- 单文件 HTML，无 CDN、无网络依赖，本地双击即可阅读。

## 在线访问

[打开学习手册](https://Mr-yangtuo.github.io/fat-layer-learning/)

## 本地访问

直接打开 docs/index.html，或运行：

    python -m http.server 8765 --directory docs

访问 http://localhost:8765 。

## 重新生成

仅需 Python 标准库：

    python build.py

build.py 验证讲解行号、所有自定义函数覆盖、来源链接与片段顺序，将模板、样式、脚本、讲解和源码合并到 docs/index.html。

在原项目的 learning-site 目录下运行，会更新源码快照；独立克隆本仓库时使用已提交的 source/ 快照。修改原源码以后也必须更新对应讲解，重新审核语义，不应只更新快照。

## 内容布局

- report.html：项目汇报正文。
- content-{data,model,training}.json：逐段讲解。
- source/：原项目核心源码快照，保持目录结构及行号。
- research-report.json：训练流程与实验记录分析。
- evidence.json：所展示实验数字的最小来源记录。
- app.js / style.css / template.html：交互、排版和模板。
- docs/index.html：可独立打开、发布的最终网页。
- build-manifest.json：构建统计与源码 SHA256（UTF-8、换行规范化）。

只发布学习网站及核心代码，不包含原始数据集、图像、清单样本路径、模型权重、缓存或训练日志。

## 理论与实验证据的边界

本网站是基于源码及本地已有 JSON 结果制作的学习材料，没有重新训练模型。结构尺寸按默认 torchvision MobileNetV3 结构静态推导。train100 验证图与训练图重叠，不能作为泛化指标；train90 采用目录分组留出，不自动代表患者级隔离或独立最终测试。

原项目未被修改。已知问题在讲解中明确列出，包括增强错位、参数被覆盖、掩码遗漏、HD 距离忽略横坐标，以及导出缺少数值等价校验。

练习与学习进度仅保存在浏览器 localStorage，不上传服务器。清除浏览器网站数据会清空本地学习记录。

## 发布

GitHub Pages 使用 main 分支的 /docs 目录。提交更新后的网页即可触发发布。
