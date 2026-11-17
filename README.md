# NLP 实验作业：自动摘要系统与《红楼梦》机器翻译

本仓库完成自然语言处理实验的两道题目：

- **第五题：自动摘要系统**——对给定的最高人民法院民事判决书自动生成摘要；
- **第六题：机器翻译**——选择 MarianMT 模型，将《红楼梦》选段翻译为英文，并给出模型训练与翻译过程。

运行环境为 Linux、Python 3.13、CPU（无 GPU）。

## 目录结构

```
work2/
├── README.md
├── requirements.txt
├── 红楼梦.txt                              # 课程提供的《红楼梦》全文（GB18030 编码）
├── THUOCL：清华大学开放中文词库/            # 课程提供的 THUOCL 词库
├── task5_summarization/
│   ├── summarize.py                        # 自动摘要脚本
│   ├── data/judgment.txt                   # 测试判决书
│   └── output/summary.txt                  # 生成的摘要
└── task6_translation/
    ├── prepare_data.py                     # 从 OPUS-100 抽取平行语料
    ├── train.py                            # MarianMT 微调脚本
    ├── translate.py                        # 《红楼梦》选段翻译脚本
    ├── data/
    │   ├── opus100_train.parquet           # OPUS-100 en-zh 训练数据
    │   ├── train.tsv / valid.tsv           # 抽取后的训练/验证集
    │   └── hongloumeng_sample.txt          # 《红楼梦》第一回选段
    ├── models/
    │   ├── opus-mt-zh-en/                  # 下载的预训练模型
    │   └── opus-mt-zh-en-ft/               # 微调后的模型
    └── output/
        ├── translation_base.txt            # 预训练模型（未微调）译文
        ├── translation_ft.txt              # 微调模型译文
        └── train.log                       # 训练日志
```

## 环境安装

pip 已配置为使用清华大学镜像源：

```bash
python3 -m pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
python3 -m pip config set install.trusted-host pypi.tuna.tsinghua.edu.cn
```

然后安装依赖：

```bash
python3 -m pip install --user --break-system-packages -r requirements.txt
```

PyTorch 使用 CPU 版本即可（本项目全部实验在 CPU 上完成）：

```bash
python3 -m pip install --user --break-system-packages torch \
  --index-url https://download.pytorch.org/whl/cpu
```

---

## 第五题：自动摘要系统

### 任务与数据

输入是课程给出的最高人民法院民事判决书（约 1.8 万字），输出一段覆盖案件基本信息、争议焦点和裁判结果的摘要。判决书属于典型的**长文档、强结构、领域术语密集**文本，因此采用“抽取式摘要 + 关键信息抽取”的混合方案。

### 实现思路

1. **领域分词**
   使用 `jieba` 分词，并把 `THUOCL_law.txt`（约 9900 个法律领域词及其频次）加入用户词典。这样“举证责任、验收标准、违约金、自有品牌及加盟连锁平台架构”等术语不会被切碎，TF-IDF 权重也更准确。

2. **句子切分与过滤**
   按 `。！？；` 和换行切句；过滤法院名称、案号、审判人员、当事人信息、地址、合同编号条款等对摘要无贡献的片段，避免它们占用摘要篇幅。

3. **句子向量与 TextRank**
   对每个句子做去停用词后的词袋表示，计算 TF-IDF 向量；以句子为节点、窗口内句子的余弦相似度为边，构建 TextRank 图并迭代求解，得到句子的全局重要性。相比逐句打分，TextRank 能利用句子之间的相互支持关系。

4. **判决书结构先验**
   在 TextRank 分数上叠加位置/角色权重：
   - `本院认为`、`争议焦点`、`判决如下`、`驳回上诉`、`维持原判` 等句子加权；
   - 立案、合议庭、到庭参加、审理终结等程序性描述降权；
   - 大段付款比例、金额罗列的合同条款降权；
   - 法律适用（民法典时间效力）等背景性论述降权。

5. **MMR 去冗余**
   用 Maximal Marginal Relevance 逐句挑选：既要求句子得分高，又惩罚与已选句子相似度过高的候选，最终选出 9 句并按原文顺序输出。

6. **关键信息抽取**
   用正则从原文抽取案号、上诉人、被上诉人、案由和裁判结果，作为摘要前的“关键信息”卡片。判决书格式相对固定，规则方法在这里比生成式模型更稳定、可解释。

### 运行

```bash
cd task5_summarization
python3 summarize.py --input data/judgment.txt --output output/summary.txt
```

可选参数：

- `--lexicon`：THUOCL 法律词库路径；
- `--max-sentences`：抽取句子数，默认 9。

### 输出示例

```text
## 关键信息

- 案号：（2021）最高法知民终1191号
- 上诉人：北京原告公司
- 被上诉人：苏州被告公司
- 案由：计算机软件开发合同纠纷
- 裁判结果：驳回上诉，维持原判。

## 抽取式摘要

1. 原告公司上诉请求：撤销原审判决，改判驳回被告公司的全部诉讼请求。
2. 事实和理由：（一）原审法院认定被告公司未完成“自有品牌及加盟连锁平台架构部分”的开发工作……
...
9. 驳回上诉，维持原判。
```

完整结果见 [`task5_summarization/output/summary.txt`](task5_summarization/output/summary.txt)。

### 方法说明

- 采用抽取式方法，摘要中的每个句子都来自原文，不会出现事实性编造，适合判决书这类对准确性要求高的文本。
- 没有使用生成式大模型，原因是本地无 GPU、判决书领域性强，抽取式方案更容易复现和解释。
- 本方法不依赖标注数据，属于无监督摘要；THUOCL 词库和判决书结构规则是仅有的领域先验。

---

## 第六题：机器翻译

### 模型与数据

- **模型**：`Helsinki-NLP/opus-mt-zh-en`（MarianMT，约 7700 万参数），已在中英平行语料上预训练。
- **调优数据**：`Helsinki-NLP/opus-100` 的 `en-zh` 子集，抽取 20000 条训练、1000 条验证。数据为通用领域中英句对。
- **测试文本**：《红楼梦》第一回“甄士隐梦幻识通灵”选段（766 字），见 `data/hongloumeng_sample.txt`。

按照作业要求，下载预训练模型后不是直接使用，而是在 OPUS-100 上做了**继续微调（fine-tuning）**。

### 数据处理

```bash
cd task6_translation
python3 prepare_data.py --train-size 20000 --valid-size 1000
```

脚本读取 parquet，过滤过短/过长和非中文样本，随机抽样后保存为 `data/train.tsv`、`data/valid.tsv`（`源句\t目标句`）。

### 模型训练

微调脚本使用 PyTorch 原生训练循环：

| 超参数 | 取值 |
| --- | --- |
| batch size | 16 |
| learning rate | 3e-5 |
| optimizer | AdamW（weight decay 0.01） |
| scheduler | 线性 warmup + 线性衰减（warmup 5%） |
| epochs | 1 |
| max length | 源/目标均 128 |
| gradient clip | 1.0 |
| 设备 | CPU（8 线程） |

```bash
python3 -u train.py \
  --train data/train.tsv --valid data/valid.tsv \
  --epochs 1 --batch-size 16 --lr 3e-5 \
  --output models/opus-mt-zh-en-ft 2>&1 | tee output/train.log
```

训练日志见 [`task6_translation/output/train.log`](task6_translation/output/train.log)，其中记录了每 50 步的平均损失、学习率以及验证集损失。本次训练在 CPU（8 线程）上完成，共 1250 步，耗时约 27 分钟。损失变化如下：

| step | train loss | 说明 |
| --- | --- | --- |
| 50 | 1.3931 | 微调初期 |
| 200 | 1.2125 | 快速下降 |
| 500 | 1.2669 | 波动下降 |
| 800 | 1.2534 | — |
| 1000 | 1.2409 | 验证集 loss = 1.2134 |
| 1250 | 1.2915 | 训练结束 |

### 翻译

```bash
# 微调模型
python3 translate.py --model models/opus-mt-zh-en-ft \
  --input data/hongloumeng_sample.txt --output output/translation_ft.txt

# 预训练模型（用于对比）
python3 translate.py --model models/opus-mt-zh-en \
  --input data/hongloumeng_sample.txt --output output/translation_base.txt
```

翻译时先把《红楼梦》电子文本中的半角标点归一化为中文标点，再按标点切块并合并短句；生成使用 beam search（beam=4）、`no_repeat_ngram_size=3`，避免重复生成。

### 翻译结果

> 原文（节选）：一日，炎夏永昼，士隐于书房闲坐，至手倦抛书，伏几少憩，不觉朦胧睡去。梦至一处，不辨是何地方。忽见那厢来了一僧一道，且行且谈。只听道人问道：“你携了这蠢物，意欲何往？”

> 预训练模型：One day, the summer and the day, when the Stewards sit idle in the study, and the hands are tired of throwing their books, and fall low, and go to sleep...

> 微调模型：One day, during the summer and the morning, the people will sit in the study, and their hands will be tired of throwing their books, and they will not sleep...

完整逐句对照见 [`output/translation_base.txt`](task6_translation/output/translation_base.txt) 与 [`output/translation_ft.txt`](task6_translation/output/translation_ft.txt)。

### 结果说明与局限

- OPUS-100 是通用现代汉语语料，而《红楼梦》是古白话，存在明显的**领域/时代差异**。微调过程中训练损失从约 1.39 降到约 1.24～1.29，第 1000 步验证集损失为 1.2134，但这并不能完全解决古文翻译问题。
- 译文中的专名（如“士隐”“绛珠草”“警幻仙子”）常被音译或误译；这是无术语表机器翻译的常见问题。本次实验没有人工构造《红楼梦》术语表，以保证方法是“通用模型 + 通用平行语料微调”。
- 评价以人工可读性为主，未计算 BLEU：测试文本没有标准译文，且只有一个段落，BLEU 的统计意义有限。
- 从对比看，微调模型与预训练模型的译文都属于**直译风格**，个别地方微调模型反而出现更明显的误译（如把“风流公案”译成 murderous crime）。这说明在领域差异极大的文本上，继续微调通用语料不一定带来正向收益。若用于正式翻译，建议：
  1. 加入《红楼梦》名家英译本做领域微调；
  2. 先做“文言→现代汉语”的预编辑，再用中英模型翻译；
  3. 对大模型译文进行译后编辑，并人工校对专名。

---

## 复现步骤

```bash
# 1. 安装依赖
python3 -m pip install --user --break-system-packages -r requirements.txt

# 2. 第五题
cd task5_summarization
python3 summarize.py

# 3. 第六题
cd ../task6_translation
python3 prepare_data.py --train-size 20000 --valid-size 1000
python3 train.py --epochs 1 --batch-size 16 --output models/opus-mt-zh-en-ft
python3 translate.py --model models/opus-mt-zh-en-ft --output output/translation_ft.txt
```

> 注：`models/opus-mt-zh-en` 为从 Hugging Face 下载的预训练权重；`data/opus100_train.parquet` 为 OPUS-100 训练集。若需重新下载，可使用 `huggingface-cli download` 或 `curl` 从镜像获取。
