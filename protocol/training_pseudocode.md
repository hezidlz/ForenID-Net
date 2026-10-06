# ForenID-Net 修订训练伪代码

## Algorithm 1：通用预训练

输入：经作者确认的通用篡改训练集与开发集、固定结构配置、随机种子集合。

1. 设定随机种子并记录数据清单、配置文件和 Noiseprint++ checkpoint 的 SHA-256。
2. 构建 ForenID-Net；加载并冻结 Noiseprint++，其余模块按预训练配置初始化。
3. 对每个 epoch：
   - 从训练清单读取图像、图像级标签及可用掩膜；
   - 对图像和掩膜执行同步几何变换，只对图像执行光照、压缩和噪声变换；
   - 前向得到区域 logits 与图像 logits；
   - 仅对 `has_mask=1` 的样本计算区域 focal 与 Dice 损失；所有样本计算图像 BCE；
   - 以 `0.7 L_focal + 0.7 L_dice + 1.0 L_bce` 反向传播并更新；
   - 在通用开发集计算阈值无关 Image AUC，以固定规则保存最佳 checkpoint。
4. 保存 checkpoint、优化器状态、配置、训练日志、种子和 SHA-256。

## Algorithm 2：FantasyID 组隔离微调

输入：Algorithm 1 的 checkpoint、FantasyID train/dev 清单、预先固定的最终重退化版本。

1. 对种子 `20260829`—`20260833` 分别执行：
   - 载入相同通用预训练 checkpoint；
   - 冻结 Noiseprint++ 与 RGB Stage 1—2；Stage 3—4、噪声适配器、F2—F4 门控、解码器和图像判别头可训练；
   - 使用 25 epochs、batch size 2、AdamW、初始学习率 `5e-5`、最小学习率 `1e-6`、权重衰减 `0.02`、1 epoch warmup 与余弦衰减；
   - 仅以开发集 Image AUC 保存该种子的最佳 checkpoint。
2. 对每个锁定 checkpoint，只在开发集计算图像阈值与区域阈值；F1 并列时取更高阈值。
3. 冻结模型、两个阈值和分析脚本；对内部测试、跨模板保留集和合法外部集各执行一次推理。
4. 保存逐图 `path/group_key/label/score/mask` 输出；按基础证件组进行 2,000 次 bootstrap，并在同样本比较中使用配对组级 bootstrap。

## Algorithm 3：单因素控制与压力测试

1. 融合控制仅替换 `fusion_mode`，统一 F2—F4 作用尺度、数据、轮数、增强预算、选择规则和阈值规则。
2. 训练因素控制只改变初始化或重退化开关；重退化最终版本不得根据修订测试结果再次切换。
3. 对锁定最终模型运行固定的 JPEG、模糊、降采样、高斯噪声、gamma、摩尔纹/打印扫描和噪声一致化强度网格。
4. 分别报告真实样本特异度、各攻击召回、pointing game、box IoU 与组级 95% 区间。
