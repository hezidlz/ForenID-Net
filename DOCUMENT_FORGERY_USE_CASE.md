# 证件伪造检测与定位交付材料

- [Project conclusion PPT](证件伪造检测与定位_最终方案版.pptx)
- [Use case guide（Word）](证件伪造检测与定位_Use_case_guide.docx)
- [代码与结果包](document_forgery_use_case.zip)

解压代码包后，在 `document_forgery_use_case/README.md` 查看安装、模型调用、代码说明和指标复算方法。包内包含完整推理源码、依赖版本、最终方案配置、200 张测试分数与校验脚本，以及上述两份材料。

方案基于 TruFor，使用全图检测和两个原图裁块复检。代码包内保留原作者许可证与来源说明。模型权重由 `setup_runtime.py` 从官方来源下载并校验 SHA-256；原始数据集不随包分发。

此交付物的代码和结果以压缩包内说明为准，与仓库根目录已有模型分别组织。
