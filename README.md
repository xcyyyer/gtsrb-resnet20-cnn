# ResNet20交通标志识别与结构消融

本项目已真实完成4种结构×种子42/10/20×80轮训练，以及12个最佳权重在官方测试集12,630张图像上的clean/dark/blur共36次评估。

## 实验结构

- baseline：CIFAR风格ResNet20，16/32/64通道，三个阶段各三个残差块，Option A捷径，43类。
- ds：九个残差块的两个3×3卷积均分解为depthwise3×3+pointwise1×1，中间不额外加入BN或激活；首层不变。
- se：残差第二个BN之后、相加之前加入SE，缩减比8。
- ds_se：组合上述两种改动。

与实验三FCN/U-Net裂缝像素分割不同，本项目是残差网络对裁剪交通标志的图像级分类。

## 数据

公开镜像：https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign

固定划分保存在study/splits。训练31,379，验证7,830，官方测试12,630。训练/验证按轨迹组划分且无组交集，划分种子2026；训练种子42/10/20。不要为复现重新做随机图像划分。

原始图像未打包。训练需要完整的Train/与Test/图像和Train.csv、Test.csv；仅复现测试只需Test/图片和固定test.csv清单。

## 环境与安装

训练环境的完整记录位于study/runs各组的environment.json和pip_freeze.txt。训练采用Python3.13.16、PyTorch2.11.0+cu130、Tesla T4；最后一组第39轮后由Colab迁移Kaggle。此次最终推理采用Python3.12.14、PyTorch2.15.0+cpu、numpy2.3.5、batch256、workers0。跨硬件和版本不保证逐位一致。

```bash
python -m pip install -r requirements.txt
python data.py download --root data --source kaggle
```

Kaggle可直接Add Input挂载上述原始图片数据集，使用其实际根目录作为--data。完整训练结果已包含在study/runs；无需为查看结果重新训练。

## 查看与复现核验

```bash
python audit_results.py --study study --out validation_analysis
python evaluate_final.py --study study --data data --out test_evaluation --device cpu --workers 0
```

测试程序只加载已由验证集选定的best.pt，不训练、不重新选择种子。clean为原始输入，dark为RGB双线性缩放至32×32后亮度乘0.5，blur为缩放后GaussianBlur(radius=1)。归一化(x/255−0.5)/0.5。

随附test_evaluation已有完整结果，重新运行会验证并复用这些结果。若要重新执行全部推理，请使用新的--out，例如test_evaluation_rerun。所有数字在CSV/JSON中保留全精度，报告与图表显示三位小数。三种子标准差为样本标准差ddof=1。

## 从头复现训练

```bash
python train_all.py --data data --splits study/splits --out reproduction_runs --device cuda
```

新训练使用独立输出目录，固定每组80轮、batch128、SGD lr0.05、momentum0.9、weight_decay0.0005、余弦退火。中断后同命令将读取last.pt继续；跨平台迁移仅允许路径变化，其他训练参数保持不变。CPU从头训练可选--device cpu。

单组训练：

```bash
python run.py train --root data --splits study/splits --out reproduction_runs/baseline_s42 --variant baseline --seed 42 --epochs 80 --batch-size 128 --lr 0.05 --size 32 --device cuda
```

## 应用演示

```bash
python infer.py --checkpoint study/runs/ds_s42/best.pt --image data/Test/00000.png --device cpu
```

真实运行的示例输出保存在application_demo.json。输入须为裁剪后的交通标志；程序不是整图目标检测器，也没有未知类拒识。softmax分数不是经过校准的正确概率。

## 输出文件

- GTSRB_大作业报告.docx：按实验模板整理的六部分正文，保留原24张实验图与全部12组训练曲线，追加2张学习率对比图。
- build_report.py与report_template.docx：模板版报告生成程序及版式来源；依赖固定实验JSON，可用python build_report.py重新生成报告。
- validation_analysis/：重建的12组验证结果、三种子统计、核验结果及16套PNG/PDF图。
- test_evaluation/：36次评估的指标、逐图预测、逐类指标、混淆矩阵；12组条件汇总；8套PNG/PDF图。
- study/：固定清单与原始训练日志、best.pt、last.pt、配置、环境、迁移记录。
- GTSRB_Final_Evaluation.ipynb：Kaggle复现笔记本，内置评估源码；要求挂载最终80轮备份，不能使用39轮旧迁移包。
- verification.json：真实执行的核验范围。

原迁移包根目录的results.json只含8组且含旧值；本项目不使用那份过期汇总，以validation_analysis与test_evaluation为准。

## 学习率补充实验

在主实验测试结果已查看后，追加验证集敏感性分析：固定ds结构、种子42、相同训练/验证清单、每组80轮，仅改变初始学习率0.01/0.05/0.1。三个候选在同一Kaggle环境独立训练；0.05也重新运行，不与原研究混用。补充实验环境为Python3.13.16、PyTorch2.11.0+cu130、NumPy2.1.3、Tesla T4，完整设置见lr_supplement/GTSRB_LR_Sensitivity/protocol.json。

| 初始学习率 | 最佳轮次 | 验证Accuracy/% | 同轮Macro-F1/% |
|---|---:|---:|---:|
| 0.01 | 53 | 98.238 | 97.579 |
| 0.05 | 66 | 99.285 | 98.725 |
| 0.1 | 70 | 99.527 | 99.336 |

本次按验证Accuracy选择0.1；这是种子42下三个候选的描述性结果，不是跨种子稳定最优值，也不替换主实验测试结果。本补充实验没有运行测试集。

查看现有结果或核验，不必重新训练：

```bash
python audit_lr_supplement.py
python lr_supplement/supplement_source/lr_sensitivity.py --root data --out lr_supplement/GTSRB_LR_Sensitivity --summarize-only
```

第二条只读取已保存结果并重绘图表，不读取图片；`--root data`是命令行必需参数，在汇总模式不访问。三个best.pt、last.pt、240轮完整日志、环境、固定清单、统计和两套PNG/PDF均在lr_supplement中。audit_verification.json记录本地实际核验范围；核验不代替重新逐图计算验证集指标。

从头复现时，使用GTSRB_LR_Sensitivity.ipynb，挂载原GTSRB数据，不挂载补充结果备份。若续训，则同时挂载补充备份，Notebook支持ZIP或Kaggle自动解压目录。先确认恢复轮数，再运行训练格；完成组跳过，未完成组从自己的last.pt恢复。关闭会话前运行末尾汇总并下载最新GTSRB_LR_results.zip，临时工作目录不能代替离线备份。

也可以在GPU环境使用独立目录从头训练：

```bash
python lr_supplement/supplement_source/lr_sensitivity.py --root data --out lr_reproduction --device cuda
```

续训时不得修改已有protocol.json绕过环境检查。若环境变化，应先评估差异或在新目录独立复现，不能把不同条件的训练拼成同一组。现有完整结果无需再次训练。修订后的Notebook只改进已解压备份恢复；原训练源码保持冻结，哈希核验通过。

## 仓库发布

当前交付为可上传的完整工程包，尚未创建线上GitHub/Gitee仓库。新建仓库后可将本目录作为仓库根目录上传。实际仓库URL需补入报告，不能把数据集链接当作代码仓库链接。

## 参考

- He等 Deep Residual Learning for Image Recognition https://arxiv.org/abs/1512.03385
- Howard等 MobileNets https://arxiv.org/abs/1704.04861
- Hu等 Squeeze-and-Excitation Networks https://arxiv.org/abs/1709.01507

合成调暗与模糊仅为压力测试，不能宣称已验证真实夜间或真实道路部署；MACs仅计卷积与全连接，不等于完整FLOPs或实际加速倍数。
