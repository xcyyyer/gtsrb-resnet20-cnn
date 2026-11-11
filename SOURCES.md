# 来源与基准核对（2026-11-03）

1. ResNet: He et al., Deep Residual Learning for Image Recognition, CVPR 2016. https://arxiv.org/abs/1512.03385 。本项目对应4.2节6n+2、n=3、通道16/32/64、option-A，不是torchvision的ImageNet ResNet-18。
2. MobileNets: Howard et al., 2017. https://arxiv.org/abs/1704.04861 。借鉴深度可分离卷积原理，未直接使用MobileNet预训练权重。
3. SE: Hu et al., CVPR 2018. https://openaccess.thecvf.com/content_cvpr_2018/html/Hu_Squeeze-and-Excitation_Networks_CVPR_2018_paper.html 。本项目自行实现SE，缩减率8为本实验设置。
4. GTSRB: Stallkamp et al., The German Traffic Sign Recognition Benchmark: A multi-class classification competition, IJCNN 2011. 官方说明 https://benchmark.ini.rub.de/gtsrb_dataset.html 。最终数据托管 https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/ 。下载压缩包MD5核对来源：https://raw.githubusercontent.com/datasets-mila/datasets--gtsrb/master/md5sums 。PyTorch官方读取说明 https://docs.pytorch.org/vision/stable/_modules/torchvision/datasets/gtsrb.html 使用不同的Training_fixed包，故本项目自行读取Final版本。
5. 备用镜像：https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign 。镜像不同打包，程序记录来源并检查图片数量、标签、序列分组。
6. 外部性能背景：ReADLBench/dnn-benchmark，关联2025年论文 Benchmark Suite for Resilience Assessment of Deep Learning Models, DOI 10.1109/TCAD.2025.3578297。 https://github.com/ReADLBench/dnn-benchmark 。README的GTSRB表列出ResNet20 Top-1 94.3%。但其预处理为50×50且包含翻转等增强，README评估图片数12640与标准12630不一致，训练划分/预算尚需进一步核对。因此94.3%只作为有出处的背景参考，不能直接称为本项目的严格同协议达标线，也不是教师规定分数或实测结果。

第一阶段目标：跑通真实数据训练、确认无泄漏和稳定收敛。验证集95%可作为项目自行设定的调优观察目标，不是教师基准，不保证达到。正式“基准达标”结论须在明确可比协议或教师指定阈值后判断。

代码为本项目实现；使用PyTorch、NumPy、Pillow、Matplotlib。报告应披露使用生成式工具辅助实现，并由本人理解、运行与核对。结构改进属于课程实验，不声称首次提出SE或深度可分离卷积。
