"""Rewrite the verified report using the supplied laboratory template.

All measured numbers come from frozen validation/test JSONs. The template's
cover, section geometry, paragraph styles, headers and footers are retained.
"""
from pathlib import Path
from copy import deepcopy
import json
from docx import Document
from docx.shared import Cm, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_CELL_VERTICAL_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

ROOT = Path(__file__).resolve().parent
VAL = ROOT / 'validation_analysis'
TEST = ROOT / 'test_evaluation'
LR = ROOT / 'lr_supplement/GTSRB_LR_Sensitivity'
lr_result = json.loads((LR/'validation_summary.json').read_text())
assert lr_result['complete'] and json.loads((LR/'audit_verification.json').read_text())['status'] == 'passed'
lr_rows = lr_result['per_lr']
lr_by = {x['initial_lr']: x for x in lr_rows}
lr_acc_delta = lr_by[0.1]['best_val_accuracy_pct'] - lr_by[0.05]['best_val_accuracy_pct']
lr_f1_delta = lr_by[0.1]['val_macro_f1_at_best_accuracy_pct'] - lr_by[0.05]['val_macro_f1_at_best_accuracy_pct']
validation = json.loads((VAL/'validation_results.json').read_text())
tests = json.loads((TEST/'test_results.json').read_text())
assert json.loads((TEST/'COMPLETE.json').read_text())['completed_evaluations'] == 36
V = ['baseline', 'ds', 'se', 'ds_se']
SEEDS = [42, 10, 20]
N = {'baseline':'baseline', 'ds':'ds', 'se':'se', 'ds_se':'ds_se'}
va = {r['variant']:r for r in validation['three_seeds']}
te = {(r['variant'],r['condition']):r for r in tests['three_seeds']}
def vf(v,k='accuracy'):
    return f"{va[v]['val_'+k+'_mean_pct']:.3f}±{va[v]['val_'+k+'_sample_std_pp']:.3f}"
def tf(v,c='clean',k='accuracy'):
    return f"{te[v,c][k+'_mean_pct']:.3f}±{te[v,c][k+'_sample_std_pp']:.3f}"
pr=100*(1-va['ds']['parameters']/va['baseline']['parameters'])
mr=100*(1-va['ds']['conv_linear_macs']/va['baseline']['conv_linear_macs'])
delta=te['baseline','clean']['accuracy_mean_pct']-te['ds','clean']['accuracy_mean_pct']

doc=Document(ROOT/'report_template.docx')
grading_table=deepcopy(doc.tables[0]._tbl)
orig=list(doc.paragraphs)
refs={name:deepcopy(orig[i]._p) for name,i in
      [('title',16),('h1',17),('h2',28),('body',29),('caption',82)]}
body=doc.element.body
start=body.index(orig[16]._p)
for child in list(body)[start:]:
    if child.tag != qn('w:sectPr'): body.remove(child)

# Preserve the cover verbatim, including its fonts, underlines and spacing.

# User requested the template, not a new design: keep its Songti/TNR body,
# Lishu cover, 18/16 pt heading ladder, caption style and page dimensions.
for name in ['Normal','Heading 1','Heading 2','Heading 3','Caption']:
    doc.styles[name].font.color.rgb=RGBColor(0,0,0)
doc.core_properties.title='基于ResNet20的交通标志识别及结构改造对比实验'
doc.core_properties.subject='神经网络与深度学习大作业报告'
doc.core_properties.author='徐淳彦'

def add(text='',kind='body',page=False):
    from docx.text.paragraph import Paragraph
    el=deepcopy(refs[kind])
    for child in list(el):
        if child.tag != qn('w:pPr'):el.remove(child)
    pp=el.get_or_add_pPr()
    for tag in ['numPr','sectPr']:
        for child in list(pp.findall(qn('w:'+tag))):pp.remove(child)
    body.insert(len(body)-1,el)
    p=Paragraph(el,doc._body)
    p.add_run(text)
    p.paragraph_format.widow_control=True
    p.paragraph_format.keep_together=False
    if page:p.paragraph_format.page_break_before=True
    if kind in ['h1','h2']:p.paragraph_format.keep_with_next=True
    else:p.paragraph_format.keep_with_next=False
    return p
def h1(num,text,page=False):return add(f'{num}、{text}：','h1',page)
def h2(num,text,page=False):return add(f'{num}. {text}','h2',page)
def p(text):return add(text)
def code(lines):
    for line in lines.splitlines():
        a=add(line)
        a.paragraph_format.space_before=Pt(0)
        a.paragraph_format.space_after=Pt(0)
        a.paragraph_format.line_spacing=1
        snap=OxmlElement('w:snapToGrid');snap.set(qn('w:val'),'0');a._p.get_or_add_pPr().append(snap)
        for r in a.runs:r.font.size=Pt(10.5)
def equation(text):
    a=add();a.alignment=WD_ALIGN_PARAGRAPH.CENTER
    snap=OxmlElement('w:snapToGrid');snap.set(qn('w:val'),'0');a._p.get_or_add_pPr().append(snap)
    def run(s):
        r=OxmlElement('m:r');t=OxmlElement('m:t');t.text=s;r.append(t);return r
    def script(base,sub=None,sup=None):
        tag='m:sSub' if sub is not None else 'm:sSup'
        el=OxmlElement(tag);e=OxmlElement('m:e');e.append(run(base));el.append(e)
        part=OxmlElement('m:sub' if sub is not None else 'm:sup');part.append(run(sub if sub is not None else sup));el.append(part)
        return el
    forms={
      'Pstandard = k² × Ci × Co':[script('P',sub='standard'),' = ',script('k',sup='2'),' × ',script('C',sub='i'),' × ',script('C',sub='o')],
      'PDS = k² × Ci + Ci × Co':[script('P',sub='DS'),' = ',script('k',sup='2'),' × ',script('C',sub='i'),' + ',script('C',sub='i'),' × ',script('C',sub='o')],
      'L = −ln(exp(zy) / Σⱼ exp(zj))':['L = −ln(exp(',script('z',sub='y'),') / (',script('Σ',sub='j'),' exp(',script('z',sub='j'),')))'],
      'Accuracy = Ncorrect / Ntotal':['Accuracy = ',script('N',sub='correct'),' / ',script('N',sub='total')],
      'F1c = 2 × Precisionc × Recallc / (Precisionc + Recallc)':[script('F1',sub='c'),' = 2 × ',script('Precision',sub='c'),' × ',script('Recall',sub='c'),' / (',script('Precision',sub='c'),' + ',script('Recall',sub='c'),')'],
      'Macro-F1 = (Σc F1c) / 43':['Macro-F1 = (',script('Σ',sub='c'),' ',script('F1',sub='c'),') / 43'],
    }
    m=OxmlElement('m:oMath')
    for part in forms.get(text,[text]):m.append(run(part) if isinstance(part,str) else part)
    a._p.append(m)

fig_index=0
def fig(path,caption,width=15.3,maxheight=None):
    global fig_index
    from PIL import Image
    path=Path(path)
    with Image.open(path) as im:w,h=im.size
    if maxheight is not None:width=min(width,maxheight*w/h)
    a=add();a.alignment=WD_ALIGN_PARAGRAPH.CENTER
    a.paragraph_format.space_before=Pt(3)
    a.paragraph_format.space_after=Pt(0)
    a.paragraph_format.keep_with_next=True
    snap=OxmlElement('w:snapToGrid');snap.set(qn('w:val'),'0');a._p.get_or_add_pPr().append(snap)
    a.add_run().add_picture(str(path),width=Cm(width))
    fig_index+=1
    cap=add(f'图 {fig_index} {caption}','caption')
    cap.alignment=WD_ALIGN_PARAGRAPH.CENTER
    cap.paragraph_format.space_before=Pt(2)
    cap.paragraph_format.space_after=Pt(6)
    snap=OxmlElement('w:snapToGrid');snap.set(qn('w:val'),'0');cap._p.get_or_add_pPr().append(snap)
    return cap

def table(headers,rows,widths):
    t=doc.add_table(rows=1,cols=len(headers));t.alignment=WD_TABLE_ALIGNMENT.CENTER;t.autofit=False
    t.style='Table Grid'
    for i,w in enumerate(widths):t.columns[i].width=Cm(w)
    for i,x in enumerate(headers):t.rows[0].cells[i].text=str(x)
    for row in rows:
        cells=t.add_row().cells
        for i,x in enumerate(row):cells[i].text=str(x)
    borders=OxmlElement('w:tblBorders')
    for side in ['top','left','bottom','right','insideH','insideV']:
        e=OxmlElement('w:'+side);e.set(qn('w:val'),'single');e.set(qn('w:sz'),'4');e.set(qn('w:color'),'777777');borders.append(e)
    t._tbl.tblPr.append(borders)
    for ri,row in enumerate(t.rows):
        prr=row._tr.get_or_add_trPr();prr.append(OxmlElement('w:cantSplit'))
        if ri==0:prr.append(OxmlElement('w:tblHeader'))
        for i,c in enumerate(row.cells):
            c.width=Cm(widths[i]);c.vertical_alignment=WD_CELL_VERTICAL_ALIGNMENT.CENTER
            mar=OxmlElement('w:tcMar')
            for side in ['top','bottom','left','right']:
                e=OxmlElement('w:'+side);e.set(qn('w:w'),'65');e.set(qn('w:type'),'dxa');mar.append(e)
            c._tc.get_or_add_tcPr().append(mar)
            for para in c.paragraphs:
                para.alignment=WD_ALIGN_PARAGRAPH.CENTER
                para.paragraph_format.space_before=Pt(2)
                para.paragraph_format.space_after=Pt(2)
                para.paragraph_format.line_spacing=1
                snap=OxmlElement('w:snapToGrid');snap.set(qn('w:val'),'0');para._p.get_or_add_pPr().append(snap)
                spacing=para._p.get_or_add_pPr().find(qn('w:spacing'))
                spacing.set(qn('w:beforeLines'),'0');spacing.set(qn('w:afterLines'),'0')
                for run in para.runs:
                    run.font.name='Times New Roman';run._element.get_or_add_rPr().rFonts.set(qn('w:eastAsia'),'宋体')
                    run.font.size=Pt(10.5);run.bold=ri==0
    return t

title=add('实验：ResNet20交通标志识别','title')
for r in title.runs:
    r.font.name='宋体';r._element.get_or_add_rPr().rFonts.set(qn('w:eastAsia'),'宋体');r.font.size=Pt(22);r.bold=True
h1('一','实验目的')
p('本实验采用ResNet20对交通标志图片进行分类，希望在实现经典CNN的同时，进一步了解卷积、残差连接、BN和池化的作用。之后尝试深度可分离卷积和SE注意力，通过三个种子的训练结果比较参数量、MACs和精度，看看修改是否有效。')
p('实验三主要使用U-Net等网络进行裂缝分割，输出逐像素掩码。本实验改为残差网络，输出图片所属的交通标志类别，任务和主干均不同。')
h1('二','实验环境')
p('训练平台：Google Colab、Kaggle Notebook；训练设备：NVIDIA Tesla T4。')
p('训练环境：Python 3.13.16，PyTorch 2.11.0+cu130，NumPy 2.4.6。')
p('最终测试环境：Python 3.12.14，PyTorch 2.15.0+cpu，NumPy 2.3.5，使用CPU进行推理。')
p('主要库：PyTorch、NumPy、Pillow、Matplotlib；报告整理使用python-docx。')
p('本地电脑用于文件准备和备份，主实验最后一组从Colab迁移到Kaggle续训。')
p('补充学习率实验在Kaggle的Tesla T4上运行，Python 3.13.16、PyTorch 2.11.0+cu130、NumPy 2.1.3。三组补充实验使用同一环境记录，与主实验分开统计。')

h1('三','实验原理',page=True)
h2(1,'CNN与ResNet20的基本结构')
p('CNN通过卷积提取局部特征，同一个卷积核在不同位置重复使用。前面提取的特征经过后续各层继续组合，最后用于类别判断。')
p('本实验采用面向小尺寸图片的ResNet20，输入为32×32的RGB图片。三个阶段各有三个残差块，每块含两次卷积，最后通过全局平均池化和全连接层输出43个类别分数。')
table(['模块','主要设置','输出尺寸'],[
 ['输入','RGB图片','3×32×32'],['首层','3×3卷积，3→16，BN，ReLU','16×32×32'],
 ['阶段1','3个残差块，通道数16','16×32×32'],['阶段2','3个残差块，通道数32','32×16×16'],
 ['阶段3','3个残差块，通道数64','64×8×8'],['分类层','全局平均池化，64→43','43']], [2.1,9.1,4.1])
h2(2,'残差连接的作用')
p('残差块把输入保留到后面，与卷积分支的结果相加：')
equation('y = ReLU(F(x) + S(x))')
p('F(x)表示卷积分支，S(x)表示捷径分支。尺寸不变时直接保留输入，阶段切换时通过步进采样与通道补零匹配形状，不额外加入投影卷积。')
p('我理解残差连接是让网络在输入基础上学习需要修正的部分，同时给梯度提供较短的传播路径。最终效果仍需结合实验判断。')

h2(3,'将普通卷积改为深度可分离卷积',page=True)
p('普通卷积会同时处理空间位置和通道之间的信息。深度可分离卷积则把这一步拆开，先对每个通道分别做3×3卷积，再使用1×1卷积把不同通道的信息组合起来。')
p('设输入通道数为Ci，输出通道数为Co，卷积核边长为k，不考虑偏置时，两种卷积的参数量分别为：')
equation('Pstandard = k² × Ci × Co')
equation('PDS = k² × Ci + Ci × Co')
p('本实验将九个残差块中的两处3×3卷积都进行替换，首层卷积保持不变。为了让比较尽量集中在卷积本身，逐通道卷积和逐点卷积之间没有额外加入BN或激活函数。')
p('在代码中，groups=ci表示每个输入通道独立进行卷积，后面的1×1卷积再完成通道混合。实际实现如下，内容对应models.py中的Separable：')
code('nn.Conv2d(ci, ci, 3, stride, 1, groups=ci, bias=False)\nnn.Conv2d(ci, co, 1, bias=False)')
h2(4,'加入SE通道注意力')
p('SE注意力先通过全局平均池化得到每个通道的整体信息，再经过降维、ReLU、升维和Sigmoid，得到各通道对应的权重。最后将权重乘回原来的特征图，让不同通道具有不同的响应大小。')
p('本实验的缩减比设置为8，SE放在残差分支第二个BN之后，与捷径分支相加之前。这里的想法是让网络自己学习哪些通道更有用，而不是手动指定权重。')
p('注意力模块本身也会增加参数，并且学到的权重不一定有利于所有类别，因此不能只根据原理就判断它一定能提高准确率。后面会把se与baseline、ds_se与ds分别进行比较。')

h2(5,'损失函数与训练方式',page=True)
p('模型输出43个原始类别分数，即logits，训练时直接传给CrossEntropyLoss，不提前重复使用softmax。')
equation('L = −ln(exp(zy) / Σⱼ exp(zj))')
p('zy是真实类别的分数，PyTorch自动求导后由SGD更新参数。这次重点是结构改造，反向传播交给框架。BN归一化中间特征，ReLU提供非线性，池化压缩空间信息，最后由全连接层分类。')
h2(6,'评价指标与统计方法')
p('Accuracy表示预测正确的图片占总图片的比例。由于各类图片数量不同，本实验还计算Macro-F1，先分别计算每一类的F1，再对43个类别取平均，使图片较少的类别也能得到相同的统计权重。')
equation('Accuracy = Ncorrect / Ntotal')
equation('F1c = 2 × Precisionc × Recallc / (Precisionc + Recallc)')
equation('Macro-F1 = (Σc F1c) / 43')
p('每个模型先根据验证准确率选择最佳权重，再对三个种子的结果求均值和样本标准差，标准差使用ddof=1。表格中的“±”表示三次运行的波动，不是置信区间，也没有据此进行显著性检验。')
p('计算量统计采用卷积层和全连接层的MACs，一次乘加计为1 MAC。BN、激活、池化、残差相加和SE逐元素相乘不在该统计范围内，所以MACs不能直接当作完整FLOPs，也不能直接换算为实际推理速度。')

h1('四','实验内容与步骤',page=True)
h2(1,'下载数据并固定划分')
p('本实验数据集来源于Kaggle上的GTSRB公开镜像，共43类交通标志。训练来源有39,209张，官方测试有12,630张，图片与类别通过Train.csv和Test.csv对应。')
p('读取路径与类别后，按照类别分层，以拍摄轨迹为组划分，避免同一标志的连续图片分到两边。划分种子为2026，训练集31,379张、验证集7,830张，两边轨迹组交集为0。')
table(['数据划分','图片数量','用途'],[['训练集','31,379','更新模型参数'],['验证集','7,830','选择最佳训练轮次'],['官方测试集','12,630','固定模型后的最终评价']], [3.4,3.3,8.6])
p('划分种子2026决定图片分配，训练种子42、10、20用于重复训练。四种结构使用同一份划分，测试集不参与训练、权重选择或调参。')
h2(2,'图片预处理')
p('读取图片后先转为RGB，再通过双线性插值缩放为32×32。训练时增加±10°旋转，以及0.8～1.2倍的亮度和对比度变化；验证和正常测试时不进行这些随机增强。')
p('交通标志可能包含方向箭头和数字，翻转后可能改变含义，因此没有加入水平或垂直翻转。像素先除以255，再按(x−0.5)/0.5进行归一化。')
p('缩小图片可以降低开销，也可能损失细节，后面通过模糊测试进一步观察其影响。')

h2(3,'搭建基础网络并设置四组结构',page=True)
p('首先用普通卷积实现baseline，再通过variant参数选择是否使用深度可分离卷积和SE。四组保留相同的残差阶段、输入和输出，只改变指定组件，方便后续比较。')
table(['结构名称','残差块内卷积','SE注意力'],[['baseline','普通3×3卷积','无'],['ds','深度可分离卷积','无'],['se','普通3×3卷积','有'],['ds_se','深度可分离卷积','有']], [3.5,7.1,4.7])
p('下面对应models.py中的Block.forward。不使用SE时attention为Identity，因此四组仍沿用同一套前向传播流程。')
code('z = F.relu(self.bn1(self.conv1(x)))\nz = self.attention(self.bn2(self.conv2(z)))\nreturn F.relu(z + self.shortcut(x))')
h2(4,'统一设置训练参数')
table(['参数','本实验设置'],[['训练轮数','每组80轮'],['训练种子','42、10、20'],['batch_size','128'],['优化器','SGD，momentum=0.9'],['初始学习率','0.05'],['weight_decay','0.0005'],['学习率调度','CosineAnnealingLR，T_max=80']], [5,10.3])
p('四种结构共12组，均为随机初始化训练，不使用ImageNet预训练权重。原结构对比统一采用0.05，后面追加的学习率实验单独进行，不改动原12组结果。')

h2(5,'记录训练过程并处理平台中断',page=True)
p('每轮将损失和验证指标写入history.csv，验证Accuracy严格提高时保存best.pt，并列保留最早一次。last.pt保存最近训练状态，用于中断后继续运行。')
p('遇到Colab GPU限额时，前11组已完成80轮，ds_se_s20完成39轮。我把代码、划分、日志和检查点迁移到Kaggle。上传曾遇到许可证错误，改为other后创建成功；下载数据又出现域名解析错误，改为挂载同版本数据集。')
p('恢复时仅调整路径，保留结构、种子和训练设置，最后一组继续至80轮。12组核验结果如下，最佳轮次以验证Accuracy为准。')
table(['结构','种子','完成轮数','最佳轮次','验证Acc/%'],[
 [r['variant'],r['seed'],r['epochs'],r['best_epoch'],f"{r['best_val_accuracy_pct']:.3f}"]
 for r in validation['per_run']], [3.1,2.2,2.6,2.6,4.8])
p('这样保留的不只是最终准确率，还能找到对应的训练轮次和权重。需要注意，日志中的轮次从1开始，last.pt中的epoch字段从0开始，两者不能直接按同一个起点理解。')

h2(6,'检查权重并进行最终测试',page=True)
p('先核对12组日志与最佳权重，确认80轮记录、最佳轮次和验证指标一致。旧汇总只有8组且有历史值，因此从12份history.csv重新计算。')
p('同版本镜像的12,630个测试标签与固定清单逐项一致。每个最佳权重分别进行正常、调暗、模糊测试，共36次，保存逐图预测、混淆矩阵和逐类指标。再从454,680条预测重新统计Accuracy、Macro-F1与Macro-Recall，结果一致。')
h2(7,'交通标志单图分类应用')
p('infer.py输入已裁剪的交通标志图片，输出前五个类别编号与softmax分数。实际用ds种子42权重测试Test/00000.png，真实类别16，最高分预测也是16，分数约0.998。')
p('该样本被正确分类，但不能代表所有图片。这里不负责整幅道路图的目标定位，softmax分数也未经校准，不能直接当作可靠的正确概率。')
h2(8,'工程文件与运行方法')
p('完整工程包含源码、配置、固定划分、12组权重与日志、测试输出和图表。原始图片按README另行下载，不放入工程压缩包。')
p('代码仓库链接：尚未发布，提交前需补充实际可访问的GitHub或Gitee地址。运行说明和完整源码已随工程包提供。')

h2(9,'复现运行说明',page=True)
p('解压工程后，先在项目目录安装依赖：')
code('python -m pip install -r requirements.txt')
p('将数据放入data目录，目录中应包含Train.csv、Test.csv、Train文件夹和Test文件夹。固定划分直接使用study/splits，不需要重新打乱。')
p('重新训练baseline、种子42的命令如下。输出单独写入reproduction_runs，避免覆盖已经完成的实验。其余结构修改variant，种子修改seed。')
code('python run.py train --root data --splits study/splits\n  --out reproduction_runs/baseline_s42 --variant baseline\n  --seed 42 --epochs 80 --batch-size 128 --lr 0.05\n  --size 32 --device cuda')
p('上面分行是为了便于排版，实际执行时合成一行。已有实验继续训练时增加--resume；跨平台且仅修改路径时再增加--allow-relocation。四种结构与三个种子的批量训练入口为train_all.py。')
p('使用已经完成的权重进行测试：')
code('python evaluate_final.py --study study --data data\n  --out test_evaluation --device cpu --workers 0')
p('执行单图分类：')
code('python infer.py --checkpoint study/runs/ds_s42/best.pt\n  --image data/Test/00000.png --device cpu')
p('测试和单图分类都可以在CPU上进行，不需要重新训练。工程同时提供Kaggle测试Notebook用于复现，其整本Notebook尚未在Kaggle重新执行；本报告的最终指标来自已经实际完成的CPU测试脚本。')

h2(10,'追加学习率对照实验',page=True)
p('原来的结构对比完成后，我又增加了一组学习率实验，进一步观察模型调优的作用。固定ds结构和训练种子42，只把初始学习率设为0.01、0.05、0.1，三组分别从随机初始化开始，各训练80轮。')
p('三组使用同一训练集和验证集，batch_size为128，SGD的momentum为0.9、weight_decay为0.0005，余弦退火T_max为80，增强和结构也不变。初始学习率决定开始时的更新步长，之后按同一余弦规则逐渐减小。')
p('为统一比较环境，0.05也重新训练，不复用原ds种子42的结果。各组按验证Accuracy最高选择最佳轮次，并列保留最早一轮；候选学习率也按最佳验证Accuracy比较。')
p('本次是在原测试结果已经查看后追加的验证分析，不运行新测试，也不替换四结构测试结论。只用种子42，不能理解为三个种子下都稳定最优。')
p('中断时，0.01完成80轮、0.05完成7轮，0.1未开始。Kaggle自动解压备份后，原代码未发现ZIP。我把解压结果复制到工作目录，核对日志与last.pt，再跳过0.01，从0.05第8轮继续，最后完成0.1。')
p('这部分的入口是GTSRB_LR_Sensitivity.ipynb，按准备工具、恢复数据、训练、汇总的顺序运行。完成组跳过，未完成组从自己的last.pt继续。')
p('在项目根目录运行python audit_lr_supplement.py，可核对240轮日志、源码、划分、配置和best/last检查点，并完成CPU前向检查，不重新训练或评价测试集。')

h1('五','实验结果与分析',page=True)
h2(1,'训练损失与准确率变化')
p('首先把三个种子在相同轮次的指标进行汇总，观察整体收敛过程。图中实线表示三次运行的均值，阴影表示样本标准差。')
fig(VAL/'figures/three_seed_learning_curves.png','四种结构的三种子训练与验证曲线',maxheight=10)
p('从曲线可以看出，训练初期损失下降较快，准确率很快提高。验证曲线前期有波动，后面逐渐稳定，可能与学习率、数据增强和BN有关，但目前没有单独隔离这些因素。')
p('训练准确率接近100%只说明拟合较好，还需要看独立测试结果。12组单独的Loss、Accuracy和验证Macro-F1曲线放在本部分后面。')

h2(2,'验证结果与模型规模对比',page=True)
table(['模型','验证Acc/%','验证Macro-F1/%'],[[v,vf(v),vf(v,'macro_f1')] for v in V],[3,6.15,6.15])
p('表格中均为三个种子的均值±样本标准差，Macro-F1取最佳验证Accuracy所在轮次，而不是另找Macro-F1最高的一轮。')
fig(VAL/'figures/validation_mean_std.png','最佳验证权重的三种子指标对比',maxheight=5)
p('从验证结果看，四组准确率都较高，但加入SE和两种修改组合以后，没有超过baseline的均值。单独看某个种子会得出不同印象，因此这里保留三次结果及其波动，再结合独立测试集判断。')
h2(2,'验证结果与模型规模对比（续）',page=True)
table(['模型','参数量','卷积与全连接MACs'],[[v,f"{va[v]['parameters']:,}",f"{va[v]['conv_linear_macs']:,}"] for v in V],[3,5.6,6.7])
fig(VAL/'figures/model_complexity.png','相同输入尺寸下的参数量和MACs',maxheight=6.5)
p(f'运行后可以看到，ds参数量减少了{pr:.3f}%，MACs减少了{mr:.3f}%。验证准确率均值比baseline低0.021个百分点，说明轻量化确实减少了模型代价，但不能只根据种子42的一次结果就认为准确率一定提高。')
p('se和ds_se比各自未加注意力的版本增加了参数。虽然SE的池化和通道映射比较轻量，但额外计算也需要计入。深度可分离版本保留残差阶段，但每个普通卷积拆成了两个卷积，因此实际卷积算子数量会增加。')

h2(3,'官方测试集上的分类结果',page=True)
table(['模型','测试Acc/%','测试Macro-F1/%'],[[v,tf(v),tf(v,k='macro_f1')] for v in V],[3,6.15,6.15])
p(f'在正常测试条件下，baseline准确率为{tf("baseline")}%，ds为{tf("ds")}%。ds的均值比baseline低{delta:.3f}个百分点，因此这次改造的主要收益是减少参数和计算量，而不是提高总体准确率。')
p('se的正常测试Accuracy比baseline低0.050个百分点，但Macro-F1高0.161个百分点。结果说明不同指标关注的内容不同，不能简单把注意力的作用概括为全部提升或全部无效。')
fig(TEST/'figures/test_conditions_mean_std.png','正常 调暗 模糊条件下的三种子测试结果')
p('四组模型的测试权重都由验证集提前选定，测试结果只用于最终分析。三次运行的差异目前作为描述性结果，不把这些小幅变化写成已经证明的显著提升。')

h2(4,'调暗和模糊条件下的表现',page=True)
p('为了进一步观察场景变化，在同一测试清单上增加两种固定处理：dark在缩放后把亮度乘0.5，blur在缩放后使用radius=1的高斯模糊。除这一步处理外，图片顺序、类别和权重都保持一致。')
fig(TEST/'figures/synthetic_conditions_examples.png','固定前八个测试样本的正常 调暗和模糊输入')
table(['模型','调暗Acc/%','模糊Acc/%','调暗下降\n百分点','模糊下降\n百分点'],[[v,tf(v,'dark'),tf(v,'blur'),
 f"{te[v,'clean']['accuracy_mean_pct']-te[v,'dark']['accuracy_mean_pct']:.3f}",
 f"{te[v,'clean']['accuracy_mean_pct']-te[v,'blur']['accuracy_mean_pct']:.3f}"] for v in V],[2.2,3.9,3.9,2.65,2.65])
p('对比后可以发现，ds在模糊条件下下降1.248个百分点，baseline下降0.713个百分点。一个可能的原因是较小的模型在细节损失后更难保留足够的判别信息，但这个解释还需要更多实验验证。')
p('这些处理只是在原测试图片上模拟亮度和细节变化，不能直接当作真实夜间道路或摄像头运动模糊数据的测试结果。')

h2(5,'逐类指标与计算代价的进一步分析',page=True)
fig(TEST/'figures/clean_per_class_f1.png','正常测试条件下各类别F1的三种子均值')
p('总体准确率比较高时，仍然可能有部分小类别识别较差，因此我又对照了各类F1。Macro-F1把每类放在同等位置，比只看总正确数量更容易发现类别之间的差异。具体每类的样本量、precision、recall和F1已经保存在per_class.csv中。')
fig(VAL/'figures/compute_accuracy_tradeoff.png','MACs与最佳验证准确率之间的关系',width=11.2)
p('这张图使用的是验证指标，横轴是MACs，纵轴是最佳验证Accuracy的三种子统计。ds和ds_se位于计算量较小的一侧，但轻量化后的误差范围仍然需要与精度一起看，不能只按照参数少就判断模型更好。')

h2(6,'混淆矩阵分析',page=True)
p('下面固定使用种子42展示四种结构的正常测试混淆矩阵，避免看完测试结果以后再挑选表现较好的种子。每一行按真实类别的样本量归一化，主对角线表示预测正确。')
fig(TEST/'figures/clean_confusion_seed42.png','四种结构固定种子42的正常测试混淆矩阵',width=14.5,maxheight=16)
p('大部分结果集中在主对角线附近，说明总体分类效果较好，但仍然存在部分类别相互混淆。仅凭矩阵可以知道哪些类别容易分错，至于错误是否由数字、颜色或边缘信息导致，还需要回到具体图片中检查。')

for idx,v in enumerate(V,7):
    h2(idx,f'{v}的错误样例',page=True)
    p('这里同样固定使用种子42，选取正常测试中softmax分数较高的错误预测进行展示。每张图片上方保留真实类别、预测类别和对应分数。')
    fig(TEST/f'figures/errors_{v}_seed42.png',f'{v}种子42的正常测试错误样例',width=14.1,maxheight=14.5)
    if v=='baseline':
        p('可以看到，模型即使给出较高分数，最终预测仍然可能错误。因此实际应用时不能把softmax分数直接当成识别正确的保证，还需要结合类别范围和输入图片是否符合训练数据来判断。')
    elif v=='ds':
        p('ds的错误统计中，真实21预测为11有18张，真实3预测为5有18张，真实22预测为25和真实6预测为42各有11张，真实41预测为9有9张。这里列出的是类别编号，不把少量样例直接解释成某个组件的确定性缺陷。')
    elif v=='se':
        p('加入SE以后，正常测试Macro-F1略有提高，但仍然有高分错误。我的理解是，通道加权可以改变特征的利用方式，却不能保证每一类都同时改善；最后还是要把逐类指标和总体结果放在一起比较。')
    else:
        p('ds_se在模糊测试中比ds的Accuracy高0.174个百分点、Macro-F1高0.668个百分点，但正常测试结果没有超过ds。说明两种改动组合后会产生不同条件下的取舍，不能因为模块更多就认为一定更好。')

for v in V:
    h2(11 if v=='baseline' else {'ds':12,'se':13,'ds_se':14}[v],f'{v}的三组完整训练曲线',page=True)
    for seed in SEEDS:
        fig(VAL/f'figures/curves_{v}_s{seed}.png',f'{v} 种子{seed}的Loss Accuracy和验证Macro-F1',width=15.3)
    p('虚线标记验证Accuracy最佳轮次。ds_se种子20图中额外的点线标记第39轮后的平台迁移位置，后续训练沿用原检查点。')

h2(15,'不同初始学习率的验证结果',page=True)
p('三组补充实验都完成80轮后，我先比较各组最佳验证Accuracy，以及同一轮的Macro-F1和验证Loss。下面均为种子42的一次运行，不计算三种子的均值或标准差。')
table(['初始学习率','最佳轮次','验证Acc/%','同轮\nMacro-F1/%','同轮验证Loss'],[
 [format(r['initial_lr'],'g'),r['best_epoch'],f"{r['best_val_accuracy_pct']:.3f}",
  f"{r['val_macro_f1_at_best_accuracy_pct']:.3f}",f"{r['val_loss_at_best_accuracy']:.3f}"]
 for r in lr_rows],[2.1,2.1,3.6,3.9,3.6])
fig(LR/'figures/lr_validation_comparison.png','ds种子42在三种初始学习率下的最佳验证指标',maxheight=6.5)
p(f'运行后可以看到，本次初始学习率0.1的验证Accuracy为{lr_by[0.1]["best_val_accuracy_pct"]:.3f}%，对应Macro-F1为{lr_by[0.1]["val_macro_f1_at_best_accuracy_pct"]:.3f}%，在三个候选中最高。与0.05相比，两项指标分别提高{lr_acc_delta:.3f}和{lr_f1_delta:.3f}个百分点。')
p('因此，在本次ds、种子42和80轮预算下，按预设规则选择0.1。这个结论只适用于此次比较，不能推广成学习率越大越好，也不能当作测试集准确率的提升。')

h2(16,'学习率曲线与调优体会',page=True)
fig(LR/'figures/lr_learning_curves.png','三种初始学习率的完整80轮训练与验证曲线',maxheight=8.5)
p('三组训练Loss最终都较低、训练Accuracy接近100%，验证结果仍有差别。0.1前期波动较大，后期逐渐稳定并取得本次最高指标，所以不能只凭前几轮是否平稳判断设置优劣。')
p(f'0.01最佳验证Accuracy为{lr_by[0.01]["best_val_accuracy_pct"]:.3f}%，低于另两组，但训练Accuracy也接近100%。因此不能仅凭汇总把原因归为训练集未拟合好。三组都使用余弦退火，本实验也不能证明它优于其他调度。')
old_ds42=next(r for r in validation['per_run'] if r['variant']=='ds' and r['seed']==42)
p(f'三组均保留80轮，未因结果较差临时增加轮数。原0.05组验证Accuracy为{old_ds42["best_val_accuracy_pct"]:.3f}%，本次为{lr_by[0.05]["best_val_accuracy_pct"]:.3f}%；环境不同且固定种子不保证跨环境逐位一致，因此不把两次结果拼接统计。')

h1('六','实验心得体会',page=True)
p('通过本次实验，我更进一步地了解了CNN中不同组件的作用。这次需要把卷积、BN、激活函数、残差连接和分类层放在完整网络里考虑，而不只是看一个卷积核怎么计算。')
p('我在实现时也需要AI辅助，但程序没有报错不代表实验就完成了，还要检查划分、配置、权重和结果是否对应。旧汇总没有包含全部实验，也让我认识到写报告前必须重新核对数字。')
p('在结构改进时同样遇到了有想法但效果不尽人意的情况。ds减少了参数和MACs，测试精度也有下降；SE只在部分指标上改善。因此改进不能想当然，需要明确希望改善什么，再用实际结果比较。')
p('追加学习率实验后，我发现结构相同也会因为训练设置不同而得到不同结果。本次0.1在三个候选中表现最好，但只有一个种子，因此我会把它写成一次调优观察，而不是普遍适用的最优参数。')
p('Colab限额和平台迁移也让我重视备份，日志、划分和检查点都要保留，中断后才能继续。补充实验恢复时，还需要区分备份已经上传和进度真正复制到工作目录这两件事。')
p('目前只使用一个数据集、三个种子和合成扰动，尚不能代表真实夜间道路场景。后续希望增加独立数据和统一硬件延迟测试，进一步判断轻量化的实际价值。')
ref_heading=p('参考资料与数据来源：')
ref_heading.paragraph_format.page_break_before=True
ref_heading.paragraph_format.keep_with_next=True
for text in [
 '[1] GTSRB公开镜像：',
 'https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign',
 '[2] He等，Deep Residual Learning for Image Recognition：',
 'https://arxiv.org/abs/1512.03385',
 '[3] Howard等，MobileNets：',
 'https://arxiv.org/abs/1704.04861',
 '[4] Hu等，Squeeze-and-Excitation Networks：',
 'https://arxiv.org/abs/1709.01507']:
    a=p(text)
    a.paragraph_format.space_before=Pt(1)
    a.paragraph_format.line_spacing=1
    snap=OxmlElement('w:snapToGrid');snap.set(qn('w:val'),'0');a._p.get_or_add_pPr().append(snap)
    for r in a.runs:r.font.size=Pt(10.5)

# Retain the supplied teacher's marking fields. Use an inline table so its
# position follows the rewritten body rather than the old floating anchor.
for el in list(grading_table.tblPr.findall(qn('w:tblpPr'))):grading_table.tblPr.remove(el)
body.insert(len(body)-1,grading_table)

# Remove unused template illustrations, which would otherwise remain as stale
# image relationships even though their paragraphs have been deleted.
used={el.get(qn('r:embed')) for el in doc.element.iter() if el.get(qn('r:embed'))}
for relid,rel in list(doc.part.rels.items()):
    if rel.reltype.endswith('/image') and relid not in used:doc.part.drop_rel(relid)

out=ROOT/'GTSRB_大作业报告.docx'
doc.save(out)
print(json.dumps({'output':str(out),'figures':fig_index,'tables':len(doc.tables),
                  'template_body_font_pt':doc.styles['Normal'].font.size.pt,
                  'sections':len(doc.sections)},ensure_ascii=False))
