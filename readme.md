# 项目说明
The demand for explainability in complex machine learning (ML) models is ever more pressing, particularly within safety-critical domains like autonomous driving.
The Integrated Gradient (IG), a prominent attribution-based explainable artificial intelligence method, offers an effective solution for explaining and diagnosing ML models.
However, IG is primarily designed for  deep neural network models, hindering its broader applicability to various structured machine-learning models. Moreover, the issue of selecting a suitable single baseline point in IG methods still needs to be solved.
In response to these challenges, this paper introduces a model-agnostic explainable technique called the Expected Integral Discrete Gradient (EIDG). This approach extends the capabilities of IG to encompass a wide range of machine-learning models by leveraging numerical differentiation. It replaces the previous single baseline point scheme with a distributed multi-baseline method to reveal how varying
baselines affect the output. Our method is thoroughly evaluated on standard machine learning models, targeting scenarios in autonomous driving scenarios, thereby validating its effectiveness in explaining and diagnosing models. Our work will inspire and equip developers and users with the necessary tools to promote the adoption of attribution explanations across various machine-learning domains.

![EIDG](https://github.com/limeng-1234/EIDG/assets/76480875/770062c3-cb93-4984-a4c7-f72f3fd17fe5)

# 项目使用
## 前提条件
ubuntu 18.04 (windows 系统也能够支持)

安装conda

## 库安装

* 使用conda创建python=3.6虚拟环境  

* conda create -n your_env_name python=3.6

* conda activate your_env_name

* pip install -r requirements.txt

将该库下载到本地

## 示例
*  横向决策模型示例 notebook/LaneChange_example.ipynb
*  纵向决策模型示例 notebook/longitudinal_example.ipynb
*  EIDG计算效率分析 notebook/calculation_time-comparasion.ipynb
*  EIDG数值稳定性分析 notebook/Numerical_stability.ipynb



# 实车中实时生成归因值的显示
[放视频](https://github.com/limeng-1234/EIDG/assets/76480875/aeba45ee-2ab7-42fe-b5e8-1ff97977b3cb)
