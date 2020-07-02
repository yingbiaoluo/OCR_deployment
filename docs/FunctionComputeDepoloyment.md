# 函数计算部署步骤

## 概述

本部分介绍使用函数计算部署深度学习AI推理模型，包括使用fun工具安装第三方依赖、本地调试、一键部署、对接API网关。

## 一、准备工作

1. 开通阿里云服务  
[开通函数计算](https://www.aliyun.com/product/fc?spm=5176.10695662.h2v3icoap.32.4a7b1a25b79N5C)，按量付费，函数计算目前有2核3G的免费额度  
[开通文件存储服务NAS](https://www.aliyun.com/product/nas?spm=5176.cnfc.h2v3icoap.41.18f6224eri9P5I)，按量付费  
[开通专有网络VPC](https://www.aliyun.com/product/vpc?spm=5176.59209.h2v3icoap.68.124d1d7ev8EDiI)

2. 本地配置  
[安装Docker](https://www.docker.com)，函数计算本地调试依赖Docker  
[安装部署工具funcraft](https://github.com/aliyun/fun/blob/master/docs/usage/installation-zh.md)，目前最新版本为3.6.14

## 二、部署

### 2.1 配置

Fun 是一个用于支持 Serverless 应用部署的工具，能帮助您便捷地管理函数计算、API 网关、日志服务等资源。它通过一个资源配置文件（template.yml），协助您进行开发、构建、部署操作。

> 文件作用解释：  
> |文件|作用|
> |---|---|
> |.env||
> |Funfile||
> |.funignore||
> |template.yml|资源配置文件|
> |.fun||
 

### 2.2 安装第三方库

根据Funfile的定义：

* 将第三方库下载到```.fun/nas/xxxxxxxx-iwo76.cn-shanghai.nas.aliyuncs.com/OCR_test/python```目录下
* 本地 model 目录移到```.fun/nas/xxxxxxxx-iwo76.cn-shanghai.nas.aliyuncs.com/OCR_test/model```目录下

安装完成后,函数计算引用的代码包解压之后已经达到了 670 M, 远超过 50M 代码包限制, 解决方案是挂载NAS访问，幸运的是 FUN 工具一键解决了 nas 的配置和文件上传问题。

### 2.3 本地调试

### 参考链接🔗

[1] [基于函数计算+TensorFlow的Serverless AI推理](https://help.aliyun.com/document_detail/146724.html?spm=5176.cnfc.0.0.18f6224eri9P5I)  
[2] Github: [alibaba/funcraft](https://github.com/alibaba/funcraft)