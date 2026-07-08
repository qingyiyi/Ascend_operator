# Ascend Transformer Boost

🔥 [2025/09] Ascend Transformer Boost项目首次上线。

## 一、什么是ATB

### ATB介绍

Ascend Transformer Boost加速库（下文简称为ATB加速库）是一款高效、可靠的加速库，基于华为Ascend AI处理器，专门为Transformer模型的训练和推理而设计。具体的工作原理可以参考[ATB加速原理](docs/ATB加速原理.md)。 

### 软件架构

![架构图](docs/images/架构图.png)  
ATB的架构图如上图所示，其接口功能主要分成三部分：

- 提供经过优化的融合算子（Operation），用户可以根据需求使用对应的算子完成计算功能。
- 提供图算子机制，用户根据模型设计对应的图算子，使用加速库提供的原生算子和创建的自定义算子创建图算子，完成相应的计算。
- 提供插件（Plugin）机制，用户可以根据自己的需求创建自定义的算子。

### ATB仓介绍

ATB库的目录结构如下：

```
ascend-transformer-boost
├── 3rdparty            //第三方依赖库文件夹
├── build               //可存放构建生成的文件
├── ci                  //持续集成相关的配置文件
├── docs                //文档文件
├── example             //算子调用示例代码，包含可直接运行的Demo
├── include             //存放公共头文件
├── ops_configs         //存放算子输入输出数据规格约束文件
├── ops_customize       //存放自定义操作相关的文件
├── output              //编译输出文件夹
├── scripts             //脚本文件存放目录
├── src                 //主体源代码目录
│   ├── atb
│   ├── kernels
│   │   ├── configs     //支持的配置说明
│   │   ├── include     //存放各算子的头文件
│   │   ├── kernels     //单算子存放目录
│   │   ├── lcal        //通信算子存放目录
│   │   ├── mixkernels  //融合算子存放目录
│   │   ├── tbe_adapter //TBE 适配器相关的源代码
│   │   └── CMakeLists.txt
│   ├── ops
│   │   ├── ops_common
│   │   ├── ops_infer   //推理OP
│   │   └── ops_train   //训练OP
│   ├── torch_atb       //与PyTorch相关的atb库文件
│   └── CMakeLists.txt
├── tests               //测试代码
└── torch_atb
```

### 为什么选择ATB

- 对Transformer模型的高效加速：ATB加速库通过优化矩阵乘法等核心算子和注意力机制的实现方式，实现了对Transformer模型的高效加速。
- 高性能和效率：ATB加速库充分利用了Ascend AI处理器的硬件特性，如算力、存储带宽和内存带宽，通过硬件加速和数据重用等技术，进一步提升了性能和效率。
- 提供了底层基础的高性能算子以及高效的算子组合技术。
- 支持多种模型框架如PyTorch、MindSpore、Paddle。

## 二、环境构建

### 版本兼容性说明

ATB的API保证前后一年的ABI兼容能力，在不涉及新功能的情况下，调用者升级一年内的ATB版本，不会出现兼容问题。由于CANN出包目录调整，ATB 8.5版本以及主线分支必须匹配8.5或以上版本的toolkit包。

### 快速安装CANN软件

本节提供快速安装CANN软件的示例命令，更多安装步骤请参考[详细安装指南](#cann详细安装指南)。

#### 安装前准备

在线安装和离线安装时，需确保已具备Python环境及pip3，当前CANN支持Python3.7.x至3.11.4版本。
离线安装时，请单击[获取链接](https://www.hiascend.com/developer/download/community/result?module=cann)下载CANN软件包，并上传到安装环境任意路径。

#### 安装CANN

由于CANN出包目录调整，ATB 8.5版本以及主线分支必须匹配8.5或以上版本的toolkit包。

```shell
chmod +x Ascend-cann-toolkit_${VERSION}_linux-$(arch).run  # 其中${VERSION}表示对应的CANN版本，如8.2.RC1
./Ascend-cann-toolkit_${VERSION}_linux-$(arch).run --install
```

#### 安装后配置

配置环境变量脚本set_env.sh，当前安装路径以${HOME}/Ascend为例。

```sh
source ${HOME}/Ascend/ascend-toolkit/set_env.sh
```  

安装业务运行时依赖的Python第三方库（如果使用root用户安装，请将命令中的--user删除）。

```sh
pip3 install attrs cython 'numpy>=1.19.2,<=1.24.0' decorator sympy cffi pyyaml pathlib2 psutil protobuf==3.20.0 scipy requests absl-py --user
```

#### 安装ops算子包

安装前需已安装配套版本的Toolkit并配置环境变量

```shell
chmod +x Ascend-cann-${chip_type}-ops_${VERSION}_linux-$(arch).run  #其中${chip_type}表示昇腾产品类型，如A3
./Ascend-cann-${chip_type}-ops_${VERSION}_linux-$(arch).run --install
```  

### CANN详细安装指南 

开发者可访问[昇腾文档-昇腾社区](https://www.hiascend.com/document)->CANN社区版->软件安装，查看CANN软件安装引导，根据机器环境、操作系统和业务场景选择后阅读详细安装步骤。

### ATB安装部署相关依赖说明

在编译加速库之前，需访问[加速库包安装部署](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850alpha001/acce/ascendtb/ascendtb_0034.html)查看加速库相关依赖的版本要求，并进行对应依赖的安装部署。

### 基础工具版本要求与安装

安装CANN之后，您可安装一些工具方便后续开发，参见以下内容：

* [CANN依赖列表](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850alpha001/softwareinst/instg/instg_0045.html?Mode=PmIns&InstallType=local&OS=Debian&Software=cannToolKit)
* [CANN安装后操作](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850alpha001/softwareinst/instg/instg_0094.html?Mode=PmIns&InstallType=local&OS=Debian&Software=cannToolKit)

## 三、快速上手

### ATB编译

 - 加速库下载

    ```sh
    git clone https://gitcode.com/cann/ascend-transformer-boost.git
    ```

   您可自行选择需要的分支。
 - 环境变量设置  
    在编译前，需要安装nnal软件包（安装方法请查看[常见问题与回答](docs/常见问题与回答.md#run-package-usage)），并根据nnal安装路径设置环境变量`ATB_BUILD_DEPENDENCY_PATH`：

    ```sh
    export ATB_BUILD_DEPENDENCY_PATH={nnal install path}/nnal/atb/latest/atb/cxx_abi_{cxx_abi_version}
    ```

    注：未设置将使用默认路径`/usr/local/Ascend/nnal/atb/latest/atb/cxx_abi_{cxx_abi_version}`  
 - 加速库编译  
    编译加速库，设置加速库环境变量：

    ```sh
    cd ascend-transformer-boost
    bash scripts/build.sh
    source output/atb/set_env.sh
    ```

    注意：该编译过程涉及①拉取算子库/MKI并编译②加速库的编译两个过程。
 - 更多编译命令说明请参考[编译与构建](docs/编译与构建.md)

### 调用示例说明

本节示例代码分别展示了如何通过Python和C++调用算子。

#### Python

在运行python代码前，需要导入ATB Python API模块torch_atb，该插件运行依赖PyTorch和torch_npu，可访问[加速库包安装部署](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850alpha001/acce/ascendtb/ascendtb_0034.html)查看版本要求和安装指导。  
安装PyTorch和torch_npu之后，需要手动安装torch_atb，安装方法包含以下两种：

- 通过`./Ascend-cann-nnal_${version_info}_linux-aarch64.run --install --torch_atb`命令安装
- 在编译步骤中使用`bash scripts/build.sh --torch_atb`编译命令，`output/whl`文件夹下会生成torch_atb的whl文件，您可使用如下命令进行安装:

    ```sh
    pip3 install torch_atb-{version}-py3-none-any.whl
    ```

如下代码展示了如何通过Python调用算子，注意不要在ATB代码仓同名目录下运行：

```Python
import torch
import torch_atb#导入ATB Python API模块

#创建参数对象
linear_param = torch_atb.LinearParam()
linear_param.has_bias = False

#创建算子对象
op = torch_atb.Operation(linear_param)

#准备输入数据
x = torch.randn(2, 3, dtype=torch.float16).npu()  
y = torch.randn(2, 3, dtype=torch.float16).npu()

#使用forward方法完成操作，并获取输出
outputs = op.forward([x, y]) 
torch.npu.synchronize()
```

如需查看调用结果，可以打印第一个输出：

```Python
result=outputs[0].cpu().numpy()
print(result)
```

代码编写指导可访问[算子使用指导（ATB Python API）-昇腾社区](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850alpha001/acce/ascendtb/ascendtb_0077.html)。

#### C++

在ATB仓库的`example/op_demo`目录下，存放了多个不依赖测试框架、即编可用的算子调用Demo示例。进入对应目录执行如下命令就可完成一个算子的调用执行。代码完整内容可参考`example/op_demo/faupdate/faupdate_demo.cpp`，下面仅展示其核心内容：

```c++
// 设置卡号、创建context、设置stream
atb::Context *context = nullptr;
void *stream = nullptr;

CHECK_STATUS(aclInit(nullptr));
CHECK_STATUS(aclrtSetDevice(DEVICE_ID));
CHECK_STATUS(atb::CreateContext(&context));
CHECK_STATUS(aclrtCreateStream(&stream));
context->SetExecuteStream(stream);

// 创建op
atb::Operation *faupdateOp = nullptr;
CHECK_STATUS(CreateFaUpdateOperation(&faupdateOp));
// 准备输入tensor
atb::VariantPack variantPack;
CHECK_STATUS(PrepareInTensor(context, stream, variantPack.inTensors)); // 放入输入tensor
// 准备输出tensor
atb::Tensor output;
CHECK_STATUS(CreateTensor(ACL_FLOAT, aclFormat::ACL_FORMAT_ND, {LOCALOUT_DIM_1, LOCALOUT_DIM_2}, output));
variantPack.outTensors = {output}; // 放入输出tensor

uint64_t workspaceSize = 0;
// 计算workspaceSize大小
CHECK_STATUS(faupdateOp->Setup(variantPack, workspaceSize, context));
uint8_t *workspacePtr = nullptr;
if (workspaceSize > 0) {
    CHECK_STATUS(aclrtMalloc((void **)(&workspacePtr), workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST));
}
// faupdate执行
CHECK_STATUS(faupdateOp->Execute(variantPack, workspacePtr, workspaceSize, context));
CHECK_STATUS(aclrtSynchronizeStream(stream)); // 流同步，等待device侧任务计算完成

// 释放资源
for (atb::Tensor &inTensor : variantPack.inTensors) {
    CHECK_STATUS(aclrtFree(inTensor.deviceData));
}
for (atb::Tensor &outTensor : variantPack.outTensors) {
    CHECK_STATUS(aclrtFree(outTensor.deviceData));
}
if (workspaceSize > 0) {
    CHECK_STATUS(aclrtFree(workspacePtr));
}
CHECK_STATUS(atb::DestroyOperation(faupdateOp)); // operation，对象概念，先释放
CHECK_STATUS(aclrtDestroyStream(stream));
CHECK_STATUS(DestroyContext(context)); // context，全局资源，后释放
CHECK_STATUS(aclFinalize());
```

文件编译说明：进入`example/op_demo/faupdate`，执行`bash build.sh`完成编译和执行。

出现以下提示即为成功：

```sh
faupdate demo success!
```

代码编写指导：可访问[单算子-昇腾社区](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850alpha001/acce/ascendtb/ascendtb_0046.html)。

#### 样例安全声明

`example`目录下的样例旨在提供快速上手、开发和调试ATB特性的最小化实现，其核心目标是使用最精简的代码展示ATB核心功能，**而非提供生产级的安全保障**。与成熟的生产级使用方法相比，此样例中的安全功能（如输入校验、边界校验）相对有限。

ATB不推荐用户直接将样例作为业务代码，也不保证此种做法的安全性。若用户将`example`中的示例代码应用在自身的真实业务场景中且发生了安全问题，则由用户自行承担。

### 日志和环境变量说明

- 加速库日志请参考
  **[日志与调试](https://gitcode.com/cann/ascend-transformer-boost/blob/master/docs/%E6%97%A5%E5%BF%97%E4%B8%8E%E8%B0%83%E8%AF%95.md)**；
- 加速库日志现在已经部分适配CANN日志，环境变量说明请参考
  **[CANN社区版文档/环境变量参考](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/83RC1alpha002/maintenref/envvar/envref_07_0119.html)**；

## 四、自定义算子开发

您可参考以下文档进行自定义算子的开发：

- [从开发一个简单算子开始](docs/从开发一个简单算子出发.md)：以简单的Add算子的增加为例，介绍了ATB算子开发的交付件和开发流程，适合新入门的选手。
- [开发指南](docs/开发指南.md)：以一个融合算子为例，详细介绍了ATB算子开发的流程，以及如何对算子进行功能、精度、性能测试。  
**注意**：您在开发过程中遇到的问题，可参考[ATB日志与调试](docs/日志与调试.md)尝试解决。

## 五、参与贡献
 
1. fork仓库
2. 修改并提交代码
3. 新建 Pull-Request

详细步骤可参考[贡献指南](docs/贡献指南.md)

## 六、学习资源

- [编译与构建](docs/编译与构建.md)：ATB的编译命令说明。
- [从开发一个简单算子开始](docs/从开发一个简单算子出发.md)：以简单的Add算子的增加为例，介绍了ATB算子开发的交付件和开发流程。
- [开发指南](docs/开发指南.md)：以一个融合算子为例，详细介绍了ATB算子开发的流程，以及如何对算子进行功能、精度、性能测试。
- [贡献指南](docs/贡献指南.md)：介绍了如何向ATB库贡献代码。
- [日志与调试](docs/日志与调试.md)：介绍ATB的日志相关环境变量，以及调测方法。
- [API文档](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850alpha001/API/ascendtbapi/ascendtb_01_0098.html)：介绍了ATB库的接口和相关术语。
- [常见问题与回答](docs/常见问题与回答.md)：介绍ATB的编译和安装使用过程中遇到的一些常见问题与解决方案。
- [问题报告](https://gitcode.com/cann/ascend-transformer-boost/issues)：通过Issue提交发现的问题。

## 七、参考文档

**[CANN社区版文档](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850alpha001/index/index.html)**  
**[ATB社区版文档](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850alpha001/acce/ascendtb/ascendtb_0001.html)**
