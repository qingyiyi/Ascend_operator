# ATB加速库外部开发者自定义算子目录

## 介绍

单独为外部开发者设置开发目录，外部开发者可以按照本目录下的customize_block_copy Operation实现自定义算子。本目录支持单独编译和测试，也支持与ATB加速库一同编译。

## 使用说明

以customize_block_copy Operation为例

### 方式一：单独编译

#### 安装CANN

```shell
chmod +x Ascend-cann-toolkit_$(version)_linux-$(arch).run
./Ascend-cann-toolkit_$(version)_linux-$(arch).run --install
```

#### 安装后配置

配置环境变量脚本set_env.sh，当前安装路径以${HOME}/Ascend为例。

```shell
source ${HOME}/Ascend/ascend-toolkit/set_env.sh
```  

#### 安装NNAL

```shell
chmod +x Ascend-cann-nnal_$(version)_linux-$(arch).run
./Ascend-cann-nnal_$(version)_linux-$(arch).run --install
```

#### 安装后配置

配置环境变量脚本set_env.sh，当前安装路径以${HOME}/Ascend为例。

```shell
source ${HOME}/Ascend/nnal/atb/set_env.sh
```

#### 编译自定义算子目录

```shell
cd ascend-transformer-boost/ops_customize
bash build.sh
```

该脚本目前支持: `default|clean|unittest| --use_cxx11_abi=0|--use_cxx11_abi=1|--debug|--msdebug`
编译命令具体功能介绍：

- `default`: 默认选项，构建ops_customize的内容
- `clean`: 清理所有构建历史，删除构建目录
- `unittest`: 构建单元测试，运行ops_customize的单元测试
- `--use_cxx11_abi=0`: 禁用 `C++11 ABI`
- `--use_cxx11_abi=1`: 启用 `C++11 ABI`
- `--debug`: 设置构建类型为 `Debug` 模式
- `--msdebug`: 启用`MSDebug`模式，用于对算子内核代码进行调测

#### 执行测试用例

执行customize_block_copy Operation的测试用例

```shell
bash build.sh unittest
```

### 方式二：与ATB加速库一同编译

#### 准备环境变量

当前安装路径以${HOME}/Ascend为例。

```shell
source ${HOME}/Ascend/ascend-toolkit/set_env.sh
source ${HOME}/Ascend/nnal/atb/set_env.sh
export ATB_BUILD_DEPENDENCY_PATH=${ATB_HOME_PATH}
```

#### 编译带有自定义算子的ATB加速库

```shell
cd ascend-transformer-boost
bash scripts/build.sh customizeops
```

#### 执行测试用例

编译带有自定算子和测试用例的ATB加速库, 执行customize_block_copy Operation的测试用例

```shell
bash scripts/build.sh customizeops --customizeops_tests
source ./output/atb/set_env.sh
cd ./build/ops_customize/ops/customize_blockcopy/tests/ && ./customize_blockcopy_test
```
