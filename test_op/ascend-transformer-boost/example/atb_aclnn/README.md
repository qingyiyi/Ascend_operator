# aclnnPluginOperation与ATBOperation混搭组图示例

## 介绍

本用例为aclnnPluginOperation与ATBOperation混搭组图的运行示例，该用例整体分为五个部分：aclnn算子部分、ATB算子部分、model部分、基础设施部分、主函数。

- aclnn算子部分：实现了aclnn算子对ATB的接入。
- ATB算子部分：使用ATB原生算子构建了一个ATB图算子。
- model部分：主要使用了ATB算子部分与aclnn算子部分生成的算子构建了一个更大的图算子，并且包含ATB图算子相关的调用逻辑。
- 基础设施部分：主要包含日志与内存池，用于辅助问题定位及加快显存分配速度。
- 主函数部分：承载着程序入口的功能，同时多线程功能也在主函数中实现。

## 用例运行

- 进入到atb-aclnn目录下:

    ```sh
    cd  ${用例所在目录}/atb-aclnn
    ```

- 设置CANN环境变量（如source /usr/local/Ascend/ascend-toolkit/set_env.sh）:

    ```sh
    source ${toolkit安装目录}/set_env.sh
    ```

- 设置ATB环境变量（如source /usr/local/Ascend/nnal/atb/set_env.sh）:

    ```sh
    source ${nnal安装目录}/atb/set_env.sh
    ```

- 执行build.sh脚本:

    ```sh
    bash ./build.sh
    ```

- 执行用例:

    ```sh
    ./build/test_model
    ```

## 说明

- 当前用例根据物理机上的device卡数创建线程，若需要调整线程个数，请自行修改main.cpp中的线程创建个数。
- 保证demo的ABI版本与ATB一致。ATB的abi版本的查看方法：

    ```sh
    env | grep ATB_HOME_PATH
    ```

    demo的ABI版本查看方法：通过查看CMakeList中的如下命令来查看ABI版本为0或1，OFF表示0，ON表示1。下面的代码则表示ABI版本为0。

    ```sh
    option(USE_CXX11_ABI "USE_CXX11_ABI" OFF) 
    ```
