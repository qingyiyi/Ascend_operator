#ifndef ADD_CUSTOM_TILING_H
#define ADD_CUSTOM_TILING_H
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(AddCustomTilingData)  //结构声明
  //结构成员的类型和名字
  TILING_DATA_FIELD_DEF(uint32_t, totalLength); // 添加tiling字段，总计算数据量
  TILING_DATA_FIELD_DEF(uint32_t, tileNum);     // 添加tiling字段，每个核上总计算数据分块个数
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(AddCustom, AddCustomTilingData)   //注册结构体到Addcustom中
}
#endif // ADD_CUSTOM_TILING_H