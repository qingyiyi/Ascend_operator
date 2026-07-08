
/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef INC_GRAPH_ATTR_GROUP_SERIALIZE_H
#define INC_GRAPH_ATTR_GROUP_SERIALIZE_H

#include "graph/ge_error_codes.h"
#include "graph/attr_store.h"
#include "graph/detail/attributes_holder.h"

namespace ge {
namespace proto {
class AttrGroups;
}
class AttrGroupSerialize {
 public:
  static graphStatus SerializeAllAttr(proto::AttrGroups &attr_groups, const AttrStore &attr_store);
  static graphStatus DeserializeAllAttr(const proto::AttrGroups &attr_group, AttrHolder *attr_holder);

 private:
  static graphStatus OtherGroupSerialize(proto::AttrGroups &attr_groups, const AttrStore &attr_store);
  static graphStatus OtherGroupDeserialize(const proto::AttrGroups &attr_groups, AttrStore &attr_store) ;
};
}

#endif  // INC_GRAPH_ATTR_GROUP_SERIALIZE_H
