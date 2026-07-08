/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GRAPH_GE_TENSOR_IMPL_H_
#define GRAPH_GE_TENSOR_IMPL_H_


#include <string>
#include <vector>
#include <memory>
#include "graph/aligned_ptr.h"
#include "graph/attr_store.h"
#include "graph/detail/attributes_holder.h"
#include "graph/buffer.h"
#include "graph/ge_error_codes.h"
#include "graph/ge_tensor.h"
#include "graph/types.h"

namespace ge {
class GeTensorDescImpl {
 public:
  GeTensorDescImpl() = default;
  GeTensorDescImpl(const GeShape &shape, const Format format, const DataType dt);
  explicit GeTensorDescImpl(proto::TensorDescriptor *const proto_msg);
  ~GeTensorDescImpl() = default;

  GeShape &ShapeReference() const;
  GeShape &OriginShapeReference() const;

  bool GeTensorDescAttrsAreEqual(const GeTensorDescImpl &other) const;
  bool operator==(const GeTensorDescImpl &other) const;

  ProtoAttrMap &MutableAttrMap();
  ConstProtoAttrMap &GetAttrMap() const;
  void SetShape(GeShape &shape) const;

  void SetDataType(const DataType dtype);
  DataType GetDataType() const;
  void SetFormat(const Format format);
  Format GetFormat() const;
  void SetOriginFormat(const Format format);
  Format GetOriginFormat() const;
  void SetOriginDataType(const DataType dtype);
  DataType GetOriginDataType() const;
  void SetName(const std::string &name);
  const std::string GetName() const;
  bool IsOriginShapeInited() const {
    return ext_meta_.IsOriginShapeInited();
  }
  void SetOriginShapeInited(const bool origin_shape_inited) {
    ext_meta_.SetOriginShapeInited(origin_shape_inited);
  }

  void SetExpandDimsRule(const std::string &expand_dims_rule) {
    ext_meta_.SetExpandDimsRule(expand_dims_rule);
  }
  std::string GetExpandDimsRule() const {
    return ext_meta_.GetExpandDimsRule();
  }

 private:
  friend class GeTensorImpl;
  friend class TensorUtils;
  friend class GeAttrValueImp;
  friend class ModelSerializeImp;
  friend class GeTensorSerializeUtils;
  friend class OnnxUtils;

  class ExtMeta {
   public:
    bool operator==(const ExtMeta& other) const {
      return (name == other.name) && (device_type == other.device_type) && (size == other.size) &&
        (weight_size == other.weight_size) && (cmps_tab_offset == other.cmps_tab_offset) &&
        (reuse_input_index == other.reuse_input_index) && (cmps_tab == other.cmps_tab) &&
        (data_offset == other.data_offset) && (cmps_size == other.cmps_size) &&
        (real_dim_cnt == other.real_dim_cnt) &&
        (other.reuse_input ? reuse_input : !reuse_input) &&
        (other.input_tensor ? input_tensor : !input_tensor) &&
        (other.output_tensor ? output_tensor : !output_tensor) &&
        (other.origin_shape_inited_ ? origin_shape_inited_ : !origin_shape_inited_);
    }
    // for name
    std::string GetName() const {
      return name;
    }

    void SetName(const std::string &v) {
      name = v;
    }

    // for device_type
    DeviceType GetDeviceType() const {
      return device_type;
    }

    std::string GetDeviceTypeStr() const;

    void SetDeviceType(const DeviceType v) {
      device_type = v;
    }

    // for size
    int64_t GetSize() const {
      return size;
    }

    void SetSize(const int64_t v) {
      size = v;
    }

    // for weight_size
    int64_t GetWeightSize() const {
      return weight_size;
    }

    void SetWeightSize(const int64_t v) {
      weight_size = v;
    }

    // for data_offset
    int64_t GetDataOffset() const {
      return data_offset;
    }

    void SetDataOffset(const int64_t v) {
      data_offset = v;
    }

    // for real_dim_cnt
    uint32_t GetRealDimCnt() const {
      return real_dim_cnt;
    }

    void SetRealDimCnt(const uint32_t v) {
      real_dim_cnt = v;
    }

    // for input_tensor
    bool GetInputTensor() const {
      return input_tensor;
    }

    void SetInputTensor(const bool v) {
      input_tensor = v;
    }

    // for reuse_input
    bool GetReuseInput() const {
      return reuse_input;
    }

    void SetReuseInput(const bool v) {
      reuse_input = v;
    }

    // for reuse_input_index
    uint32_t GetReuseInputIndex() const {
      return reuse_input_index;
    }

    void SetReuseInputIndex(const uint32_t v) {
      reuse_input_index = v;
    }

    // for output_tensor
    bool GetOutputTensor() const {
      return output_tensor;
    }

    void SetOutputTensor(const bool v) {
      output_tensor = v;
    }

    // for cmps_size
    int64_t GetCmpsSize() const {
      return cmps_size;
    }

    void SetCmpsSize(const int64_t v) {
      cmps_size = v;
    }

    // for cmps_tab
    std::string GetCmpsTab() const {
      return cmps_tab;
    }

    void SetCmpsTab(const std::string &v) {
      cmps_tab = v;
    }

    // for cmps_tab_offset
    int64_t GetCmpsTabOffset() const {
      return cmps_tab_offset;
    }

    void SetCmpsTabOffset(const int64_t v) {
      cmps_tab_offset = v;
    }

    bool IsOriginShapeInited() const {
      return origin_shape_inited_;
    }

    void SetOriginShapeInited(const bool origin_shape_inited) {
      origin_shape_inited_ = origin_shape_inited;
    }

    void SetExpandDimsRule(const std::string &expand_dims_rule) {
      expand_dims_rule_ = expand_dims_rule;
    }
    std::string GetExpandDimsRule() const {
      return expand_dims_rule_;
    }

   private:
    int64_t size{0};
    int64_t data_offset{0};
    int64_t cmps_tab_offset{0};
    int64_t cmps_size{0};
    int64_t weight_size{0};

    uint32_t real_dim_cnt{0U};
    uint32_t reuse_input_index{0U};

    DeviceType device_type{NPU};
    bool input_tensor{false};
    bool reuse_input{false};
    bool output_tensor{false};
    bool origin_shape_inited_{false};

    std::string cmps_tab;
    std::string name;

    std::string expand_dims_rule_;
  };

  mutable GeShape shape_;
  Format format_{FORMAT_ND};
  DataType dtype_{DT_FLOAT};

  mutable GeShape origin_shape_;
  Format origin_format_{FORMAT_ND};
  DataType origin_dtype_{DT_UNDEFINED};

  ExtMeta ext_meta_;
  AttrStore attrs_;
};

class TensorDataImpl {
 public:
  TensorDataImpl() = default;

  TensorDataImpl(const TensorDataImpl &other);

  ~TensorDataImpl() = default;

  TensorDataImpl &operator=(const TensorDataImpl &other);

  graphStatus SetData(const uint8_t * const data, const size_t size);
  graphStatus SetData(uint8_t * const data, const size_t size, const AlignedPtr::Deleter &delete_fuc);
  void SetData(std::shared_ptr<AlignedPtr> aligned_ptr, const size_t size);

  graphStatus ResetData(uint8_t *const data, const size_t size, const AlignedPtr::Deleter &delete_fuc);

  const uint8_t *MallocAlignedPtr(const size_t size);

  size_t GetSize() const;
  const uint8_t *GetData() const;
  uint8_t *GetData();
  bool IsTensorDataValid() const;

  void clear();

  uint8_t operator[](const size_t index) const;

  const std::shared_ptr<AlignedPtr> &GetAlignedPtr() const { return aligned_ptr_; }

 private:
  friend class GeTensorImpl;
  friend class TensorUtils;
  friend class GeAttrValueImp;
  friend class ModelSerializeImp;
  friend class GeTensorSerializeUtils;
  // TensorDatat shared with a GeTensorDesc by holding the impl of GeTensorDesc
  std::shared_ptr<GeTensorDescImpl> tensor_descriptor_;
  std::shared_ptr<AlignedPtr> aligned_ptr_ = nullptr;
  size_t length_ = 0UL;
  // functions data() & mutable_data() return address of invalid_data_ when length_ is 0
  // defined for coding convenience
  static uint32_t invalid_data_;
};

class GeTensorImpl {
 public:
  GeTensorImpl();
  explicit GeTensorImpl(const GeTensorDesc &tensor_desc);
  GeTensorImpl(const GeTensorDesc &tensor_desc, const std::vector<uint8_t> &data);
  GeTensorImpl(const GeTensorDesc &tensor_desc, const uint8_t * const data, const size_t size);
  GeTensorImpl(GeTensorDesc &&tensor_desc, std::vector<uint8_t> &&data);
  GeTensorImpl(const GeTensorDesc &tensor_desc, const Buffer &data);
  GeTensorImpl(const GeTensorDesc &tensor_desc, std::shared_ptr<AlignedPtr> aligned_ptr, const size_t size);
  GeTensorImpl(const GeTensorDesc &tensor_desc, const size_t size);
  GeTensorImpl(const ProtoMsgOwner &proto_owner, proto::TensorDef *proto_msg);
  GeTensorImpl(const GeTensorImpl &other);

  ~GeTensorImpl() = default;

  GeTensorImpl &operator=(const GeTensorImpl &other);

  GeTensorDesc &DescReference() const;
  void BuildAlignerPtrWithProtoData();
  graphStatus SetData(std::vector<uint8_t> &&data);
  graphStatus SetData(const std::vector<uint8_t> &data);
  graphStatus SetData(const uint8_t * const data, size_t const size);
  graphStatus SetData(const Buffer &data);
  graphStatus SetData(const TensorData &data);
  graphStatus SetData(uint8_t * const data, const size_t size, const AlignedPtr::Deleter &delete_fuc);
  void ClearData();
  void Clone(GeTensorImpl &tensor) const;

  std::shared_ptr<AlignedPtr> GetAlignedPtr() const;
  const TensorData &GetData() const { return tensor_data_; }
  TensorData &MutableData() { return tensor_data_; }
  bool IsTensorDataValid() const;
  // zero copy SetData
  void SetData(std::shared_ptr<AlignedPtr> aligned_ptr, const size_t size) {
    tensor_data_.SetData(std::move(aligned_ptr), size);
  }
  graphStatus ResetData(uint8_t *const data, const size_t size, const AlignedPtr::Deleter &delete_fuc);

 private:
  friend class TensorUtils;
  friend class GeAttrValueImp;
  friend class ModelSerializeImp;
  friend class GeTensorSerializeUtils;
  GeIrProtoHelper<proto::TensorDef> tensor_def_;
  // Reference from tensor_data_, do not direct use
  mutable GeTensorDesc desc_;
  TensorData tensor_data_;
};
}  // namespace ge
#endif  // GRAPH_GE_TENSOR_IMPL_H_

