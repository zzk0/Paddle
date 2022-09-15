// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "paddle/fluid/framework/ir/common_subexpression_elimination_pass.h"
#include "paddle/fluid/framework/ir/graph_helper.h"
#include "paddle/fluid/framework/ir/graph_pattern_detector.h"
#include "paddle/fluid/framework/op_version_registry.h"
#include "paddle/fluid/framework/type_defs.h"
#include "paddle/phi/core/enforce.h"
#include "paddle/utils/variant.h"

namespace {

std::string NodeTypeToString(paddle::framework::ir::Node::Type type) {
  if (type == paddle::framework::ir::Node::Type::kOperation) {
    return "kOperation";
  } else {
    return "kVariable";
  }
}

const std::unordered_set<std::string> commutative_operators{"elementwise_add"};

template <class T>
inline void HashCombine(size_t& seed, const T& v) {
  std::hash<T> hasher{};
  seed ^= hasher(v) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
}

}  // namespace


namespace std {

#define HASH_ATTRIBUTE(attr, id, type)         \
  do {                                         \
    if (attr.index() == id) {                  \
      return std::hash<type>{}(get<id>(attr)); \
    }                                          \
  } while(0)

#define HASH_VECTOR_ATTRIBUTE(attr, id, type)         \
  do {                                                \
    if (attr.index() == id) {                         \
      std::vector<type> vec = get<id>(attr);          \
      size_t seed = 0;                                \
      for (const auto& v : vec) {                     \
        HashCombine(seed, v);                         \
      }                                               \
      return seed;                                    \
    }                                                 \
  } while(0)


template<>
struct hash<paddle::framework::Attribute> {
size_t operator()(const paddle::framework::Attribute& attr) const {
    if (attr.index() == 0) {
      return 0;
    }
    if (attr.index() == 7) {
      return static_cast<size_t>(get<7>(attr));
    }

    HASH_ATTRIBUTE(attr, 1, int);
    HASH_ATTRIBUTE(attr, 2, float);
    HASH_ATTRIBUTE(attr, 3, std::string);
    HASH_VECTOR_ATTRIBUTE(attr, 4, int);
    HASH_VECTOR_ATTRIBUTE(attr, 5, float);
    HASH_VECTOR_ATTRIBUTE(attr, 6, std::string);
    HASH_ATTRIBUTE(attr, 8, std::vector<bool>);
    // NOTE(zzk0): Is this reasonable?
    HASH_ATTRIBUTE(attr, 9, paddle::framework::BlockDesc*);
    HASH_ATTRIBUTE(attr, 10, int64_t);
    HASH_VECTOR_ATTRIBUTE(attr, 11, paddle::framework::BlockDesc*);
    HASH_VECTOR_ATTRIBUTE(attr, 12, int64_t);
    HASH_VECTOR_ATTRIBUTE(attr, 13, double);
    return 0;
  }
};

}

namespace paddle {
namespace framework {
namespace ir {

void CommonSubexpressionEliminationPass::ApplyImpl(ir::Graph* graph) const {
  std::unordered_set<ir::Node*, HashOpNode, EqualOpNode> exist_nodes;
  std::vector<Node*> nodes = TopologySortOperations(*graph);
  for (Node* node : nodes) {
    auto res = exist_nodes.insert(node);
    if (!res.second) {
      auto exist_node = *res.first;
      for (size_t i = 0; i < exist_node->outputs.size(); ++i) {
        Node* exist_node_output = exist_node->outputs[i];
        Node* current_node_output = node->outputs[i];
        std::vector<Node*> current_node_output_outputs =
            current_node_output->outputs;
        for (size_t i = 0; i < current_node_output_outputs.size(); ++i) {
          IR_NODE_LINK_TO(exist_node_output, current_node_output_outputs[i]);
        }
      }
      GraphSafeRemoveNodes(graph,
                           std::unordered_set<const Node*>(
                               node->outputs.begin(), node->outputs.end()));
      GraphSafeRemoveNodes(graph, {node});
    }
  }
}

size_t HashOpNode::operator()(const Node* node) const {
  PADDLE_ENFORCE_EQ(node->IsOp(),
                    true,
                    platform::errors::InvalidArgument(
                        "HashOpNode only supports operation node type"));

  size_t seed = 0;
  for (size_t i = 0; i < node->inputs.size(); ++i) {
    HashCombine(seed, node->inputs[i]->id());
    HashCombine(seed, node->GraphId());
  }
  for (size_t i = 0; i < node->outputs.size(); ++i) {
    if (node->outputs[i]->IsVar()) {
      HashCombine(seed, node->outputs[i]->Var()->GetDataType());
    }
  }
  OpDesc* desc = node->Op();
  std::vector<std::string> attributes = desc->AttrNames();
  sort(attributes.begin(), attributes.end());
  for (const std::string& attribute : attributes) {
    HashCombine(seed, desc->GetAttr(attribute));
  }
  return seed;
}

bool EqualOpNode::operator()(const Node* lhs, const Node* rhs) const {
  if (lhs == nullptr && rhs == nullptr) {
    return true;
  }
  if (lhs == nullptr || rhs == nullptr) {
    return false;
  }
  if (lhs->NodeType() != rhs->NodeType()) {
    return false;
  }
  if (lhs->Name() != rhs->Name()) {
    return false;
  }

  std::vector<Node*> lhs_inputs(lhs->inputs);
  std::vector<Node*> lhs_outputs(lhs->outputs);
  std::vector<Node*> rhs_inputs(rhs->inputs);
  std::vector<Node*> rhs_outputs(rhs->outputs);
  if (commutative_operators.count(lhs->Name()) != 0) {
    auto comparator = [](Node* a, Node* b) { return a->Name() > b->Name(); };
    std::stable_sort(lhs_inputs.begin(), lhs_inputs.end(), comparator);
    std::stable_sort(lhs_outputs.begin(), lhs_outputs.end(), comparator);
    std::stable_sort(rhs_inputs.begin(), rhs_inputs.end(), comparator);
    std::stable_sort(rhs_outputs.begin(), rhs_outputs.end(), comparator);
  }

  // compare inputs value
  if (lhs_inputs.size() != rhs_inputs.size()) {
    return false;
  }
  if (!std::equal(lhs_inputs.begin(), lhs_inputs.end(), rhs_inputs.begin())) {
    return false;
  }

  // compare attribute
  if (lhs->IsOp() && rhs->IsOp()) {
    OpDesc* lhs_desc = lhs->Op();
    OpDesc* rhs_desc = rhs->Op();
    std::vector<std::string> lhs_attr_names = lhs_desc->AttrNames();
    std::vector<std::string> rhs_attr_names = rhs_desc->AttrNames();
    if (lhs_attr_names.size() != rhs_attr_names.size()) {
      return false;
    }
    std::sort(lhs_attr_names.begin(), lhs_attr_names.end());
    std::sort(rhs_attr_names.begin(), rhs_attr_names.end());
    for (size_t i = 0; i < lhs_attr_names.size(); ++i) {
      if (lhs_attr_names[i] != rhs_attr_names[i]) {
        return false;
      }
      if (lhs_desc->GetAttr(lhs_attr_names[i]) !=
          rhs_desc->GetAttr(rhs_attr_names[i])) {
        return false;
      }
    }
  }
  return true;
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(common_subexpression_elimination_pass,
              paddle::framework::ir::CommonSubexpressionEliminationPass);
REGISTER_PASS_CAPABILITY(common_subexpression_elimination_pass);
