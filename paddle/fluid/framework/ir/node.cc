/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/framework/ir/node.h"
#include <algorithm>
#include <sstream>
#include <string>
#include <vector>
#include "paddle/fluid/framework/framework.pb.h"
#include "paddle/fluid/framework/ir/pass_tester_helper.h"

namespace paddle {
namespace framework {
namespace ir {
// msvc15 don't support constexpr in correct way.
#if !defined(_WIN32)
constexpr char Node::kControlDepVarName[];
#else
const char Node::kControlDepVarName[] = "__control_var";
#endif

std::unique_ptr<Node> CreateNodeForTest(const std::string& name,
                                        Node::Type type) {
  return std::unique_ptr<Node>(new Node(name, type));
}

std::unique_ptr<Node> CreateNodeForTest(VarDesc* var_desc) {
  return std::unique_ptr<Node>(new Node(var_desc, 0));
}

std::unique_ptr<Node> CreateNodeForTest(OpDesc* op_desc) {
  return std::unique_ptr<Node>(new Node(op_desc));
}

std::string NodeTypeToString(Node::Type type) {
  if (type == Node::Type::kOperation) {
    return "kOperation";
  } else {
    return "kVariable";
  }
}

size_t HashNode::operator()(const Node* node) const {
  std::vector<Node*> sorted_inputs(node->inputs);
  std::vector<Node*> sorted_outputs(node->outputs);
  auto comparator = [](Node* a, Node* b) { return a->Name() > b->Name(); };
  std::stable_sort(sorted_inputs.begin(), sorted_inputs.end(), comparator);
  std::stable_sort(sorted_outputs.begin(), sorted_outputs.end(), comparator);

  std::stringstream ss;
  ss << NodeTypeToString(node->NodeType()) << ":" << node->Name() << "#";
  for (auto input : sorted_inputs) {
    ss << NodeTypeToString(input->NodeType()) << ":" << input->Name() << "#";
  }
  for (auto output : sorted_outputs) {
    ss << NodeTypeToString(output->NodeType()) << "#";
  }
  return std::hash<std::string>{}(ss.str());
}

bool EqualNode::operator()(const Node* lhs, const Node* rhs) const {
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

  std::vector<Node*> lhs_sorted_inputs(lhs->inputs);
  std::vector<Node*> lhs_sorted_outputs(lhs->outputs);
  std::vector<Node*> rhs_sorted_inputs(rhs->inputs);
  std::vector<Node*> rhs_sorted_outputs(rhs->outputs);
  auto comparator = [](Node* a, Node* b) { return a->Name() > b->Name(); };
  std::stable_sort(
      lhs_sorted_inputs.begin(), lhs_sorted_inputs.end(), comparator);
  std::stable_sort(
      lhs_sorted_outputs.begin(), lhs_sorted_outputs.end(), comparator);
  std::stable_sort(
      rhs_sorted_inputs.begin(), rhs_sorted_inputs.end(), comparator);
  std::stable_sort(
      rhs_sorted_outputs.begin(), rhs_sorted_outputs.end(), comparator);

  // compare outputs type
  if (lhs_sorted_outputs.size() != rhs_sorted_outputs.size()) {
    return false;
  }
  for (size_t i = 0; i < lhs_sorted_outputs.size(); ++i) {
    Node* a = lhs_sorted_outputs[i];
    Node* b = rhs_sorted_outputs[i];
    if (a->NodeType() != b->NodeType()) {
      return false;
    }
    // TODO(zzk0): need check data type
  }

  // compare inputs value
  if (lhs_sorted_inputs.size() != rhs_sorted_inputs.size()) {
    return false;
  }
  if (!std::equal(lhs_sorted_inputs.begin(),
                  lhs_sorted_inputs.end(),
                  rhs_sorted_inputs.begin())) {
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
