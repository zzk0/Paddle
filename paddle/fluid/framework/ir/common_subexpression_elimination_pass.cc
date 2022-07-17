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

namespace paddle {
namespace framework {
namespace ir {

void CommonSubexpressionElimainationPass::ApplyImpl(ir::Graph* graph) const {
  std::unordered_set<ir::Node*, HashNode, EqualNode> exist_nodes;
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

}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(common_subexpression_elimination_pass,
              paddle::framework::ir::CommonSubexpressionElimainationPass);
REGISTER_PASS_CAPABILITY(common_subexpression_elimination_pass);
