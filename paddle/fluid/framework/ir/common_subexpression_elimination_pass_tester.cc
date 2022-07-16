/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include <gtest/gtest.h>

#include "paddle/fluid/framework/ir/common_subexpression_elimination_pass.h"
#include "paddle/fluid/framework/ir/pass_tester_helper.h"
#include "paddle/fluid/framework/op_version_registry.h"


namespace paddle {
namespace framework {
namespace ir {

TEST(CommonSubexpressionEliminationPass, basic) {
  // inputs                           operator            output
  // --------------------------------------------------------------------
  // (a, b)                        elementweise_add ->      e
  // (a, b)                        elementweise_add ->      f
  // (e, c)                        elementweise_add ->      g
  // (f, d)                        elementweise_add ->      h

  std::cout << "Test CSE" << std::endl;
  Layers layers;
  auto* a = layers.data("a", {1024, 768});
  auto* b = layers.data("b", {1024, 768});
  auto* c = layers.data("c", {1024, 768});
  auto* d = layers.data("d", {1024, 768});
  auto* e = layers.data("d", {1024, 768});
  auto* f = layers.data("d", {1024, 768});
  auto* g = layers.data("d", {1024, 768});
  auto* h = layers.data("d", {1024, 768});

  layers.elementwise_add(a, b, e, 0);
  layers.elementwise_add(a, b, f, 0);
  layers.elementwise_add(e, c, g, 0);
  layers.elementwise_add(f, d, h, 0);

  std::unique_ptr<ir::Graph> graph(new ir::Graph(layers.main_program()));
  auto pass =
      PassRegistry::Instance().Get("common_subexpression_elimination_pass");
  int num_nodes_before = graph->Nodes().size();
  VLOG(3) << DebugString(graph);

  graph.reset(pass->Apply(graph.release()));
  int num_nodes_after = graph->Nodes().size();
  VLOG(3) << DebugString(graph);

  std::cout << num_nodes_before << " -> " << num_nodes_after << std::endl;
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

USE_PASS(common_subexpression_elimination_pass);
