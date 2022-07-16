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
#include "paddle/fluid/framework/op_version_registry.h"

namespace paddle {
namespace framework {
namespace ir {

void CommonSubexpressionElimainationPass::ApplyImpl(ir::Graph* graph) const {
  std::cout << "CSE ApplyImpl" << std::endl;
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(common_subexpression_elimination_pass, paddle::framework::ir::CommonSubexpressionElimainationPass);
REGISTER_PASS_CAPABILITY(common_subexpression_elimination_pass);
