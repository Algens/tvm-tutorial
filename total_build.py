from tvm import relay
from tvm.relay import testing
import tvm
from tvm.contrib import graph_runtime

batch_size = 1
num_class = 1000
image_shape = (3, 224, 224)
data_shape = (batch_size,) + image_shape
out_shape = (batch_size, num_class)


#load a pre-trained model
mod, params_all = relay.testing.resnet.get_workload(
    num_layers=18, batch_size=batch_size, image_shape=image_shape)

#print out which operations had been done in the model
print(mod.astext())

#example of mod.astext()
"""
def @main(%data: Tensor[(1, 3, 224, 224), float32], %bn_data_gamma: Tensor[(3), float32], %bn_data_beta: Tensor[(3), float32], %bn_data_moving_mean: Tensor[(3), float32], %bn_data_moving_var: Tensor[(3), float32]) -> Tensor[(1, 3, 224, 224), float32] {
  %0 = nn.batch_norm(%data, %bn_data_gamma, %bn_data_beta, %bn_data_moving_mean, %bn_data_moving_var, epsilon=2e-05f, scale=False) /* ty=(Tensor[(1, 3, 224, 224), float32], Tensor[(3), float32], Tensor[(3), float32]) */;
  %0.0
}
"""


#function that builds a model with tvm.relay
def _build(m):
	opt_level = 2
	target = "llvm"
	with relay.build_config(opt_level=opt_level):
		graph, lib, params = relay.build_module.build(
		m, target, params=params_all)
	return graph, lib, params


#result of relay build
graph, lib, params = _build(mod)

# create module
ctx = tvm.cpu()
module = graph_runtime.create(graph, lib, ctx)

# set input and parameters
first_data = np.random.uniform(-1, 1, size=origin_shape).astype("float32")
module.set_input("data", first_data)
module.set_input(**params)

# run the model
module.run()

# get output
total = module.get_output(0, tvm.nd.empty((1, 1000))).asnumpy()

# Print first 10 elements of output
print("\ntotal.flatten()")
print(total.flatten()[0:10])

