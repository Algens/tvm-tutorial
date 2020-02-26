#this example code shows how to complie
#tvm module line-by-line.
#mod.astext() shows every operation in a module
#and we can use that information to
#build a single operation.


from tvm import relay
from tvm.relay import testing
import tvm
from tvm.contrib import graph_runtime

batch_size = 1
num_class = 1000
image_shape = (3, 224, 224)
data_shape = (batch_size,) + image_shape
out_shape = (batch_size, num_class)
input_data = np.random.uniform(-1, 1, size=origin_shape).astype("float32")


#load a pre-trained model
mod, params_all = relay.testing.resnet.get_workload(
    num_layers=18, batch_size=batch_size, image_shape=image_shape)

#print out which operations had been done in the model
print("mod.astext()\n")
print(mod.astext())

#example of mod.astext()'s first operation
"""
%0 = nn.batch_norm(%data, %bn_data_gamma, %bn_data_beta, %bn_data_moving_mean, %bn_data_moving_var, epsilon=2e-05f, scale=False) /* ty=(Tensor[(1, 3, 224, 224), float32], Tensor[(3), float32], Tensor[(3), float32]) */;
"""
#in this example, we're gonna build this single operation.

#placeholder for input
data_var = relay.var("data", shape="data_shape", dtype="float32")

#actual operation
#relay API finds correct values for relay.var()
#we always use 0th element of batch_norm operation, so there's [0] at the end.
v0 = relay.nn.batch_norm(data_var, relay.var("bn_data_gamma"), relay.var("bn_data_beta"),
			relay.var("bn_data_moving_mean"), relay.var("bn_data_moving_var"),
			epsilon=2e-05f, scale=False)[0]
v0 = relay.Function(relay.analysis.free_vars(v0), v0)
v0 = relay.Module.from_expr(v0)
#print actual operation
print("\nv0.astext()")
print(v0.astext())


#function that builds a model with tvm.relay
def _build(m):
	opt_level = 2
	target = "llvm"
	with relay.build_config(opt_level=opt_level):
		graph, lib, params = relay.build_module.build(
		m, target, params=params_all)
	return graph, lib, params


#result of relay build
graph, lib, params = _build(v0)

# create module
ctx = tvm.cpu()
module = graph_runtime.create(graph, lib, ctx)

# set input and parameters
module.set_input("data", input_data)
module.set_input(**params)

# run the model
module.run()

# get output
v0_result = module.get_output(0, tvm.nd.empty((1, 3, 224, 224))).asnumpy()

# Print first 10 elements of output
print("\nv0.flatten()")
print(v0_result.flatten()[0:10])

