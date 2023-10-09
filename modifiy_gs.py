import onnx_graphsurgeon as gs
import onnx

graph = gs.import_onnx(onnx.load("graphs/simplified_model.onnx"))
cnt=0
for node in graph.nodes:
    if 'to_qkv/MatMul' in node.name:
        print(node.name)
        reshape_node=node.o(1).o()
        transpose_node=node.o(1).o().o().o()
        matmul_node=node.o(1).o().o().o().o()
        # matmul_node.inputs[1]=reshape_node.outputs[0]

        # print(transpose_node)
        trans_attrs={'perm': [0,2,3,1]}
        transpose_new=gs.Node("Transpose","transpose_new_{}".format(str(cnt)),inputs=[reshape_node.outputs[0]],outputs=[matmul_node.inputs[1]],attrs=trans_attrs)
        graph.nodes.append(transpose_new)
        transpose_node.outputs.clear()
        cnt+=1

# Remove the fake node from the graph completely
graph.cleanup()
onnx.save(gs.export_onnx(graph), "graphs/gs.onnx")
