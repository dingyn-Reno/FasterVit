import onnx_graphsurgeon as gs
import onnx
import numpy as np

X = gs.Variable(name="X", dtype=np.float32, shape=(8, 50, 16, 64))
Y = gs.Variable(name="Y", dtype=np.float32, shape=(8,16,64,50))
Z=gs.Variable(name="Z", dtype=np.float32, shape=(8,16,50,64))
trans_attrs = {'perm': [0, 2, 1, 3]}
node = gs.Node("Transpose", "transpose_new", inputs=[X],
                        outputs=[Z], attrs=trans_attrs)

trans_attrs = {'perm': [0, 1,3,2]}
node2= gs.Node("Transpose", "transpose_new2", inputs=[Z],
                        outputs=[Y], attrs=trans_attrs)

graph = gs.Graph(nodes=[node,node2], inputs=[X], outputs=[Y])
onnx.save(gs.export_onnx(graph), "test_2.onnx")