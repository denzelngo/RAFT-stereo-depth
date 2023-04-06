import onnx
from onnxsim import simplify
import sys
from core.raft_stereo import RAFTStereo
import argparse
import torch

from torch.onnx import register_custom_op_symbolic
import torch.onnx.symbolic_helper as sym_help

# def grid_sampler(g, input, grid, mode, padding_mode, align_corners):
#     # mode
#     #   'bilinear'      : onnx::Constant[value={0}]
#     #   'nearest'       : onnx::Constant[value={1}]
#     #   'bicubic'       : onnx::Constant[value={2}]
#     # padding_mode
#     #   'zeros'         : onnx::Constant[value={0}]
#     #   'border'        : onnx::Constant[value={1}]
#     #   'reflection'    : onnx::Constant[value={2}]
#     mode = sym_help._maybe_get_const(mode, "i")
#     padding_mode = sym_help._maybe_get_const(padding_mode, "i")
#     mode_str = ['bilinear', 'nearest', 'bicubic'][mode]
#     padding_mode_str = ['zeros', 'border', 'reflection'][padding_mode]
#     align_corners = int(sym_help._maybe_get_const(align_corners, "b"))
#
#     return g.op("com.microsoft::GridSample", input, grid,
#                 mode_s=mode_str,
#                 padding_mode_s=padding_mode_str,
#                 align_corners_i=align_corners)
#
#
# register_custom_op_symbolic('::grid_sampler', grid_sampler, 1)

# Architecture choices
parser = argparse.ArgumentParser()
parser.add_argument('--valid_iters', type=int, default=3, help='number of flow-field updates during forward pass')
parser.add_argument('--hidden_dims', nargs='+', type=int, default=[128] * 3,
                    help="hidden state and context dimensions")
parser.add_argument('--corr_implementation', choices=["reg", "alt", "reg_cuda", "alt_cuda"], default="reg",
                    help="correlation volume implementation")
parser.add_argument('--shared_backbone', action='store_true',
                    help="use a single backbone for the context and feature encoders")
parser.add_argument('--corr_levels', type=int, default=4, help="number of levels in the correlation pyramid")
parser.add_argument('--corr_radius', type=int, default=4, help="width of the correlation pyramid")
parser.add_argument('--n_downsample', type=int, default=3, help="resolution of the disparity field (1/2^K)")
parser.add_argument('--slow_fast_gru', action='store_true', help="iterate the low-res GRUs more frequently")
parser.add_argument('--n_gru_layers', type=int, default=2, help="number of hidden GRU levels")

args = parser.parse_args()

args.shared_backbone = True
args.slow_fast_gru = True
args.mixed_precision = False

# model = torch.nn.DataParallel(RAFTStereo(args))
model = RAFTStereo(args)
model.load_state_dict(torch.load('best_faster_sceneflow.pt'))

model.eval().cuda()

onnx_file = 'raft_faster_gray_batch2.onnx'
x = torch.rand(2, 3, 200, 320).cuda()

torch.onnx.export(
    model,
    args=(x, x),
    input_names=['img_l', 'img_r'],
    output_names=['disp'],
    f=onnx_file,
    opset_version=12
)
print('Done onnx! Now simplify...')

model = onnx.load(onnx_file)
model_simp, check = simplify(model)
onnx.save(model_simp, 'raft_sim_faster_gray_batch2.onnx')

sys.exit(0)
