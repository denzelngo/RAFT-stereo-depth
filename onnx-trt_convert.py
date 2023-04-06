import tensorrt as trt
import argparse
import os


class OnnxError(Exception):
    def __init__(self, message):
        self.message = message


def onnx_to_plan(model_path,
                 output_path,
                 fp16=False,
                 max_batch_size=1,
                 dynamic_batch=False,
                 max_workspace=1 << 30):
    EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    # create trt logger and builder
    trt_logger = trt.Logger(trt.Logger.VERBOSE)
    builder = trt.Builder(trt_logger)
    network = builder.create_network(EXPLICIT_BATCH)
    config = builder.create_builder_config()
    if fp16:
        config.set_flag(trt.BuilderFlag.FP16)

    # create onnx parser
    parser = trt.OnnxParser(network, trt_logger)

    config.max_workspace_size = max_workspace

    builder.max_batch_size = max_batch_size

    if dynamic_batch:
        profile = builder.create_optimization_profile()
        profile.set_shape('images', (1, 3, 80, 240), (5, 3, 80, 240), (10, 3, 80, 240))
        # profile.set_shape('box',(1,1500,4),(5,1500,4),(10,1500,4))
        # profile.set_shape('score',(1,1500,35),(5,1500,35),(10,1500,35))
        config.add_optimization_profile(profile)

    # infer with onnx model
    with open(model_path, 'rb') as model:
        print('ONNX parsing... from {}'.format(model_path))
        if not parser.parse(model.read()):
            print('ERROR: Failed to parse the ONNX file.')
            for error in range(parser.num_errors):
                print(parser.get_error(error))
        else:
            # build optimized inference engine
            engine = builder.build_engine(network, config)

    # save inference engine
    if engine:
        with open(output_path, "wb") as f:
            f.write(engine.serialize())
        print('Done')
    else:
        print('FAIL CONVERSION!')


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--i', required=True, help='onnx model name')
    parser.add_argument('--o', required=True, help='tensorrt engine name')
    args = parser.parse_args()

    onnx_model_path = args.i
    plan_model_path = args.o

    try:
        if not os.path.isfile(args.i):
            raise OnnxError('error')
        onnx_to_plan(onnx_model_path,
                     plan_model_path,
                     fp16=False,  # change this for different TRT precision
                     max_batch_size=10,
                     dynamic_batch=False,
                     max_workspace=1 << 30)

    except OnnxError:
        print('Can not find the onnx model')
