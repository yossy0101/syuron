from ptflops import get_model_complexity_info
def param(model_name=None, net=None, size=256):
    # prepare input tensor
    flops, params = get_model_complexity_info(net, (3, size, size), as_strings=None,
                                            print_per_layer_stat=False, verbose=True)
    print(model_name)
    print('{:<30}'.format('FLOPs: '), '{:,}'.format(flops))
    print('{:<30}'.format('parameters: '), '{:,}'.format(params))