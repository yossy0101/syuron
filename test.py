from ptflops import get_model_complexity_info
if __name__ == '__main__':
    # prepare input tensor
    net=vgg16(num_classes=100, alpha=0.5)
    flops, params = get_model_complexity_info(net, (3, 256, 256), as_strings=None,
                                            print_per_layer_stat=False, verbose=True)
    print('{:<30}  {:<8}'.format('FLOPs: ', flops))
    print('{:<30}  {:<8}'.format('parameters: ', params))