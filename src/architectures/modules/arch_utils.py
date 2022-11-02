def enable_bias(norm_type):
    return False if norm_type != "identity" else True


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv2d") != -1:
        m.weight.data.normal_(0.0, 0.02)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find("Norm") != -1 and "Tanh" not in classname:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.zero_()
