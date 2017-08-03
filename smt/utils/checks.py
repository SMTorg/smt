def check_support(sm, name):
    if not sm.supports[name]:
        class_name = sm.__class__.__name__
        raise NotImplementedError('{} does not support {}'.format(class_name, name))

def check_x_shape(nx, x):
    if x.shape[1] != nx:
        if nx == 1:
            raise ValueError('x should have shape [:, 1] or [:]')
        else:
            raise ValueError('x should have shape [:, {}]'.format(nx))
