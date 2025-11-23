# Copyright (c) Microsoft Corporation


def create_model(opt):
    assert opt.model == "pix2pixHD"

    from .pix2pixHD_model import InferenceModel

    assert not opt.isTrain
    model = InferenceModel()

    model.initialize(opt)
    if opt.verbose:
        print("model [%s] was created" % (model.name()))

    return model


def create_da_model(opt):
    assert opt.model == "pix2pixHD"

    from .pix2pixHD_model_DA import InferenceModel

    assert not opt.isTrain
    model = InferenceModel()

    model.initialize(opt)
    if opt.verbose:
        print("model [%s] was created" % (model.name()))

    return model
