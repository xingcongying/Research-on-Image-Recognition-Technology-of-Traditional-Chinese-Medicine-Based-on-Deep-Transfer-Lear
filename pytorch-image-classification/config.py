class DefaultConfigs(object):
    #1.string parameters
    train_data = "/home/hyq-user/TCM/train/"
    test_data = "/home/hyq-user/TCM/test/"
    test_data2 = ""
    test_data3 = ""
    val_data = "/home/hyq-user/TCM/val/"
    model_name = "densesnet121(64)"
    weights = "./checkpoints/"
    best_models = weights + "best_model/"
    submit = "./submit/"
    logs = "./logs/"
    gpus = "0"

    #2.numeric parameters
    epochs = 40
    batch_size =32
    img_height = 256
    img_weight = 256
    num_classes = 80
    seed = 888
    lr = 0.001
    lr_decay = 1e-4
    weight_decay = 1e-4

config = DefaultConfigs()
