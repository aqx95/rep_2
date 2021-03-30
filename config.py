
class GlobalConfig:
    #directory
    IMG_PATH = '../train'
    MASK_PATH = '../masks'
    LOG_PATH = 'log'
    SAVE_PATH = 'save'

    #training
    batch_size = 8
    num_epochs = 10
    num_folds = 5
    seed = 2020
    train_one_fold = True

    criterion = 'dicebce'
    criterion_params = {'dice': {'weight':None,'size_average':True},
                        'dicebce': {'weight':None,'size_average':True},
                       }

    # Scheduler config
    scheduler = 'CosineAnnealingWarmRestarts'
    scheduler_params = {'StepLR': {'step_size':2, 'gamma':0.3, 'last_epoch':-1, 'verbose':True},

                'ReduceLROnPlateau': {'mode':'max', 'factor':0.5, 'patience':0, 'threshold':0.0001,
                                      'threshold_mode':'rel', 'cooldown':0, 'min_lr':0,
                                      'eps':1e-08, 'verbose':True},

                'CosineAnnealingWarmRestarts': {'T_0':20, 'T_mult':1, 'eta_min':1e-6, 'last_epoch':-1,
                                                'verbose':True}, #train step

                'CosineAnnealingLR':{'T_max':20, 'last_epoch':-1} #validation step
                }

    train_step_scheduler = True
    val_step_scheduler = False


    # optimizer
    optimizer = 'Adam'
    optimizer_params = {'AdamW':{'lr':1e-4, 'betas':(0.9,0.999), 'eps':1e-08,
                                 'weight_decay':1e-6,'amsgrad':False},
                        'SGD':{'lr':0.001, 'momentum':0., 'weight_decay':0.01},
                        'Adam':{'lr':1e-4, 'weight_decay':1e-5}
                        }

    #model
    net = 'unet'
    encoder = 'se_resnext50_32x4d'
