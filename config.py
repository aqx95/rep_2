
class GlobalConfig:
    #directory
    IMG_PATH = '../train'
    MASK_PATH = '../masks'
    LOG_PATH = 'log'
    SAVE_PATH = 'save'

    #training
    batch_size = 16
    num_epochs = 20
    num_folds = 5
    seed = 2020
    train_one_fold = True

    #classification head
    use_cls = False
    threshold = 0.5
    aux_params={'classes':1,
          'dropout': 0.0,
          'activation':'sigmoid'}

    criterion = 'dicebce'
    criterion_params = {'dice': {'weight':None,'size_average':True},
                        'dicebce': {'bce_weight':0.5,'size_average':True},
                        'focal': {'alpha':0.8, 'gamma':2.0}
                       }

    # Scheduler config
    scheduler = 'CosineAnnealingLR'
    scheduler_params = {'StepLR': {'step_size':2, 'gamma':0.3, 'last_epoch':-1, 'verbose':True},

                'ReduceLROnPlateau': {'mode':'max', 'factor':0.5, 'patience':0, 'threshold':0.0001,
                                      'threshold_mode':'rel', 'cooldown':0, 'min_lr':0,
                                      'eps':1e-08, 'verbose':True},

                'CosineAnnealingWarmRestarts': {'T_0':20, 'T_mult':1, 'eta_min':1e-6, 'last_epoch':-1,
                                                'verbose':True}, #train step

                'CosineAnnealingLR':{'T_max':15, 'last_epoch':-1} #validation step
                }

    train_step_scheduler = False
    val_step_scheduler = True


    # optimizer
    optimizer = 'Adam'
    optimizer_params = {'AdamW':{'lr':1e-4, 'betas':(0.9,0.999), 'eps':1e-08,
                                 'weight_decay':1e-6,'amsgrad':False},
                        'SGD':{'lr':0.001, 'momentum':0., 'weight_decay':0.01},
                        'Adam':{'lr':1e-4, 'weight_decay':1e-5}
                        }

    #model
    net = 'unet'
    encoder = 'efficientnet-b0'
