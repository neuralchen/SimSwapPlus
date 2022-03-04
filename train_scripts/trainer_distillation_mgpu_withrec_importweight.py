#!/usr/bin/env python3
# -*- coding:utf-8 -*-
#############################################################
# File: trainer_naiv512.py
# Created Date: Sunday January 9th 2022
# Author: Chen Xuanhong
# Email: chenxuanhongzju@outlook.com
# Last Modified:  Friday, 4th March 2022 7:02:04 pm
# Modified By: Chen Xuanhong
# Copyright (c) 2022 Shanghai Jiao Tong University
#############################################################

import  os
import  time
import  random
import  shutil
import  tempfile

import  numpy as np

import  torch
import  torch.nn.functional as F

from    torch_utils import misc
from    torch_utils import training_stats
from    torch_utils.ops import conv2d_gradfix
from    torch_utils.ops import grid_sample_gradfix

from    losses.KA import KA
from    utilities.plot import plot_batch
from    train_scripts.trainer_multigpu_base import TrainerBase


class Trainer(TrainerBase):

    def __init__(self, 
                config, 
                reporter):
        super(Trainer, self).__init__(config, reporter)

        import inspect
        print("Current training script -----------> %s"%inspect.getfile(inspect.currentframe()))

    def train(self):
        # Launch processes.
        num_gpus = len(self.config["gpus"])
        print('Launching processes...')
        torch.multiprocessing.set_start_method('spawn')
        with tempfile.TemporaryDirectory() as temp_dir:
            torch.multiprocessing.spawn(fn=train_loop, args=(self.config, self.reporter, temp_dir), nprocs=num_gpus)

def add_mapping_hook(network, features,mapping_layers):
        mapping_hooks = []

        def get_activation(mem, name):
            def get_output_hook(module, input, output):
                mem[name] = output

            return get_output_hook

        def add_hook(net, mem, mapping_layers):
            for n, m in net.named_modules():
                if n in mapping_layers:
                    mapping_hooks.append(
                        m.register_forward_hook(get_activation(mem, n)))

        add_hook(network, features, mapping_layers)
        

# TODO modify this function to build your models
def init_framework(config, reporter, device, rank):
    '''
        This function is designed to define the framework,
        and print the framework information into the log file
    '''
    #===============build models================#
    print("build models...")
    # TODO [import models here]
    torch.cuda.set_device(rank)
    torch.cuda.empty_cache()
    model_config    = config["model_configs"]
    
    if config["phase"] == "train":
        gscript_name     = "components." + model_config["g_model"]["script"]
        file1       = os.path.join("components", model_config["g_model"]["script"]+".py")
        tgtfile1    = os.path.join(config["project_scripts"], model_config["g_model"]["script"]+".py")
        shutil.copyfile(file1,tgtfile1)
        dscript_name     = "components." + model_config["d_model"]["script"]
        file1       = os.path.join("components", model_config["d_model"]["script"]+".py")
        tgtfile1    = os.path.join(config["project_scripts"], model_config["d_model"]["script"]+".py")
        shutil.copyfile(file1,tgtfile1)
        
    elif config["phase"] == "finetune":
        gscript_name     = config["com_base"] + model_config["g_model"]["script"]
        dscript_name     = config["com_base"] + model_config["d_model"]["script"]
    com_base        = "train_logs."+config["teacher_model"]["version"]+".scripts"
    tscript_name    = com_base +"."+ config["teacher_model"]["model_configs"]["g_model"]["script"]
    class_name      = config["teacher_model"]["model_configs"]["g_model"]["class_name"]
    package         = __import__(tscript_name, fromlist=True)
    gen_class       = getattr(package, class_name)
    tgen            = gen_class(**config["teacher_model"]["model_configs"]["g_model"]["module_params"])
    tgen            = tgen.eval()
    
    class_name      = model_config["g_model"]["class_name"]
    package         = __import__(gscript_name, fromlist=True)
    gen_class       = getattr(package, class_name)
    gen             = gen_class(**model_config["g_model"]["module_params"])

    
    
    # print and recorde model structure
    reporter.writeInfo("Generator structure:")
    reporter.writeModel(gen.__str__())
    reporter.writeInfo("Teacher structure:")
    reporter.writeModel(tgen.__str__())

    class_name      = model_config["d_model"]["class_name"]
    package         = __import__(dscript_name, fromlist=True)
    dis_class       = getattr(package, class_name)
    dis             = dis_class(**model_config["d_model"]["module_params"])
    
    
    # print and recorde model structure
    reporter.writeInfo("Discriminator structure:")
    reporter.writeModel(dis.__str__())
    
    arcface1        = torch.load(config["arcface_ckpt"], map_location=torch.device("cpu"))
    arcface         = arcface1['model'].module
    
    # train in GPU

    # if in finetune phase, load the pretrained checkpoint
    if config["phase"] == "finetune":
        model_path = os.path.join(config["project_checkpoints"],
                                    "step%d_%s.pth"%(config["ckpt"],
                                    config["checkpoint_names"]["generator_name"]))
        gen.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))

        model_path = os.path.join(config["project_checkpoints"],
                                    "step%d_%s.pth"%(config["ckpt"],
                                    config["checkpoint_names"]["discriminator_name"]))
        dis.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
        
        print('loaded trained backbone model step {}...!'.format(config["project_checkpoints"]))
    model_path = os.path.join(config["teacher_model"]["project_checkpoints"],
                                    "step%d_%s.pth"%(config["teacher_model"]["model_step"],
                                    config["teacher_model"]["checkpoint_names"]["generator_name"]))
    tgen.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    print('loaded trained teacher backbone model step {}...!'.format(config["teacher_model"]["model_step"]))
    tgen   = tgen.to(device)
    tgen.requires_grad_(False)
    gen    = gen.to(device)
    dis    = dis.to(device)
    arcface= arcface.to(device)
    arcface.requires_grad_(False)
    arcface.eval()

    t_features = {}
    s_features = {}
    add_mapping_hook(tgen,t_features,config["feature_list"])
    add_mapping_hook(gen,s_features,config["feature_list"])
    
    return tgen, gen, dis, arcface, t_features, s_features

# TODO modify this function to configurate the optimizer of your pipeline
def setup_optimizers(config, reporter, gen, dis, rank):

    torch.cuda.set_device(rank)
    torch.cuda.empty_cache()
    g_train_opt     = config['g_optim_config']
    d_train_opt     = config['d_optim_config']

    g_optim_params  = []
    d_optim_params  = []
    for k, v in gen.named_parameters():
        if v.requires_grad:
            g_optim_params.append(v)
        else:
            reporter.writeInfo(f'Params {k} will not be optimized.')
            print(f'Params {k} will not be optimized.')
    
    for k, v in dis.named_parameters():
        if v.requires_grad:
            d_optim_params.append(v)
        else:
            reporter.writeInfo(f'Params {k} will not be optimized.')
            print(f'Params {k} will not be optimized.')

    optim_type = config['optim_type']
    
    if optim_type == 'Adam':
        g_optimizer = torch.optim.Adam(g_optim_params,**g_train_opt)
        d_optimizer = torch.optim.Adam(d_optim_params,**d_train_opt)
    else:
        raise NotImplementedError(
            f'optimizer {optim_type} is not supperted yet.')
    # self.optimizers.append(self.optimizer_g)
    if config["phase"] == "finetune":
        opt_path = os.path.join(config["project_checkpoints"],
                                    "step%d_optim_%s.pth"%(config["ckpt"],
                                    config["optimizer_names"]["generator_name"]))
        g_optimizer.load_state_dict(torch.load(opt_path))

        opt_path = os.path.join(config["project_checkpoints"],
                                    "step%d_optim_%s.pth"%(config["ckpt"],
                                    config["optimizer_names"]["discriminator_name"]))
        d_optimizer.load_state_dict(torch.load(opt_path))
        
        print('loaded trained optimizer step {}...!'.format(config["project_checkpoints"]))
    return g_optimizer, d_optimizer


def train_loop(
        rank,
        config,
        reporter,
        temp_dir
    ):

    version     = config["version"]

    ckpt_dir    = config["project_checkpoints"]
    sample_dir  = config["project_samples"]
    
    log_freq    = config["log_step"]
    model_freq  = config["model_save_step"]
    sample_freq = config["sample_step"]
    total_step  = config["total_step"]
    random_seed = config["dataset_params"]["random_seed"]

    
    id_w        = config["id_weight"]
    rec_w       = config["reconstruct_weight"]
    feat_w      = config["feature_match_weight"]
    distill_w   = config["distillation_weight"]
    distill_rec_w   = config["teacher_reconstruction"]
    distill_feat_w   = config["teacher_featurematching"]

    feat_num    = len(config["feature_list"])

    num_gpus    = len(config["gpus"])
    batch_gpu   = config["batch_size"] // num_gpus

    init_file = os.path.abspath(os.path.join(temp_dir, '.torch_distributed_init'))
    if os.name == 'nt':
        init_method = 'file:///' + init_file.replace('\\', '/')
        torch.distributed.init_process_group(backend='gloo', init_method=init_method, rank=rank, world_size=num_gpus)
    else:
        init_method = f'file://{init_file}'
        torch.distributed.init_process_group(backend='nccl', init_method=init_method, rank=rank, world_size=num_gpus)

    # Init torch_utils.
    sync_device = torch.device('cuda', rank)
    training_stats.init_multiprocessing(rank=rank, sync_device=sync_device)

    

    if rank == 0:
        img_std     = torch.Tensor([0.229, 0.224, 0.225]).view(3,1,1)
        img_mean    = torch.Tensor([0.485, 0.456, 0.406]).view(3,1,1)

    
    # Initialize.
    device = torch.device('cuda', rank)
    np.random.seed(random_seed * num_gpus + rank)
    torch.manual_seed(random_seed * num_gpus + rank)
    torch.backends.cuda.matmul.allow_tf32 = False       # Improves numerical accuracy.
    torch.backends.cudnn.allow_tf32 = False             # Improves numerical accuracy.
    conv2d_gradfix.enabled = True                       # Improves training speed.
    grid_sample_gradfix.enabled = True                  # Avoids errors with the augmentation pipe.
    
    # Create dataloader.
    if rank == 0:
        print('Loading training set...')

    dataset   = config["dataset_paths"][config["dataset_name"]]
    #================================================#
    print("Prepare the train dataloader...")
    dlModulename    = config["dataloader"]
    package         = __import__("data_tools.data_loader_%s"%dlModulename, fromlist=True)
    dataloaderClass = getattr(package, 'GetLoader')
    dataloader_class= dataloaderClass
    dataloader      = dataloader_class(dataset,
                                    rank,
                                    num_gpus,
                                    batch_gpu,
                                    **config["dataset_params"])

    # Construct networks.
    if rank == 0:
        print('Constructing networks...')
    tgen, gen, dis, arcface, t_feat, s_feat  = init_framework(config, reporter, device, rank)

    # Check for existing checkpoint

    # Print network summary tables.
    # if rank == 0:
    #     attr    = torch.empty([batch_gpu, 3, 512, 512], device=device)
    #     id      = torch.empty([batch_gpu, 3, 112, 112], device=device)
    #     latent  = misc.print_module_summary(arcface, [id])
    #     img     = misc.print_module_summary(gen, [attr, latent])
    #     misc.print_module_summary(dis, [img, None])
    #     del attr
    #     del id
    #     del latent
    #     del img
    #     torch.cuda.empty_cache()
        

    # Distribute across GPUs.
    if rank == 0:
        print(f'Distributing across {num_gpus} GPUs...')
    for module in [gen, dis, arcface, tgen]:
        if module is not None and num_gpus > 1:
            for param in misc.params_and_buffers(module):
                torch.distributed.broadcast(param, src=0)

    # Setup training phases.
    if rank == 0:
        print('Setting up training phases...')
    #===============build losses===================#
    # TODO replace below lines to build your losses
    # MSE_loss    = torch.nn.MSELoss()
    l1_loss     = torch.nn.L1Loss()
    l1_loss_import = torch.nn.L1Loss(reduce=False)
    cos_loss    = torch.nn.CosineSimilarity()

    g_optimizer, d_optimizer = setup_optimizers(config, reporter, gen, dis, rank)

    # Initialize logs.
    if rank == 0:
        print('Initializing logs...')
        #==============build tensorboard=================#
        if config["logger"] == "tensorboard":
            import torch.utils.tensorboard as tensorboard
            tensorboard_writer  = tensorboard.SummaryWriter(config["project_summary"])
            logger              = tensorboard_writer

        elif config["logger"] == "wandb":
            import wandb
            wandb.init(project="Simswap_HQ", entity="xhchen", notes="512",
                tags=[config["tag"]], name=version)
    
            wandb.config = {
                "total_step": config["total_step"],
                "batch_size": config["batch_size"]
                }
            logger = wandb

    
    random.seed(random_seed)
    randindex = [i for i in range(batch_gpu)]

    # set the start point for training loop
    if config["phase"] == "finetune":
        start = config["ckpt"]
    else:
        start = 0
    if rank == 0:
        import datetime    
        start_time  = time.time()

        # Caculate the epoch number
        print("Total step = %d"%total_step)

        print("Start to train at %s"%(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
        
        from utilities.logo_class import logo_class
        logo_class.print_start_training()

    dis.feature_network.requires_grad_(False)
    
    for step in range(start, total_step):
        gen.train()
        dis.train()
        for interval in range(2):
            random.shuffle(randindex)
            src_image1, src_image2  = dataloader.next()
            # if rank ==0:
                
            #     elapsed = time.time() - start_time
            #     elapsed = str(datetime.timedelta(seconds=elapsed))
            #     print("dataloader:",elapsed)
            
            if step%2 == 0:
                img_id = src_image2
            else:
                img_id = src_image2[randindex]

            img_id_112      = F.interpolate(img_id,size=(112,112), mode='bicubic')
            latent_id       = arcface(img_id_112)
            latent_id       = F.normalize(latent_id, p=2, dim=1)
            
            if interval == 0:
                 
                
                img_fake        = gen(src_image1, latent_id)
                gen_logits,_    = dis(img_fake.detach(), None)
                loss_Dgen       = (F.relu(torch.ones_like(gen_logits) + gen_logits)).mean()

                real_logits,_   = dis(src_image2,None)
                loss_Dreal      = (F.relu(torch.ones_like(real_logits) - real_logits)).mean()

                loss_D          = loss_Dgen + loss_Dreal
                d_optimizer.zero_grad(set_to_none=True)
                loss_D.backward()
                with torch.autograd.profiler.record_function('discriminator_opt'):
                    # params = [param for param in dis.parameters() if param.grad is not None]
                    # if len(params) > 0:
                    #     flat = torch.cat([param.grad.flatten() for param in params])
                    #     if num_gpus > 1:
                    #         torch.distributed.all_reduce(flat)
                    #         flat /= num_gpus
                    #     misc.nan_to_num(flat, nan=0, posinf=1e5, neginf=-1e5, out=flat)
                    #     grads = flat.split([param.numel() for param in params])
                    #     for param, grad in zip(params, grads):
                    #         param.grad = grad.reshape(param.shape)
                    params = [param for param in dis.parameters() if param.grad is not None]
                    flat = torch.cat([param.grad.flatten() for param in params])
                    torch.distributed.all_reduce(flat)
                    flat /= num_gpus
                    misc.nan_to_num(flat, nan=0, posinf=1e5, neginf=-1e5, out=flat)
                    grads = flat.split([param.numel() for param in params])
                    for param, grad in zip(params, grads):
                        param.grad = grad.reshape(param.shape)
                    d_optimizer.step()
                # if rank ==0:
                
                #     elapsed = time.time() - start_time
                #     elapsed = str(datetime.timedelta(seconds=elapsed))
                #     print("Discriminator training:",elapsed)
            else:
                
                # model.netD.requires_grad_(True)
                t_fake          = tgen(src_image1, latent_id)
                t_id            = arcface(t_fake.detach())
                t_feat          = dis.get_feature(t_fake.detach())
                realism         = cos_loss(t_id, latent_id)
                


                img_fake        = gen(src_image1, latent_id)

                Sacts = [
                    s_feat[key] for key in sorted(s_feat.keys())
                ]
                Tacts = [
                    t_feat[key] for key in sorted(t_feat.keys())
                ]
                loss_distill = 0
                for Sact, Tact in zip(Sacts, Tacts):
                    loss_distill += -KA(Sact, Tact) 
                # G loss
                loss_distill /= feat_num
                gen_logits,feat = dis(img_fake, None)
                
                loss_Gmain      = (-gen_logits).mean()
                img_fake_down   = F.interpolate(img_fake, size=(112,112), mode='bicubic')
                latent_fake     = arcface(img_fake_down)
                latent_fake     = F.normalize(latent_fake, p=2, dim=1)
                loss_G_ID       = (1 - cos_loss(latent_fake, latent_id)).mean()
                real_feat       = dis.get_feature(src_image1)
                feat_match_loss = l1_loss(feat["3"],real_feat["3"])

                feat_match_ts   = (realism * l1_loss_import(feat["3"],t_feat)).mean()
                t_rec_loss      = (realism * l1_loss_import(t_fake.detach(), img_fake)).mean()
                
                loss_G          = loss_Gmain + loss_G_ID * id_w + \
                                            feat_match_loss * feat_w + loss_distill * distill_w +\
                                                distill_feat_w * feat_match_ts + distill_rec_w * t_rec_loss
                if step%2 == 0:
                    #G_Rec
                    loss_G_Rec  = l1_loss(img_fake, src_image1)
                    loss_G      += loss_G_Rec * rec_w

                g_optimizer.zero_grad(set_to_none=True)
                loss_G.backward()
                with torch.autograd.profiler.record_function('generator_opt'):
                    params = [param for param in gen.parameters() if param.grad is not None]
                    flat = torch.cat([param.grad.flatten() for param in params])
                    torch.distributed.all_reduce(flat)
                    flat /= num_gpus
                    misc.nan_to_num(flat, nan=0, posinf=1e5, neginf=-1e5, out=flat)
                    grads = flat.split([param.numel() for param in params])
                    for param, grad in zip(params, grads):
                        param.grad = grad.reshape(param.shape)
                    g_optimizer.step()
                # if rank ==0:
                
                #     elapsed = time.time() - start_time
                #     elapsed = str(datetime.timedelta(seconds=elapsed))
                #     print("Generator training:",elapsed)
            
            
        # Print out log info
        if rank == 0 and (step + 1) % log_freq == 0:
            elapsed = time.time() - start_time
            elapsed = str(datetime.timedelta(seconds=elapsed))
            # print("ready to report losses")
            # ID_Total= loss_G_ID
            # torch.distributed.all_reduce(ID_Total)
            
            epochinformation="[{}], Elapsed [{}], Step [{}/{}], \
                    G_ID: {:.4f}, G_loss: {:.4f}, Rec_loss: {:.4f}, Fm_loss: {:.4f}, \
                    Distillaton_loss: {:.4f}, D_loss: {:.4f}, D_fake: {:.4f}, D_real: {:.4f}". \
                    format(version, elapsed, step, total_step, \
                    loss_G_ID.item(), loss_G.item(), loss_G_Rec.item(), feat_match_loss.item(), \
                    loss_distill.item(), loss_D.item(), loss_Dgen.item(), loss_Dreal.item())
            print(epochinformation)
            reporter.writeInfo(epochinformation)

            if config["logger"] == "tensorboard":
                logger.add_scalar('G/G_loss', loss_G.item(), step)
                logger.add_scalar('G/G_Rec', loss_G_Rec.item(), step)
                logger.add_scalar('G/G_feat_match', feat_match_loss.item(), step)
                logger.add_scalar('G/G_distillation', loss_distill.item(), step)
                logger.add_scalar('G/G_ID', loss_G_ID.item(), step)
                logger.add_scalar('D/D_loss', loss_D.item(), step)
                logger.add_scalar('D/D_fake', loss_Dgen.item(), step)
                logger.add_scalar('D/D_real', loss_Dreal.item(), step)
            elif config["logger"] == "wandb":
                logger.log({"G_Loss": loss_G.item()}, step = step)
                logger.log({"G_Rec": loss_G_Rec.item()}, step = step)
                logger.log({"G_feat_match": feat_match_loss.item()}, step = step)
                logger.log({"G_distillation": loss_distill.item()}, step = step)
                logger.log({"G_ID": loss_G_ID.item()}, step = step)
                logger.log({"D_loss": loss_D.item()}, step = step)
                logger.log({"D_fake": loss_Dgen.item()}, step = step)
                logger.log({"D_real": loss_Dreal.item()}, step = step)
            torch.cuda.empty_cache()
        
        if rank == 0 and ((step + 1) % sample_freq == 0 or (step+1) % model_freq==0):
            gen.eval()
            with torch.no_grad():
                imgs        = []
                zero_img    = (torch.zeros_like(src_image1[0,...]))
                imgs.append(zero_img.cpu().numpy())
                save_img    = ((src_image1.cpu())* img_std + img_mean).numpy()
                for r in range(batch_gpu):
                    imgs.append(save_img[r,...])
                arcface_112     = F.interpolate(src_image2,size=(112,112), mode='bicubic')
                id_vector_src1  = arcface(arcface_112)
                id_vector_src1  = F.normalize(id_vector_src1, p=2, dim=1)

                for i in range(batch_gpu):
                    
                    imgs.append(save_img[i,...])
                    image_infer = src_image1[i, ...].repeat(batch_gpu, 1, 1, 1)
                    img_fake    = gen(image_infer, id_vector_src1).cpu()
                    
                    img_fake    = img_fake * img_std
                    img_fake    = img_fake + img_mean
                    img_fake    = img_fake.numpy()
                    for j in range(batch_gpu):
                        imgs.append(img_fake[j,...])
                print("Save test data")
                imgs = np.stack(imgs, axis = 0).transpose(0,2,3,1)
                plot_batch(imgs, os.path.join(sample_dir, 'step_'+str(step+1)+'.jpg'))
                torch.cuda.empty_cache()
                
                    
            
        #===============adjust learning rate============#
        # if (epoch + 1) in self.config["lr_decay_step"] and self.config["lr_decay_enable"]:
        #     print("Learning rate decay")
        #     for p in self.optimizer.param_groups:
        #         p['lr'] *= self.config["lr_decay"]
        #         print("Current learning rate is %f"%p['lr'])

        #===============save checkpoints================#
        if rank == 0 and (step+1) % model_freq==0:
            
            torch.save(gen.state_dict(),
                    os.path.join(ckpt_dir, 'step{}_{}.pth'.format(step + 1, 
                                config["checkpoint_names"]["generator_name"])))
            torch.save(dis.state_dict(),
                    os.path.join(ckpt_dir, 'step{}_{}.pth'.format(step + 1, 
                                config["checkpoint_names"]["discriminator_name"])))
            
            torch.save(g_optimizer.state_dict(),
                    os.path.join(ckpt_dir, 'step{}_optim_{}'.format(step + 1, 
                                config["checkpoint_names"]["generator_name"])))
            
            torch.save(d_optimizer.state_dict(),
                    os.path.join(ckpt_dir, 'step{}_optim_{}'.format(step + 1, 
                                config["checkpoint_names"]["discriminator_name"])))
            print("Save step %d model checkpoint!"%(step+1))
            torch.cuda.empty_cache()
    print("Rank %d process done!"%rank)
    torch.distributed.barrier()