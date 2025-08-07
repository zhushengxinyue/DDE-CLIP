import models
import torch
import argparse
import torch.nn.functional as F
from utils.prompt_ensemble import DDECLIP_PromptLearner
from utils.loss import FocalLoss, BinaryDiceLoss
from utils.dataset import Dataset
from utils.logger import get_logger
from tqdm import tqdm
import numpy as np
import os
import random
from utils.utils import get_transform
import torch.nn as nn

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train(args):

    logger = get_logger(args.save_path)

    preprocess, target_transform = get_transform(args)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    DDECLIP_parameters = {"Prompt_length": args.n_ctx, "learnabel_text_embedding_depth": args.depth, "learnabel_text_embedding_length": args.t_n_ctx}

    model, _ = models.load("pretrained_weight/ViT-L-14-336px.pt", device=device, design_details = DDECLIP_parameters)
    model.eval()

    train_data = Dataset(root=args.train_data_path, transform=preprocess, target_transform=target_transform, dataset_name = args.dataset)
    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True)


    prompt_learner = DDECLIP_PromptLearner(model.to("cpu"), DDECLIP_parameters)
    prompt_learner.to(device)
    model.to(device)
    model.visual.DAPM_replace(DPAM_layer = 20)

    for name, param in model.visual.named_parameters():
        if 'adapters' not in name: 
            param.requires_grad = False

    prompt_learner_params = list(prompt_learner.parameters())
    adapter_params = []
    for adapter_module in model.visual.adapters:
        adapter_params.extend(list(adapter_module.parameters())) 
    cross_attn_params = []
    if hasattr(model, 'cross_attn_layers'):
        for cross_attn_module in model.cross_attn_layers:
            cross_attn_params.extend(list(cross_attn_module.parameters()))
    gamma_text_param = [model.gamma_text]  
    diff_ln_params = list(model.diff_ln.parameters()) if hasattr(model, 'diff_ln') else []
    params_to_optimize = (
        prompt_learner_params + 
        adapter_params + 
        cross_attn_params + 
        gamma_text_param + 
        diff_ln_params
    )

    optimizer = torch.optim.Adam(params_to_optimize, lr=args.learning_rate, betas=(0.5, 0.999))

    # optimizer = torch.optim.Adam(list(prompt_learner.parameters()), lr=args.learning_rate, betas=(0.5, 0.999))

    # losses
    loss_focal = FocalLoss()
    loss_dice = BinaryDiceLoss()
    
    
    model.eval()
    prompt_learner.train()
    for epoch in tqdm(range(args.epoch)):
        model.eval()
        prompt_learner.train()
        loss_list = []
        image_loss_list = []

        for items in tqdm(train_dataloader):
            image = items['img'].to(device)
            label =  items['anomaly']

            gt = items['img_mask'].squeeze().to(device)
            gt[gt > 0.5] = 1
            gt[gt <= 0.5] = 0

            image_features, patch_features = model.encode_image(image, args.features_list, DPAM_layer = 20)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                    

            prompts, tokenized_prompts, compound_prompts_text = prompt_learner(cls_id = None)

            text_features = model.encode_text_learn(prompts, tokenized_prompts, compound_prompts_text).float()
            text_features = torch.stack(torch.chunk(text_features, dim = 0, chunks = 2), dim = 1) 
            text_features = text_features/text_features.norm(dim=-1, keepdim=True) 
             
            text_probs = image_features.unsqueeze(1) @ text_features.permute(0, 2, 1) 
            text_probs = text_probs[:, 0, ...]/.007  # N,2
            image_loss = F.cross_entropy(text_probs.squeeze(), label.long().cuda()) 
            image_loss_list.append(image_loss.item())

            similarity_map_list = []
            # similarity_map_list.append(similarity_map)
            for idx, patch_feature in enumerate(patch_features):
                if idx >= args.feature_map_layer[0]:
                    patch_feature = patch_feature/ patch_feature.norm(dim = -1, keepdim = True)
                    similarity, _ = models.compute_similarity(patch_feature, text_features[0])
                    similarity_map = models.get_similarity_map(similarity[:, 1:, :], args.image_size).permute(0, 3, 1, 2)
                    similarity_map_list.append(similarity_map)

            loss = 0
            for i in range(len(similarity_map_list)):
                loss += loss_focal(similarity_map_list[i], gt)
                loss += loss_dice(similarity_map_list[i][:, 1, :, :], gt)
                loss += loss_dice(similarity_map_list[i][:, 0, :, :], 1-gt)

            optimizer.zero_grad()
            (loss+image_loss).backward()   # L = L_global + L_local
            optimizer.step()
            loss_list.append(loss.item())
            
        if (epoch + 1) % args.print_freq == 0:
            logger.info('epoch [{}/{}], loss:{:.4f}, image_loss:{:.4f}'.format(epoch + 1, args.epoch, np.mean(loss_list), np.mean(image_loss_list)))

        if (epoch + 1) % args.save_freq == 0:
            ckp_path = os.path.join(args.save_path, 'epoch_' + str(epoch + 1) + '.pth')
            torch.save({"prompt_learner": prompt_learner.state_dict()}, ckp_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser("DDECLIP", add_help=True)
    parser.add_argument("--train_data_path", type=str, default="./data/visa", help="train dataset path")
    parser.add_argument("--save_path", type=str, default='./checkpoint', help='path to save results')


    parser.add_argument("--dataset", type=str, default='mvtec', help="train dataset name")

    parser.add_argument("--depth", type=int, default=6, help="depth")
    parser.add_argument("--n_ctx", type=int, default=12, help="L")
    parser.add_argument("--t_n_ctx", type=int, default=2, help="k")
    parser.add_argument("--feature_map_layer", type=int, nargs="+", default=[0, 1, 2, 3], help="zero shot")
    parser.add_argument("--features_list", type=int, nargs="+", default=[6, 12, 18, 24], help="features used")

    parser.add_argument("--epoch", type=int, default=15, help="epochs")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="learning rate")
    parser.add_argument("--batch_size", type=int, default=8, help="batch size")
    parser.add_argument("--image_size", type=int, default=518, help="image size")
    parser.add_argument("--print_freq", type=int, default=1, help="print frequency")
    parser.add_argument("--save_freq", type=int, default=1, help="save frequency")
    parser.add_argument("--seed", type=int, default=111, help="random seed")
    args = parser.parse_args()
    setup_seed(args.seed)
    train(args)
