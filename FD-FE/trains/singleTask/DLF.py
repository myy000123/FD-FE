import logging
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
from ..utils import MetricsTop, dict_to_str
from .HingeLoss import HingeLoss


logger = logging.getLogger('MMSA')

class MSE(nn.Module):
    def __init__(self):
        super(MSE, self).__init__()

    def forward(self, pred, real):
        diffs = torch.add(real, -pred)
        n = torch.numel(diffs.data)
        mse = torch.sum(diffs.pow(2)) / n
        return mse

class DLF():
    def __init__(self, args):
        self.args = args
        self.criterion = nn.L1Loss()           
        self.cosine = nn.CosineEmbeddingLoss()
        self.metrics = MetricsTop(args.train_mode).getMetics(args.dataset_name)
        self.MSE = MSE()
        self.sim_loss = HingeLoss()

    def do_train(self, model, dataloader, return_epoch_results=False):

        # 0: DLF model
        params = model[0].parameters() 

        optimizer = optim.Adam(params, lr=self.args.learning_rate)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, verbose=True, patience=self.args.patience)

        epochs, best_epoch = 0, 0
        if return_epoch_results:
            epoch_results = {
                'train': [],
                'valid': [],
                'test': []
            }
        min_or_max = 'min' if self.args.KeyEval in ['Loss'] else 'max'
        best_valid = 1e8 if min_or_max == 'min' else 0

        net = []
        net_DLF = model[0]
        net.append(net_DLF)    
        model = net
        
        # 预处理数据，用于初始化和预训练外部记忆
        print('Collecting samples for memory initialization and pretraining...')
        logger.info("Collecting samples for memory initialization and pretraining...")
        text_samples, audio_samples, video_samples = [], [], []
        sample_count = 0
        max_samples = 1000  # 最多收集1000个样本
        
        # 从训练数据中收集样本
        with torch.no_grad():
            for batch_data in dataloader['train']:
                vision = batch_data['vision'].to(self.args.device)
                audio = batch_data['audio'].to(self.args.device)
                text = batch_data['text'].to(self.args.device)
                
                # 使用模型的编码器提取特征
                if model[0].use_bert:
                    text_encoded = model[0].text_model(text)
                else:
                    text_encoded = text
                
                # 应用投影
                text_proj = text_encoded.transpose(1, 2) if model[0].orig_d_l == model[0].d_l else model[0].proj_l(text_encoded.transpose(1, 2))
                audio_proj = audio.transpose(1, 2) if model[0].orig_d_a == model[0].d_a else model[0].proj_a(audio.transpose(1, 2))
                video_proj = vision.transpose(1, 2) if model[0].orig_d_v == model[0].d_v else model[0].proj_v(vision.transpose(1, 2))
                
                # 转换为序列
                text_proj = text_proj.permute(2, 0, 1)
                audio_proj = audio_proj.permute(2, 0, 1)
                video_proj = video_proj.permute(2, 0, 1)
                
                # 应用模态特定编码器
                with torch.no_grad():
                    text_encoded = model[0].encoder_s_l(text_proj)
                    audio_encoded = model[0].encoder_s_a(audio_proj)
                    video_encoded = model[0].encoder_s_v(video_proj)
                    
                    if isinstance(text_encoded, tuple):
                        text_encoded = text_encoded[0]
                    if isinstance(audio_encoded, tuple):
                        audio_encoded = audio_encoded[0]
                    if isinstance(video_encoded, tuple):
                        video_encoded = video_encoded[0]
                
                # 获取最后一层的表示
                text_features = text_encoded[-1]
                audio_features = audio_encoded[-1]
                video_features = video_encoded[-1]
                
                # 添加到样本列表
                text_samples.append(text_features)
                audio_samples.append(audio_features)
                video_samples.append(video_features)
                
                sample_count += text_features.size(0)
                if sample_count >= max_samples:
                    break
        
        # 将样本转换为张量
        text_samples = torch.cat(text_samples, dim=0)
        audio_samples = torch.cat(audio_samples, dim=0)
        video_samples = torch.cat(video_samples, dim=0)
        
        logger.info(f"Collected {text_samples.size(0)} samples for memory initialization")
        
        # 使用K-means聚类初始化记忆
        logger.info("Initializing external memories with K-means clustering...")
        
        def kmeans_clustering(features, n_clusters):
            # 简单的K-means聚类实现
            device = features.device
            features_np = features.cpu().numpy()
            from sklearn.cluster import KMeans
            kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init=10)
            kmeans.fit(features_np)
            centroids = torch.tensor(kmeans.cluster_centers_, dtype=features.dtype).to(device)
            return centroids
        
        # 对每个模态进行聚类，获取中心点作为记忆初始化
        mem_size = 64  # 与模型中定义的记忆大小相同
        text_centroids = kmeans_clustering(text_samples, mem_size)
        audio_centroids = kmeans_clustering(audio_samples, mem_size)
        video_centroids = kmeans_clustering(video_samples, mem_size)
        
        # 初始化外部记忆
        model[0].initialize_external_memories(
            text_vectors=text_centroids,
            audio_vectors=audio_centroids,
            video_vectors=video_centroids
        )
        
        # 预训练外部记忆
        logger.info("Pretraining external memories...")
        model[0].pretrain_memory(
            text_samples=text_samples,
            audio_samples=audio_samples,
            video_samples=video_samples,
            num_epochs=3,  # 预训练3个epoch，myyg改为4
            batch_size=32,
            lr=1e-4
        )
        
        logger.info("Starting main training loop...")##############开始
        while True:
            epochs += 1
            y_pred, y_true = [], []
            for mod in model:
                mod.train()
              

            train_loss = 0.0
            left_epochs = self.args.update_epochs
            with tqdm(dataloader['train']) as td:
                for batch_data in td:

                    if left_epochs == self.args.update_epochs:
                        optimizer.zero_grad()
                    left_epochs -= 1
                    vision = batch_data['vision'].to(self.args.device)
                    audio = batch_data['audio'].to(self.args.device)
                    text = batch_data['text'].to(self.args.device)
                    labels = batch_data['labels']['M'].to(self.args.device)
                    labels = labels.view(-1, 1)


                   
                    output = model[0](text, audio, vision) 

                    # task loss
                    loss_task_all = self.criterion(output['output_logit'], labels)
                    
                    loss_task_l_hetero = self.criterion(output['logits_l_hetero'], labels)  
                    loss_task_v_hetero = self.criterion(output['logits_v_hetero'], labels)
                    loss_task_a_hetero = self.criterion(output['logits_a_hetero'], labels)
                    loss_task_c = self.criterion(output['logits_c'], labels)
                    
                    # total MSA loss L_msa
                    loss_task = 1* (1 * loss_task_all + 1*loss_task_c  + 3 * loss_task_l_hetero + 1*loss_task_v_hetero + 1*loss_task_a_hetero)
                    
                    # reconstruction loss L_r
                    loss_recon_l = self.MSE(output['recon_l'], output['origin_l'])
                    loss_recon_v = self.MSE(output['recon_v'], output['origin_v'])
                    loss_recon_a = self.MSE(output['recon_a'], output['origin_a'])
                    loss_recon = loss_recon_l + loss_recon_v + loss_recon_a

                    # specific loss L_s 
                    loss_sl_slr = self.MSE(output['s_l'].permute(1, 2, 0), output['s_l_r'])
                    loss_sv_slv = self.MSE(output['s_v'].permute(1, 2, 0), output['s_v_r'])
                    loss_sa_sla = self.MSE(output['s_a'].permute(1, 2, 0), output['s_a_r'])
                    loss_s_sr = loss_sl_slr + loss_sv_slv + loss_sa_sla

                    # ort loss L_o
                    if self.args.dataset_name == 'mosi':
                        num = 50
                    elif self.args.dataset_name == 'mosei':
                        num = 10

                    cosine_similarity_s_c_l = self.cosine(output['s_l'].reshape(-1, num), output['c_l'].reshape(-1, num), torch.tensor([-1]).cuda())
                    cosine_similarity_s_c_v = self.cosine(output['s_v'].reshape(-1, num), output['c_v'].reshape(-1, num), torch.tensor([-1]).cuda())
                    cosine_similarity_s_c_a = self.cosine(output['s_a'].reshape(-1, num), output['c_a'].reshape(-1, num), torch.tensor([-1]).cuda())
                    
                    loss_ort = cosine_similarity_s_c_l + cosine_similarity_s_c_v + cosine_similarity_s_c_a

                    # triplet margin loss L_m
                    c_l, c_v, c_a = output['c_l_sim'], output['c_v_sim'], output['c_a_sim']
                    ids, feats = [], []
                    for i in range(labels.size(0)):
                        feats.append(c_l[i].view(1, -1))
                        feats.append(c_v[i].view(1, -1))
                        feats.append(c_a[i].view(1, -1))
                        ids.append(labels[i].view(1, -1))
                        ids.append(labels[i].view(1, -1))
                        ids.append(labels[i].view(1, -1))
                    feats = torch.cat(feats, dim=0)
                    ids = torch.cat(ids, dim=0)
                    loss_sim = self.sim_loss(ids, feats)

                    #overall loss L_DLF
                    combined_loss = loss_task + (loss_s_sr + loss_recon + (loss_sim+loss_ort) * 0.1) * 0.1   
                
                    combined_loss.backward()


                    if self.args.grad_clip != -1.0:
                        params = list(model[0].parameters())  
                                 
                        nn.utils.clip_grad_value_(params, self.args.grad_clip)

                    train_loss += combined_loss.item()
                    

                    y_pred.append(output['output_logit'].cpu())
                    y_true.append(labels.cpu())
                    if not left_epochs:
                        optimizer.step()
                        left_epochs = self.args.update_epochs
                if not left_epochs:
                    # update
                    optimizer.step()
            

            train_loss = train_loss / len(dataloader['train'])
            pred, true = torch.cat(y_pred), torch.cat(y_true)
            train_results = self.metrics(pred, true)
            logger.info(
                f">> Epoch: {epochs} "
                f"TRAIN -({self.args.model_name}) [{epochs - best_epoch}/{epochs}/{self.args.cur_seed}] "
                f">> total_loss: {round(train_loss, 4)} "
                f"{dict_to_str(train_results)}"
            )
            # validation
            val_results = self.do_test(model[0], dataloader['valid'], mode="VAL")
            test_results = self.do_test(model[0], dataloader['test'], mode="TEST")
            cur_valid = val_results[self.args.KeyEval]
            scheduler.step(val_results['Loss'])
            # save each epoch model
            torch.save(model[0].state_dict(), '/home/cqupt/MYY/DLF-main-ronghe/pt/' + str(self.args.dataset_name) + '_' + str(epochs) + '.pth')
            # save best model
            isBetter = cur_valid <= (best_valid - 1e-6) if min_or_max == 'min' else cur_valid >= (best_valid + 1e-6)
            '''if isBetter:
                best_valid, best_epoch = cur_valid, epochs
                # save model
                model_save_path = '/home/cqupt/MYY/DLF-main-ronghe/pt/DLF' + str(self.args.dataset_name)+'.pth'
                torch.save(model[0].state_dict(), model_save_path)'''

            if isBetter: 
                best_valid, best_epoch = cur_valid, epochs
                # 保存模型和配置myy
                save_dict = {
                    'model_state_dict': model[0].state_dict(),
                    'args': vars(self.args)  # 保存训练时使用的所有参数
                }
                model_save_path = '/home/cqupt/MYY/DLF-main-ronghe/pt/DLF' + str(self.args.dataset_name)+'.pth'
                torch.save(save_dict, model_save_path)
                
            if return_epoch_results:
                train_results["Loss"] = train_loss
                epoch_results['train'].append(train_results)
                epoch_results['valid'].append(val_results)
                test_results = self.do_test(model, dataloader['test'], mode="TEST")
                epoch_results['test'].append(test_results)
           
            # early stop
            if epochs - best_epoch >= self.args.early_stop:
                return epoch_results if return_epoch_results else None

    def do_test(self, model, dataloader, mode="VAL", return_sample_results=False):

        model.eval()
        y_pred, y_true = [], []

        eval_loss = 0.0
        if return_sample_results:
            ids, sample_results = [], []
            all_labels = []
            features = {
                "Feature_t": [],
                "Feature_a": [],
                "Feature_v": [],
                "Feature_f": [],
            }

        with torch.no_grad():
            with tqdm(dataloader) as td:
                for batch_data in td:
                    vision = batch_data['vision'].to(self.args.device)
                    audio = batch_data['audio'].to(self.args.device)
                    text = batch_data['text'].to(self.args.device)
                    labels = batch_data['labels']['M'].to(self.args.device)
                    labels = labels.view(-1, 1)
                    output = model(text, audio, vision)
                    loss = self.criterion(output['output_logit'], labels)
                    eval_loss += loss.item()
                    y_pred.append(output['output_logit'].cpu())
                    y_true.append(labels.cpu())

        eval_loss = eval_loss / len(dataloader)
        pred, true = torch.cat(y_pred), torch.cat(y_true)

        eval_results = self.metrics(pred, true)
        eval_results["Loss"] = round(eval_loss, 4)
        logger.info(f"{mode}-({self.args.model_name}) >> {dict_to_str(eval_results)}")

        if return_sample_results:
            eval_results["Ids"] = ids
            eval_results["SResults"] = sample_results
            for k in features.keys():
                features[k] = np.concatenate(features[k], axis=0)
            eval_results['Features'] = features
            eval_results['Labels'] = all_labels

        return eval_results