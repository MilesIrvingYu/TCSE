from evaluation import metrics
from utils import AverageMeter, get_iou
import copy
import numpy
import torch
import os


class Trainer(object):
    def __init__(self, model, optimizer, train_loader, val_loader, save_name, save_step, val_step):
        self.model = model.cuda()
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.save_name = save_name
        self.save_step = save_step
        self.val_step = val_step
        self.epoch = 1
        self.best_score = 0
        self.score = 0
        self.stats = {'loss': AverageMeter(), 'iou': AverageMeter()}

    def train(self, max_epochs, path):
        for epoch in range(self.epoch, max_epochs + 1):
            self.epoch = epoch
            self.train_epoch()
            if self.epoch % self.save_step == 0:
                print('saving checkpoint\n')
                self.save_checkpoint(folder_path=path)
            if self.score > self.best_score:
                print('new best checkpoint, after epoch {}\n'.format(self.epoch))
                self.save_checkpoint(alt_name='best', folder_path=path)
                self.best_score = self.score
        print('finished training!\n', flush=True)

    def train_epoch(self):

        # train
        self.model.train()
        self.cycle_dataset(mode='train')

        # val
        self.model.eval()
        if self.epoch % self.val_step == 0:
            if self.val_loader is not None:
                with torch.no_grad():
                    self.score = self.cycle_dataset(mode='val')

        # update stats
        for stat_value in self.stats.values():
            stat_value.new_epoch()

    def cycle_dataset(self, mode):
        if mode == 'train':
            for vos_data in self.train_loader:
                # L=num_frames
                #imgs = vos_data['center_img'].cuda()
                #flows = vos_data['center_flows'].cuda()

                masks = vos_data['center_masks'].cuda() # [B,1,C,H,W]

                ref_imgs = vos_data['ref_imgs'].cuda() # [B,L,C,H,W]
                ref_flows = vos_data['ref_flows'].cuda()



                B, L, S, _, H, W = masks.size() # 这里的L===1, 只有当前帧一帧，模型一次只返回当前帧的分割结果, 这里的S===1

                # model run
                vos_out = self.model(ref_imgs, ref_flows)  #对应的model也得改
                loss = torch.nn.CrossEntropyLoss()(vos_out['scores'].view(B * L * S, 2, H, W), masks.reshape(B * L * S, H, W))

                # backward
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # loss, iou
                self.stats['loss'].update(loss.detach().cpu().item(), B)
                iou = torch.mean(get_iou(vos_out['scores'].view(B * L * S, 2, H, W), masks.reshape(B * L * S, H, W))[:, 1:])
                self.stats['iou'].update(iou.detach().cpu().item(), B)

            print('[ep{:04d}] loss: {:.5f}, iou: {:.5f}'.format(self.epoch, self.stats['loss'].avg, self.stats['iou'].avg))

        if mode == 'val':
            #这个貌似求的是，所有视频所有帧的平均

            metrics_res = {}
            metrics_res['J'] = []
            metrics_res['F'] = []

            for i, video_data in enumerate(self.val_loader):
                # 从字典中获取数据
                video_name = video_data['video_name'][0]  # DataLoader会将字符串封装成列表
                #center_imgs = video_data['center_imgs']
                #center_flows = video_data['center_flows']

                center_masks = video_data['center_masks'] #[B,L,S,C,H,W],S=1,C=1
                ref_imgs = video_data['ref_imgs']
                ref_flows = video_data['ref_flows']

                # inference
                vos_out = self.model(ref_imgs, ref_flows)
                res_masks = vos_out['masks'][:, 1:-1].squeeze(2)
                gt_masks = center_masks[:, 1:-1].squeeze(2).squeeze(2)#[B,L,H,W]


                B, L, H, W = res_masks.shape
                object_ids = numpy.unique(gt_masks.cpu()).tolist()
                object_ids.remove(0)

                # evaluate output
                all_res_masks = numpy.zeros((len(object_ids), L, H, W))
                all_gt_masks = numpy.zeros((len(object_ids), L, H, W))
                for k in object_ids:
                    res_masks_k = copy.deepcopy(res_masks).cpu().numpy()
                    res_masks_k[res_masks_k != k] = 0
                    res_masks_k[res_masks_k != 0] = 1
                    all_res_masks[k - 1] = res_masks_k[0]
                    gt_masks_k = copy.deepcopy(gt_masks).cpu().numpy()
                    gt_masks_k[gt_masks_k != k] = 0
                    gt_masks_k[gt_masks_k != 0] = 1
                    all_gt_masks[k - 1] = gt_masks_k[0]

                # calculate scores
                j_metrics_res = numpy.zeros(all_gt_masks.shape[:2])
                f_metrics_res = numpy.zeros(all_gt_masks.shape[:2])
                for i in range(all_gt_masks.shape[0]):
                    j_metrics_res[i] = metrics.db_eval_iou(all_gt_masks[i], all_res_masks[i])
                    f_metrics_res[i] = metrics.db_eval_boundary(all_gt_masks[i], all_res_masks[i])
                    [JM, _, _] = metrics.db_statistics(j_metrics_res[i])
                    metrics_res['J'].append(JM)
                    [FM, _, _] = metrics.db_statistics(f_metrics_res[i])
                    metrics_res['F'].append(FM)

                # gather scores
            J, F = metrics_res['J'], metrics_res['F']
            final_mean = (numpy.mean(J) + numpy.mean(F)) / 2.
            print('[ep{:04d}] J&F score: {:.5f}\n'.format(self.epoch, final_mean))
            return final_mean





    def save_checkpoint(self, alt_name=None, folder_path=""):
        save_dir = f"weights/{folder_path}" if folder_path else "weights"
        # 确保目录存在，如果不存在则创建
        os.makedirs(save_dir, exist_ok=True)
        if alt_name is not None:
            file_path = f"{save_dir}/{self.save_name}_{alt_name}.pth"
        else:
            file_path = f"{save_dir}/{self.save_name}_{self.epoch:04d}.pth"
        torch.save(self.model.module.state_dict(), file_path)
