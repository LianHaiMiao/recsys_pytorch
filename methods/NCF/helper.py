import numpy as np
import torch


def getHitRatio(gt_item, pred_items):
	if gt_item in pred_items:
		return 1
	return 0


def getNDCG(gt_item, pred_items):
	if gt_item in pred_items:
		index = pred_items.index(gt_item)
		return np.reciprocal(np.log2(index+2))
	return 0


def metrics(model, test_loader, gpu, top_k):
	HR, NDCG = [], []

	for user, item, label in test_loader:
		if gpu:
			user = user.cuda()
			item = item.cuda()

		predictions = model(user, item)
		_, indices = torch.topk(predictions, top_k)

		if gpu:
			recommends = torch.take(
				item, indices).cpu().numpy().tolist()
		else:
			recommends = torch.take(
				item, indices).numpy().tolist()

		gt_item = item[0].item()
		HR.append(getHitRatio(gt_item, recommends))
		NDCG.append(getNDCG(gt_item, recommends))

	return np.mean(HR), np.mean(NDCG)