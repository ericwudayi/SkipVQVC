from torch.nn.utils import clip_grad_norm_
def OptimStep(model_optim_loss, max_grad):
	for (model, optim, loss, retain_graph) in model_optim_loss:
		model.zero_grad()
		loss.backward(retain_graph=retain_graph)
		clip_grad_norm_(model.parameters(), max_grad)
		optim.step()