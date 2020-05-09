from torch.nn.utils import clip_grad_norm_
def OptimStep(model_optim_loss, max_grad):
    for (model, optim, loss, retain_graph) in model_optim_loss:
        model.zero_grad()
        loss.backward(retain_graph=retain_graph)
        clip_grad_norm_(model.parameters(), max_grad)
        optim.step()

def OptimStep_Revise_Grad(model_optim_loss, max_grad, iteration, grad_content = 1
        ,grad_speaker = 1):
    for (model, optim, loss, retain_graph) in model_optim_loss:
        model.zero_grad()
        loss.backward(retain_graph=retain_graph)
        total_norm = 0
        
        clip_grad_norm_(model.quantize_speakers.parameters(), grad_speaker)
        clip_grad_norm_(model.quantize.parameters(), grad_content)
        clip_grad_norm_(model.parameters(), max_grad)
        
        for p in model.quantize.parameters():
            p.grad = p.grad * 0.999**(iteration)
        
        
        optim.step()
        #return total_norm_content, total_norm_speaker