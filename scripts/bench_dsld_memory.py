import sys, torch
sys.path.insert(0, 'D:/DBLIRST')
from dsld.models.dsld_core import DsldCore
from dsld.train.losses import focal_dice_loss, recon_loss, decouple_loss

m = DsldCore(width=1.0, use_checkpoint=True).cuda().eval()
x = torch.rand(1, 4, 1, 480, 640, device='cuda')
tgt = (torch.rand(1, 4, 1, 480, 640, device='cuda') > 0.999).float()
with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
    out = m(x, quality=torch.ones(1, device='cuda'))
seg = focal_dice_loss(out['logits'], tgt)
rec = recon_loss(out['y_b'], out['x_main'], out['m_tgt'], tgt)
dec = decouple_loss(out['h_t'], out['h_b'])
print('seg', float(seg), 'recon', float(rec), 'dec', float(dec))
print('x_main range', float(out['x_main'].min()), float(out['x_main'].max()))
print('y_b range', float(out['y_b'].min()), float(out['y_b'].max()))
print('logits range', float(out['logits'].min()), float(out['logits'].max()))
print('h_t rms', float(out['h_t'].pow(2).mean().sqrt()))
