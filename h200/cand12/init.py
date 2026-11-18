import json, torch
import run_va_eight_ablations as target
from benchmark_dw_terminal_rmsnorm_training import runner, _install_runner
_install_runner()
torch.manual_seed(501)
model=target._build(target.VARIANT,runner.runtime.model_config())
recipe=json.load(open('contract.json'))['recipe']
model=runner.prepare._prepare_model(model,recipe)
opt=target._build_optimizer(model,recipe)
for g in opt.param_groups:
    g['initial_lr']=g['lr']
torch.save(dict(variant=target.VARIANT,seed=501,epoch=1,global_step=0,model=model.state_dict(),optimizer=opt.state_dict()),'checkpoint.pt')
torch.save(torch.randn(32,3,224,224),'images.pt')
