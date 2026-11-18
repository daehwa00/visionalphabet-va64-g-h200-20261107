"""Exact, reviewable evaluation-only transforms of the retained Cconv runtime."""

def replace_once(text,old,new):
 if text.count(old)!=1:raise RuntimeError('expected evaluation patch context not unique')
 return text.replace(old,new,1)


def runtime_source(text):
 text=replace_once(text,'def _contract(task: BaselineTask, device: torch.device) -> dict[str, Any]:','VALIDATION_INTERVAL = 1\n\n\ndef _contract(task: BaselineTask, device: torch.device) -> dict[str, Any]:')
 text=replace_once(text,'        "execution": asdict(task.execution),','        "execution": asdict(task.execution),\n        "evaluation": {"interval_epochs": VALIDATION_INTERVAL, "final_epoch_always": True},')
 old='''        if stop_without_checkpoint:
            validation = _evaluate(model, loaders.validation, task, resolved_device, stop_requested=stop_requested)
        else:
            validation = _evaluate(model, loaders.validation, task, resolved_device)'''
 new='''        validation = None
        if epoch % VALIDATION_INTERVAL == 0 or epoch == task.epochs:
            if stop_without_checkpoint:
                validation = _evaluate(model, loaders.validation, task, resolved_device, stop_requested=stop_requested)
            else:
                validation = _evaluate(model, loaders.validation, task, resolved_device)'''
 text=replace_once(text,old,new)
 text=replace_once(text,'"validation_accuracy": validation["accuracy"],','"validation_accuracy": validation["accuracy"] if validation is not None else None,')
 text=replace_once(text,'"final_validation": _evaluate(model, loaders.validation, task, resolved_device),','"final_validation": history[-1]["validation"],')
 return text


def entry_source(text):
 text=replace_once(text,'def make_task(args: argparse.Namespace) -> runtime.BaselineTask:\n','def make_task(args: argparse.Namespace) -> runtime.BaselineTask:\n    runtime.VALIDATION_INTERVAL = args.validation_interval\n')
 text=replace_once(text,'"execution_backend": args.execution_backend,','"execution_backend": args.execution_backend,\n            "validation_interval": str(args.validation_interval),')
 text=replace_once(text,'        accuracy = row["validation"]["accuracy"]\n        if accuracy > best:','        accuracy = row["validation"]["accuracy"] if row["validation"] is not None else None\n        if accuracy is not None and accuracy > best:')
 text=replace_once(text,'    parser.add_argument("--code", type=Path, default=DEFAULT_CODE)','    parser.add_argument("--validation-interval", type=int, choices=(1,20,100), default=20)\n    parser.add_argument("--code", type=Path, default=DEFAULT_CODE)')
 return text
