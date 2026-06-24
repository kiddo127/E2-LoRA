def get_model(model_name, args):
    name = model_name.lower()
    if name == 'e2lora':
        from models.e2lora import Learner
    else:
        assert 0
    return Learner(args)