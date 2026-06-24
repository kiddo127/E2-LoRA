from methods.e2lora import Learner


def get_model(model_name, args):
    name = model_name.lower()
    options = {
            'e2lora': Learner,
               }
    return options[name](args)

