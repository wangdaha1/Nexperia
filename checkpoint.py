import os
import torch


def save_checkpoint(save_dir, checkpoint, order: int, is_best=False, is_last=False):
    '''
    save the last and best 3 models (order)
    :param save_dir:
    :param checkpoint: the states of the current saved model
    :param order: 1st/2nd/3rd
    :param is_best:
    :param is_last:
    examples:
    checkpoint_last_1st: the last model
    checkpoint_last_2nd: the second last model
    checkpoint_best_1st: the best model
    checkpoint_best_2nd: the second best model
    '''
    if is_best==True and is_last==True:
        print("Choose one type of this model, best or last?")
        return None
    if order>3:
        print("You should only preserve the best/last three models")
        return None
    order_str = '1st' if order==1 else '2nd' if order==2 else '3rd'
    filename = 'checkpoint_'+ ('best_' if is_best==True else 'last_') + order_str+'.pkl'
    filename = os.path.join(save_dir, filename)
    torch.save(checkpoint, filename)


def takeFirst(elem):
    return elem[0]
