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
    assert is_best+is_last==1, 'only choose save_best or save_last'
    assert order<=3
    if order==1:
        order_str = '1st'
    elif order==2:
        order_str = '2nd'
    elif order==3:
        order_str= '3rd'
    filename = 'checkpoint_'+ ('best_' if is_best==True else 'last_') + order_str+'.pth'
    filename = os.path.join(save_dir, filename)
    torch.save(checkpoint, filename)

def takeFirst(elem):
    return elem[0]

# Jasper's code 改动一下
def save_model(epoch, auc, _save_models_dir, model_dict, max_epoch, topk):
    '''
    根据auc来保存 best 3 models
    :param epoch: 当前的epoch
    :param auc: 当前模型的validation auc
    :param _save_models_dir: save model directory
    :param model_dict: model state
    :param max_epoch:
    :param topk: top3 aucs
    :return:
    '''
    for i,rec in enumerate(topk):
        if auc > rec:
            for j in range(len(topk)-1,i,-1):
                topk[j] = topk[j-1]
                _j, _jm1 = os.path.join(_save_models_dir, f"_best{j+1}.pth"),\
                os.path.join(_save_models_dir, f"_best{j}.pth")
                if  os.path.exists(_jm1):
                    os.rename(_jm1,_j)
            topk[i] = auc
            model_saved_path = os.path.join(_save_models_dir, f"_best{i+1}.pth")
            state_to_save = {'model_state_dict':model_dict, 'auc_dict':auc,'epoch':epoch+1}
            torch.save(state_to_save, model_saved_path)
            print(f'=>Best{i+1} model updated')
            break

    # 最后三个模型才保存
    if epoch in range(max_epoch-3, max_epoch):
        model_saved_path = os.path.join(_save_models_dir, f"_epoch{epoch+1}.pth")
        state_to_save = {'model_state_dict': model_dict, 'auc_dict': auc,'epoch': epoch+1}
        torch.save(state_to_save, model_saved_path)
        print("=>Last3 epoches model updated")
