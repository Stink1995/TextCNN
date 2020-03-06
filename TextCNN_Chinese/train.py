import os
import sys
import torch
from config import Config

def train(train_iter, dev_iter, model,args,config):

    if torch.cuda.is_available(): # 判断是否有GPU，如果有把模型放在GPU上训练，速度质的飞跃
      model.cuda()

    config = Config()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate) # 梯度下降优化器，采用Adam
    steps = 0
    best_acc = 0
    last_step = 0
    model.train()
    for epoch in range(1, args.epochs + 1): 
        for batch in train_iter:
            feature, target = batch.content, batch.label
            if torch.cuda.is_available(): # 如果有GPU将特征更新放在GPU上
              feature,target = feature.cuda(),target.cuda() 
            optimizer.zero_grad() # 将梯度初始化为0，每个batch都是独立训练地，因为每训练一个batch都需要将梯度归零
            logits = model(feature)
            loss = F.cross_entropy(logits, target) # 计算损失函数 采用交叉熵损失函数
            loss.backward()  # 反向传播
            optimizer.step() # 放在loss.backward()后进行参数的更新
            steps += 1 
            if steps % config.steps_show == 0: # 每训练多少步计算一次准确率，我这边是1，可以自己修改
                corrects = (torch.max(logits, 1)[1].view(target.size()).data == target.data).sum() # logits是[128,10],torch.max(logits, 1)也就是选出第一维中概率最大的值，输出为[128,1],torch.max(logits, 1)[1]相当于把每一个样本的预测输出取出来，然后通过view(target.size())平铺成和target一样的size (128,),然后把与target中相同的求和，统计预测正确的数量
                train_acc = 100.0 * corrects / batch.batch_size # 计算每个mini batch中的准确率
                print('steps:{} - loss: {:.6f}  acc:{:.4f}'.format(
                  steps,
                  loss.item(),
                  train_acc))
                
            if steps % config.steps_eval == 0: # 每训练100步进行一次验证
              dev_acc = eval(dev_iter,model)
              if dev_acc > best_acc:
                best_acc = dev_acc
                last_step = steps
                print('Saving best model, acc: {:.4f}%\n'.format(best_acc))
                save(model,config.save_dir, steps)
              else:
                if steps - last_step >= config.early_stopping:
                  print('\n提前停止于 {} steps, acc: {:.4f}%'.format(last_step, best_acc))
                  raise KeyboardInterrupt


def eval(dev_iter,model):
  model.eval()
  corrects, avg_loss = 0, 0
  for batch in dev_iter:
      feature, target = batch.content, batch.label
      if torch.cuda.is_available():
          feature, target = feature.cuda(), target.cuda()
      logits = model(feature)
      loss = F.cross_entropy(logits, target)
      avg_loss += loss.item()
      corrects += (torch.max(logits, 1)
                    [1].view(target.size()).data == target.data).sum()
  size = len(dev_iter.dataset)
  avg_loss /= size
  accuracy = 100.0 * corrects / size
  print('\nEvaluation - loss: {:.6f}  acc: {:.4f}%({}/{}) \n'.format(avg_loss,
                                                                      accuracy,
                                                                      corrects,
                                                                      size))
  return accuracy


  # 定义模型保存函数
def save(model, save_dir, steps):
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    save_path = 'bestmodel_steps{}.pt'.format(steps)
    save_bestmodel_path = os.path.join(save_dir, save_path)
    torch.save(model.state_dict(), save_bestmodel_path)