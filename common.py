
def get_dice_coeff(pred, target):
    pred = (pred>0).float()
    return 2.0*(pred*target).sum() / ((pred+target).sum() + 1.0)

def reduce(values):
    return sum(values)/len(values)
