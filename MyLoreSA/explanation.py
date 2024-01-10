from Lib.LoreSA.explanation import Explanation


class SuperExplanation(Explanation):

    def __init__(self):
        
        super(Explanation, self).__init__()
        
        self.bb_pred_proba = None
        self.c_pred_proba = None
        self.Xc = None
