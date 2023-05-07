import os
import datetime
from stable_baselines3.common.callbacks import BaseCallback

class DumpBestModelInfo(BaseCallback):
    """ 
    Dump the current best reward metrics to the best model directory
    """

    def __init__(self, model_type: str = "undefined", verbose: int = 0):
        super().__init__(verbose=verbose)
        self._called = False
        self.model_type = model_type

    def _on_step(self) -> bool:
        assert self.parent is not None, "``DumpBestModelInfo`` callback must be used " "with an ``EvalCallback``"
        if self.parent.best_model_save_path is not None:
            os.makedirs(self.parent.best_model_save_path, exist_ok=True)

        fn = os.path.join(self.parent.best_model_save_path,"best_model.csv")
        
        # write header on first call
        if not self._called:
            f = open(fn,"w")
            f.write("Model,Steps,BestMeanReward,ClockTime\n")
            f.close()
            self._called = True

        f = open(fn,"a")
        f.write(self.model_type + "," + str(self.parent.num_timesteps) + "," + str(self.parent.best_mean_reward) + "," + str(datetime.datetime.now()) + "\n")
        f.close()
        return True