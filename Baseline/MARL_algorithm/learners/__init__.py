from .q_learner import QLearner
from .coma_learner import COMALearner
from .qtran_learner import QLearner as QTranLearner
from .local_ppo_learner import LocalPPOLearner
from .local_q_learner import LocalQLearner
from .local_ldqn_learner import LocalLDQNLearner
from .ppo_learner import PPOLearner
from .whittle_gradient_learner import WhittleGradientLearner
from .whittle_target_learner import WhittleTargetLearner

REGISTRY = {}

REGISTRY["q_learner"] = QLearner
REGISTRY["coma_learner"] = COMALearner
REGISTRY["qtran_learner"] = QTranLearner
REGISTRY["local_ppo_learner"] = LocalPPOLearner
REGISTRY["local_q_learner"] = LocalQLearner
REGISTRY["local_ldqn_learner"] = LocalLDQNLearner
REGISTRY["ppo_learner"] = PPOLearner
REGISTRY["whittle_targ_learner"] = WhittleTargetLearner
REGISTRY["whittle_grad_learner"] = WhittleGradientLearner