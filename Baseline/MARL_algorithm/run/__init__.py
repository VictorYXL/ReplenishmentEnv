from .whittle_disc_run import run as whittle_disc_run
from .whittle_cont_run import run as whittle_cont_run
from .iql_run import run as iql_run
from .ippo_run import run as ippo_run
from .ippo_with_base_stock_run import run as ippo_with_base_stock_run
from .qtran_run import run as qtran_run
from .qtran_with_base_stock_run import run as qtran_with_base_stock_run
from .simple_multi_echelon_run import run as simple_multi_echelon_run

REGISTRY = {}
REGISTRY["whittle_run"] = whittle_disc_run
REGISTRY["whittle_disc_run"] = whittle_disc_run
REGISTRY["whittle_cont_run"] = whittle_cont_run
REGISTRY["iql_run"] = iql_run
REGISTRY["ippo_run"] = ippo_run
REGISTRY["ippo_with_base_stock_run"] = ippo_with_base_stock_run
REGISTRY["qtran_run"] = qtran_run
REGISTRY["qtran_with_base_stock_run"] = qtran_with_base_stock_run
REGISTRY["qplex_run"] = qtran_run
REGISTRY["simple_multi_echelon_run"] = simple_multi_echelon_run


