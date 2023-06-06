import datetime
import glob
import os
import re
import threading
import time
import copy
from os.path import abspath, dirname
from types import SimpleNamespace as SN
import pandas as pd
import numpy as np
import torch
import pdb

import wandb
from components.episode_buffer import ReplayBuffer
from components.transforms import OneHot
from components.reward_scaler import RewardScaler
from controllers import REGISTRY as mac_REGISTRY
from learners import REGISTRY as le_REGISTRY
from runners import REGISTRY as r_REGISTRY

from utils.logging import Logger
from utils.timehelper import time_left, time_str


def run(_run, _config, _log):

    # check args sanity
    _config = args_sanity_check(_config, _log)

    args = SN(**_config)
    args.device = "cuda" if args.use_cuda else "cpu"

    # setup loggers
    logger = Logger(_log)

    _log.info("Experiment Parameters:")
    tmp_config = {k: _config[k] for k in _config if k != "env_args"}
    tmp_config.update(
        {f"env_agrs.{k}": _config["env_args"][k] for k in _config["env_args"]}
    )
    print(
        pd.Series(tmp_config, name="HyperParameter Value")
        .transpose()
        .sort_index()
        .fillna("")
        .to_markdown()
    )

    # configure tensorboard logger
    ts = datetime.datetime.now().strftime("%m%dT%H%M")
    unique_token = f"{_config['name']}_{_config['env_args']['n_agents']}_{_config['env_args']['task_type']}_seed{_config['seed']}_{ts}"
    args.unique_token = unique_token
    if args.use_tensorboard:
        tb_logs_direc = os.path.join(
            dirname(dirname(dirname(abspath(__file__)))), "results", "tb_logs"
        )
        tb_exp_direc = os.path.join(tb_logs_direc, "{}").format(unique_token)
        logger.setup_tb(tb_exp_direc)

    # sacred is on by default
    logger.setup_sacred(_run)

    # Run and train
    run_sequential(args=args, logger=logger)

    # Clean up after finishing
    print("Exiting Main")

    print("Stopping all threads")
    for t in threading.enumerate():
        if t.name != "MainThread":
            print("Thread {} is alive! Is daemon: {}".format(t.name, t.daemon))
            t.join(timeout=1)
            print("Thread joined")

    print("Exiting script")

    # Making sure framework really exits
    os._exit(os.EX_OK)


def run_sequential(args, logger):

    # Init runner so we can get env info
    runner = r_REGISTRY[args.runner](args=args, logger=logger)

    # Set up schemes and groups here
    env_info = runner.get_env_info()
    args.n_agents = env_info["n_agents"]
    args.n_actions = env_info["n_actions"]
    args.state_shape = env_info["state_shape"]
    args.accumulated_episodes = getattr(args, "accumulated_episodes", None)

    scheme = {
        "state": {"vshape": env_info["state_shape"]},
        "obs": {"vshape": env_info["obs_shape"], "group": "agents"},
        "mean_action": {
            "vshape": (env_info["n_actions"],),
            "group": "agents",
            "dtype": torch.float,
        },
        "actions": {"vshape": (1,), "group": "agents", "dtype": torch.long},
        "avail_actions": {
            "vshape": (env_info["n_actions"],),
            "group": "agents",
            "dtype": torch.int,
        },
        "probs": {
            "vshape": (env_info["n_actions"],),
            "group": "agents",
            "dtype": torch.float,
        },
        "reward": {"vshape": (1,)},
        "lambda": {"vshape": (1,)},
        "individual_rewards": {"vshape": (1,), "group": "agents"},
        "cur_balance": {"vshape": (1,), "group": "agents"},
        "terminated": {"vshape": (1,), "dtype": torch.uint8},
    }
    groups = {"agents": args.n_agents}
    preprocess = {"actions": ("actions_onehot", [OneHot(out_dim=args.n_actions)])}

    buffer = ReplayBuffer(
        scheme,
        groups,
        args.buffer_size,
        env_info["episode_limit"] + 1,
        preprocess=preprocess,
        device="cpu" if args.buffer_cpu_only else args.device,
    )

    logger.console_logger.info("MDP Components:")
    print(pd.DataFrame(buffer.scheme).transpose().sort_index().fillna("").to_markdown())

    # Setup multiagent controller here
    mac = mac_REGISTRY[args.mac](buffer.scheme, groups, args)
    w_mac = mac_REGISTRY[args.w_mac](buffer.scheme, groups, args)
            
    val_args = copy.deepcopy(args)
    val_args.env_args["mode"] = "validation"
    val_runner = r_REGISTRY[args.runner](args=val_args, logger=logger)

    w_val_args = copy.deepcopy(args)
    w_val_args.env_args["mode"] = "validation"
    w_val_runner = r_REGISTRY[args.runner](args=w_val_args, logger=logger)

    test_args = copy.deepcopy(args)
    test_args.env_args["mode"] = "test"
    test_runner = r_REGISTRY[args.runner](args=test_args, logger=logger)

    # Give runner the scheme
    runner.setup(scheme=scheme, groups=groups, preprocess=preprocess, mac=mac)
    val_runner.setup(scheme=scheme, groups=groups, preprocess=preprocess, mac=mac)
    w_val_runner.setup(scheme=scheme, groups=groups, preprocess=preprocess, mac=w_mac)
    test_runner.setup(scheme=scheme, groups=groups, preprocess=preprocess, mac=mac)
    train_cap = args.training_runner_capacity

    if args.visualize:
        visual_runner = r_REGISTRY["episode"](args=args, logger=logger)
        visual_runner.setup(scheme=scheme, groups=groups, preprocess=preprocess, mac=mac)

    # Learner
    learner = le_REGISTRY[args.learner](mac, buffer.scheme, logger, args)
    w_learner = le_REGISTRY[args.w_learner](w_mac, mac, buffer.scheme, logger, args)

    # Reward scaler
    reward_scaler = RewardScaler()

    if args.use_cuda:
        learner.cuda()
        w_learner.cuda()

    if args.checkpoint_path:
        test_runner.mac.load_models(args.checkpoint_path)

        if args.evaluate or args.save_replay:
            vis_save_path = os.path.join(
                args.local_results_path, args.unique_token, "vis"
            ) if os.getenv("AMLT_OUTPUT_DIR") is None else os.path.join(os.getenv("AMLT_OUTPUT_DIR"), "results", args.unique_token, "vis")
            test_runner.run(test_mode=True, visual_outputs_path=vis_save_path)
            test_cur_avg_balances = test_runner.get_overall_avg_balance()
            logger.console_logger.info("test_cur_avg_balances : {}".format(test_cur_avg_balances))
            return

    # Start training
    episode = 0
    last_test_T = 0
    last_log_T = 0
    model_save_time = 0
    visual_time = 0
    max_avg_balance = -1
    test_max_avg_balance = -1
    max_model_path = None

    start_time = time.time()
    last_time = start_time

    logger.console_logger.info("Beginning training for {} timesteps".format(args.t_max))

    # Pre-collect samples to fit reward scaler
    if args.use_reward_normalization:
        episode_batch = runner.run(test_mode=False, storage_capacity=train_cap)
        reward_scaler.fit(episode_batch)

    while runner.t_env <= args.t_max:

        # Step 1: Collect samples
        with torch.no_grad():
            episode_batch = runner.run(test_mode=False, storage_capacity=train_cap)
            if args.use_reward_normalization:
                episode_batch = reward_scaler.transform(episode_batch)
            buffer.insert_episode_batch(episode_batch)

        # Step 2: Train
        if buffer.can_sample(args.batch_size):
            next_episode = episode + args.batch_size_run
            if (
                args.accumulated_episodes
                and next_episode % args.accumulated_episodes != 0
            ):
                continue

            episode_sample = buffer.sample(args.batch_size)

            # Truncate batch to only filled timesteps
            max_ep_t = episode_sample.max_t_filled()
            episode_sample = episode_sample[:, :max_ep_t]

            if episode_sample.device != args.device:
                episode_sample.to(args.device)

            learner.train(episode_sample, runner.t_env, episode)
            w_learner.train(episode_sample, runner.t_env, episode)
            del episode_sample

        # Step 3: Evaluate
        if runner.t_env >= last_test_T + args.test_interval:

            # Log to console
            logger.console_logger.info(
                "t_env: {} / {}".format(runner.t_env, args.t_max)
            )
            logger.console_logger.info(
                "Estimated time left: {}. Time passed: {}".format(
                    time_left(last_time, last_test_T, runner.t_env, args.t_max),
                    time_str(time.time() - start_time),
                )
            )
            last_time = time.time()
            last_test_T = runner.t_env

            # Evaluate the policy executed by argmax for the corresponding Q
            log_dict = []
            wandb_dict = {}

            lbda_list = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
            for lbda in lbda_list:
                for storage_capacity in [20000, 10000, 5000, 2500]:
                    stats, val_lambda_return, val_old_return = val_runner.run(
                        test_mode=True, lbda=lbda, storage_capacity=storage_capacity)
                    w_stats, w_val_lambda_return, w_val_old_return = w_val_runner.run(
                        test_mode=True, lbda=lbda, storage_capacity=storage_capacity)
                    log_dict.append({
                        'step': runner.t_env,
                        'lambda': lbda,
                        'storage_capacity': storage_capacity,
                        'return_lbda': val_lambda_return,
                        'return_old': val_old_return,
                        'max_instock': stats['max_in_stock_sum'],
                        'mean_instock': stats['mean_in_stock_sum'],
                        'w_return_lbda': w_val_lambda_return,
                        'w_return_old': w_val_old_return,
                        'w_max_instock': w_stats['max_in_stock_sum'],
                        'w_mean_instock': w_stats['mean_in_stock_sum'],
                    })

            log_df = pd.DataFrame(log_dict)

            # Record brief performance by running single lambda policy
            log_items = log_df.loc[(log_df['storage_capacity'] == 20000) \
                & log_df['lambda'].apply(lambda x: x in lbda_list)]
            log_items = log_items[['lambda', 'return_lbda', 'return_old', 'max_instock', 'mean_instock',
                'w_return_lbda', 'w_return_old', 'w_max_instock', 'w_mean_instock']].copy()

            log_items.index = log_items['lambda'].apply(lambda x: 'return_lbda_{}'.format(int(x)))
            wandb_dict.update(log_items['return_lbda'].to_dict())
            log_items.index = log_items['lambda'].apply(lambda x: 'return_old_{}'.format(int(x)))
            wandb_dict.update(log_items['return_old'].to_dict())
            log_items.index = log_items['lambda'].apply(lambda x: 'max_instock_{}'.format(int(x)))
            wandb_dict.update(log_items['max_instock'].to_dict())
            log_items.index = log_items['lambda'].apply(lambda x: 'mean_instock_{}'.format(int(x)))
            wandb_dict.update(log_items['mean_instock'].to_dict())

            log_items.index = log_items['lambda'].apply(lambda x: 'w_return_lbda_{}'.format(int(x)))
            wandb_dict.update(log_items['w_return_lbda'].to_dict())
            log_items.index = log_items['lambda'].apply(lambda x: 'w_return_old_{}'.format(int(x)))
            wandb_dict.update(log_items['w_return_old'].to_dict())
            log_items.index = log_items['lambda'].apply(lambda x: 'w_max_instock_{}'.format(int(x)))
            wandb_dict.update(log_items['w_max_instock'].to_dict())
            log_items.index = log_items['lambda'].apply(lambda x: 'w_mean_instock_{}'.format(int(x)))
            wandb_dict.update(log_items['w_mean_instock'].to_dict())

            # Record performance by running Whittle index policies
            metrics = log_df.groupby('storage_capacity')['return_old'].max()
            metrics.index = ['return_old_cap_{}'.format(item) for item in metrics.index]
            wandb_dict.update(metrics.to_dict())

            metrics = log_df.groupby('storage_capacity')['return_lbda'].max()
            metrics.index = ['return_lbda_cap_{}'.format(item) for item in metrics.index]
            wandb_dict.update(metrics.to_dict())

            metrics = log_df.groupby('storage_capacity')['w_return_old'].max()
            metrics.index = ['w_return_old_cap_{}'.format(item) for item in metrics.index]
            wandb_dict.update(metrics.to_dict())

            metrics = log_df.groupby('storage_capacity')['w_return_lbda'].max()
            metrics.index = ['w_return_lbda_cap_{}'.format(item) for item in metrics.index]
            wandb_dict.update(metrics.to_dict())

            if args.use_wandb:
                wandb.log(wandb_dict, step=runner.t_env)
                wandb.log({'eval_detail': log_df})

        # Step 4: Save model
        if args.save_model and (
            runner.t_env - model_save_time >= args.save_model_interval
            or model_save_time == 0
        ):
            model_save_time = runner.t_env
            save_path = os.path.join(
                args.local_results_path, args.unique_token, "models", str(runner.t_env)
            ) if os.getenv("AMLT_OUTPUT_DIR") is None else os.path.join(os.getenv("AMLT_OUTPUT_DIR"), "results", args.unique_token, "models", str(runner.t_env))
            save_path = save_path.replace('*', '_')
            os.makedirs(save_path, exist_ok=True)
            logger.console_logger.info("Saving models to {}".format(save_path))

            # learner should handle saving/loading -- delegate actor save/load to mac,
            # use appropriate filenames to do critics, optimizer states
            learner.save_models(save_path)

        # Step 5: Visualize
        if args.visualize and ((runner.t_env - visual_time) / args.visualize_interval >= 1.0):

            visual_time = runner.t_env
            visual_outputs_path = os.path.join(
                args.local_results_path, args.unique_token, "visual_outputs"
            )
            logger.console_logger.info(
                f"Saving visualizations to {visual_outputs_path}/{runner.t_env}"
            )

            visual_runner.run_visualize(visual_outputs_path, runner.t_env)

        # Step 6: Finalize
        episode += args.batch_size_run
        if (runner.t_env - last_log_T) >= args.log_interval:
            logger.log_stat("episode", episode, runner.t_env)
            logger.print_recent_stats()
            last_log_T = runner.t_env

    # Evaluation at the end of the training
    n_test_runs = max(1, args.test_nepisode // runner.batch_size)
    record = []
    for i_test in range(n_test_runs):

        # Save the final model
        save_path = os.path.join(
            args.local_results_path, args.unique_token, "models", str(runner.t_env)
        ) if os.getenv("AMLT_OUTPUT_DIR") is None else os.path.join(os.getenv("AMLT_OUTPUT_DIR"), "results", args.unique_token, "models", "best")
        save_path = save_path.replace('*', '_')
        os.makedirs(save_path, exist_ok=True)
        logger.console_logger.info("Saving best models to {}".format(save_path))
        learner.save_models(save_path)

        log_record = {}
        test_runner.t_env = runner.t_env
        lbda_list = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        for lbda in lbda_list:
            for storage_capacity in [20000, 10000, 5000, 2500]:
                _, test_lambda_return, test_old_return = test_runner.run(
                    test_mode=True, lbda=lbda, storage_capacity=storage_capacity)
                log_record.update({
                    'val_return_lbda_l{:02.0f}_c{:03d}k'.format(lbda, storage_capacity // 1000): test_lambda_return,
                    'val_return_old_l{:02.0f}_c{:03d}k'.format(lbda, storage_capacity // 1000): test_old_return})

        record.append(log_record)

    record = pd.DataFrame(record)
    if args.use_wandb:
        wandb.log({'test_result_final': record})

    # Close the environments
    runner.close_env()
    val_runner.close_env()
    test_runner.close_env()
    logger.console_logger.info("Finished Training")


def args_sanity_check(config, _log):

    # set CUDA flags
    # config["use_cuda"] = True # Use cuda whenever possible!
    if config["use_cuda"] and not torch.cuda.is_available():
        config["use_cuda"] = False
        _log.warning(
            "CUDA flag use_cuda was switched OFF automatically because no CUDA devices are available!"
        )

    if config["test_nepisode"] < config["batch_size_run"]:
        config["test_nepisode"] = config["batch_size_run"]
    else:
        config["test_nepisode"] = (
            config["test_nepisode"] // config["batch_size_run"]
        ) * config["batch_size_run"]

    return config