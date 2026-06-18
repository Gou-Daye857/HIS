# module_fl/fl_server.py
import flwr as fl
import argparse
import os
import torch
from collections import OrderedDict
from fl_models import BareMetalUnet
from fl_state import get_fl_status, update_fl_status, append_traffic_log

MODEL_SIZE_MB = 142.5
CHECKPOINT_DIR = "./checkpoints"

class HospitalFedAvg(fl.server.strategy.FedAvg):
    def __init__(self, start_round, total_rounds, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.start_round = start_round
        self.total_rounds = total_rounds

    def configure_fit(self, server_round, parameters, client_manager):
        actual_round = self.start_round + server_round
        clients = client_manager.sample(num_clients=client_manager.num_available())
        num_clients = len(clients)

        downlink_traffic = num_clients * MODEL_SIZE_MB
        status = get_fl_status()
        new_traffic = status.get("total_traffic_mb", 0.0) + downlink_traffic

        update_fl_status({
            "current_round": actual_round,
            "current_stage": f"第 {actual_round} 轮：中枢下发全局参数...",
            "current_node": "边缘节点并发训练中",
            "total_traffic_mb": round(new_traffic, 2)
        })
        append_traffic_log(f"📡 下发全局参数至 {num_clients} 家医院，下行并发流量: {downlink_traffic:.1f} MB")
        return super().configure_fit(server_round, parameters, client_manager)

    def aggregate_fit(self, server_round, results, failures):
        actual_round = self.start_round + server_round
        num_clients = len(results)
        uplink_traffic = num_clients * MODEL_SIZE_MB

        status = get_fl_status()
        new_traffic = status.get("total_traffic_mb", 0.0) + uplink_traffic

        update_fl_status({
            "current_stage": f"第 {actual_round} 轮：接收梯度包，执行聚合...",
            "current_node": "中枢网关",
            "total_traffic_mb": round(new_traffic, 2)
        })
        append_traffic_log(f"📥 成功接收 {num_clients} 家医院本地参数，上行并发流量: {uplink_traffic:.1f} MB")

        # 聚合核心
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, results, failures)

        if aggregated_parameters is not None:
            ndarrays = fl.common.parameters_to_ndarrays(aggregated_parameters)
            model = BareMetalUnet(num_classes=7)
            params_dict = zip(model.state_dict().keys(), ndarrays)
            state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})

            participating_nodes = [str(client.cid) for client, _ in results]

            # 落盘
            os.makedirs(CHECKPOINT_DIR, exist_ok=True)
            round_pth = os.path.join(CHECKPOINT_DIR, f"fl_model_round_{actual_round}.pth")
            torch.save(state_dict, round_pth)
            latest_pth = os.path.join(CHECKPOINT_DIR, "fl_checkpoint_latest.pth")
            torch.save(state_dict, latest_pth)

            # 指标捕获
            losses = [res.metrics.get("loss", 0.0) for _, res in results if res.metrics]
            accs = [res.metrics.get("accuracy", 0.0) for _, res in results if res.metrics]
            dices = [res.metrics.get("dice", 0.0) for _, res in results if res.metrics]

            cur_loss = sum(losses) / len(losses) if losses else 0.5
            cur_acc = sum(accs) / len(accs) if accs else 0.0
            cur_dice = sum(dices) / len(dices) if dices else 0.0

            append_traffic_log(f"✨ 加权聚合完成！当前第 {actual_round} 轮 Loss: {cur_loss:.4f}, Acc: {cur_acc:.4f}, Dice: {cur_dice:.4f}")

            # 数据结构封装回传
            metrics_history = status.get("metrics_history", [])
            while len(metrics_history) < actual_round:
                metrics_history.append({"round": len(metrics_history)+1, "loss": 0.0, "acc": 0.0, "dice": 0.0})

            metrics_history[actual_round - 1] = {
                "round": actual_round, "loss": cur_loss, "acc": cur_acc, "dice": cur_dice
            }

            best_metrics = status.get("best_metrics", {"round": 0, "loss": 999.0, "acc": 0.0, "dice": 0.0})
            if cur_dice > best_metrics.get("dice", 0.0) or (cur_dice == best_metrics.get("dice", 0.0) and cur_loss < best_metrics.get("loss", 999.0)):
                best_metrics = {"round": actual_round, "loss": cur_loss, "acc": cur_acc, "dice": cur_dice}
                torch.save(state_dict, os.path.join(CHECKPOINT_DIR, "fl_model_best.pth"))
                append_traffic_log(f"🏆 模型已刷新历史最优纪录，权重被成功锁定存盘！")

            round_participants = status.get("round_participants", {})
            round_participants[str(actual_round)] = participating_nodes

            update_fl_status({
                "current_round": actual_round,
                "current_stage": "等待下一轮时钟心跳...",
                "metrics_history": metrics_history,
                "best_metrics": best_metrics,
                "round_participants": round_participants,
                "model_saved_path": latest_pth
            })

        return aggregated_parameters, aggregated_metrics

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--min_clients", type=int, default=2)
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()

    status = get_fl_status()
    start_round = status.get("current_round", 0)
    total_rounds = args.rounds

    # 🚨 核心修复：阻断断点盲目+5的问题，只跑剩余轮次！
    remaining_rounds = total_rounds - start_round
    if remaining_rounds <= 0:
        append_traffic_log(f"🛑 目标轮次({total_rounds})已达成。系统退出。")
        update_fl_status({"is_training": False, "current_stage": "联邦训练已完成"})
        return

    initial_parameters = None
    latest_pth = os.path.join(CHECKPOINT_DIR, "fl_checkpoint_latest.pth")
    if start_round > 0 and os.path.exists(latest_pth):
        append_traffic_log(f"🔐 成功从磁盘载入上一次的最新模型参数包。")
        state_dict = torch.load(latest_pth, map_location="cpu")
        ndarrays = [val.cpu().numpy() for val in state_dict.values()]
        initial_parameters = fl.common.ndarrays_to_parameters(ndarrays)

    strategy = HospitalFedAvg(
        start_round=start_round,
        total_rounds=total_rounds,
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=args.min_clients,
        min_available_clients=args.min_clients,
        initial_parameters=initial_parameters
    )

    fl.server.start_server(
        server_address=f"0.0.0.0:{args.port}",
        config=fl.server.ServerConfig(num_rounds=remaining_rounds),
        strategy=strategy,
    )

    update_fl_status({"is_training": False, "current_stage": "联邦训练圆满完成"})
    append_traffic_log("🧹 协同调度收尾，内存与显存已交由系统彻底物理回收。")

if __name__ == "__main__":
    main()