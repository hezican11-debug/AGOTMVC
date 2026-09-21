import argparse
import json
import os
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR

from dataset.dataloader import load_data
from metric import valid
from networks.network import Network
from utils.granular_loss import MultiviewGCLoss
from utils.loss import LTwoLoss, SimLoss
from utils.tools import Logger
from utils.train_epoches import SplitExtract, ViewsFusion


os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")


def getConfig():
    parser = argparse.ArgumentParser(
        description="GFSAF improved-2 minimal hotfix"
    )

    # Keep improved-2 defaults unchanged.
    parser.add_argument("--lambda1", default=0.3, type=float)
    parser.add_argument("--lambda2", default=0.01, type=float)

    parser.add_argument("--dataset", default="WebKB")
    parser.add_argument("--class_num", default=None)
    parser.add_argument("--view", default=3, type=int)
    parser.add_argument("--dims", default=None)
    parser.add_argument("--data_size", default=None, type=int)

    parser.add_argument("--device", default=None)
    parser.add_argument("--backbone", default="AE", choices=["AE", "DAE"])
    parser.add_argument("--iteration", default=4, type=int)
    parser.add_argument("--learning_rate", default=5e-5, type=float)
    parser.add_argument("--weight_decay", default=0.0, type=float)
    parser.add_argument("--ori_epochs", default=100, type=int)
    parser.add_argument("--epochs", default=100, type=int)
    parser.add_argument("--feature_dim", default=256, type=int)
    parser.add_argument("--high_feature_dim", default=64, type=int)
    parser.add_argument("--temperature", default=0.2, type=float)
    parser.add_argument("--batch_size", default=256, type=int)

    parser.add_argument("--gb_min_samples", default=8, type=int)
    parser.add_argument("--gb_max_balls", default=64, type=int)
    parser.add_argument("--gb_min_gain", default=0.08, type=float)
    parser.add_argument("--gb_min_separation", default=0.20, type=float)

    parser.add_argument("--ot_epsilon", default=0.10, type=float)
    parser.add_argument("--ot_iters", default=30, type=int)
    parser.add_argument("--ot_semantic_weight", default=0.55, type=float)
    parser.add_argument("--ot_overlap_weight", default=0.40, type=float)

    parser.add_argument("--gb_positive_topk", default=3, type=int)
    parser.add_argument("--gb_positive_rel_threshold", default=0.25, type=float)

    parser.add_argument("--gb_drift_weight", default=0.05, type=float)
    parser.add_argument("--gb_drift_warmup", default=5, type=int)
    parser.add_argument("--gb_memory_size", default=32, type=int)
    parser.add_argument("--gb_memory_momentum", default=0.95, type=float)

    # ================= Ablation switches =================
    parser.add_argument(
        "--gb_generation",
        default="adaptive",
        choices=["adaptive", "kmeans"],
        help="adaptive=Full Model; kmeans=w/o Adaptive Granular-Ball Generation",
    )
    parser.add_argument(
        "--matching",
        default="ot",
        choices=["ot", "hard"],
        help="ot=Full Model; hard=w/o OT soft correspondence",
    )
    parser.add_argument(
        "--positive_mode",
        default="multi",
        choices=["multi", "single"],
        help="multi=weighted multi-positive; single=strict single-positive ablation",
    )
    parser.add_argument(
        "--use_uncertainty",
        default=1,
        type=int,
        choices=[0, 1],
        help="1=uncertainty-aware weighting; 0=w/o uncertainty",
    )

    # Minimal training-control hotfixes.
    parser.add_argument(
        "--no_reload_best_ses",
        action="store_true",
        help=(
            "Disable reloading the best SES checkpoint before VFS. "
            "Default behavior reloads it."
        ),
    )
    parser.add_argument(
        "--ses_patience",
        default=0,
        type=int,
        help=(
            "Optional SES early stopping patience measured in evaluated epochs. "
            "0 disables early stopping."
        ),
    )

    args = parser.parse_args()
    args.device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    return args


def _safe_float(value):
    return None if value is None else float(value)


def _record_to_metrics(record):
    # max_records layout: [ACC, NMI, ARI, PUR]
    return {
        "ACC": _safe_float(record[0]),
        "NMI": _safe_float(record[1]),
        "ARI": _safe_float(record[2]),
        "PUR": _safe_float(record[3]),
    }


def _json_safe(value):
    if isinstance(value, torch.device):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    return value


def _atomic_write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    tmp.replace(path)


def _format_metrics(record):
    m = _record_to_metrics(record)
    return (
        f"ACC={m['ACC']:.4f} "
        f"NMI={m['NMI']:.4f} "
        f"ARI={m['ARI']:.4f} "
        f"PUR={m['PUR']:.4f}"
    )


def _save_result_json(
    args,
    max_records,
    data_size,
    view,
    class_num,
    elapsed_seconds,
    checkpoint_path,
    logger,
):
    checkpoint_meta = {}
    if Path(checkpoint_path).exists():
        try:
            ckpt = torch.load(
                checkpoint_path,
                map_location="cpu",
            )
            checkpoint_meta = {
                "ori_epochs": ckpt.get("ori_epochs"),
                "Total_epochs": ckpt.get("Total_epochs"),
            }
        except Exception as exc:
            checkpoint_meta = {
                "metadata_read_error": str(exc)
            }

    payload = {
        "status": "success",
        "dataset": args.dataset,
        "finished_at": datetime.now()
        .astimezone()
        .isoformat(timespec="seconds"),
        "metric_scale": "0-1",
        "ablation": {
            "gb_generation": str(getattr(args, "gb_generation", "adaptive")),
            "matching": str(getattr(args, "matching", "ot")),
            "positive_mode": str(getattr(args, "positive_mode", "multi")),
            "use_uncertainty": bool(int(getattr(args, "use_uncertainty", 1))),
            "use_drift": float(getattr(args, "gb_drift_weight", 0.05)) > 0.0,
            "gb_drift_weight": float(getattr(args, "gb_drift_weight", 0.05)),
        },
        "selection_rule": (
            "max_records keeps the highest-ACC evaluation encountered during "
            "training; NMI/ARI/PUR are from that same evaluation."
        ),
        "metrics": {
            "multi": _record_to_metrics(max_records[0]),
            "single": _record_to_metrics(max_records[1]),
            "mean": _record_to_metrics(max_records[2]),
        },
        "dataset_info": {
            "data_size": int(data_size),
            "views": int(view),
            "classes": int(class_num),
            "dims": _json_safe(args.dims),
        },
        "training": {
            "backbone": args.backbone,
            "device": str(args.device),
            "ori_epochs_requested": int(args.ori_epochs),
            "vfs_epochs_per_iteration": int(args.epochs),
            "iteration": int(args.iteration),
            "batch_size": int(args.batch_size),
            "learning_rate": float(args.learning_rate),
            "weight_decay": float(args.weight_decay),
            "reload_best_ses_before_vfs": (
                not bool(args.no_reload_best_ses)
            ),
            "ses_patience": int(args.ses_patience),
            "elapsed_seconds": float(elapsed_seconds),
        },
        "granular_ball": {
            "min_samples": int(args.gb_min_samples),
            "max_balls": int(args.gb_max_balls),
            "min_gain": float(args.gb_min_gain),
            "min_separation": float(args.gb_min_separation),
            "positive_topk": int(args.gb_positive_topk),
            "positive_rel_threshold": float(
                args.gb_positive_rel_threshold
            ),
        },
        "optimal_transport": {
            "epsilon": float(args.ot_epsilon),
            "iterations": int(args.ot_iters),
            "semantic_weight": float(args.ot_semantic_weight),
            "overlap_weight": float(args.ot_overlap_weight),
        },
        "drift_control": {
            "weight": float(args.gb_drift_weight),
            "warmup_epochs": int(args.gb_drift_warmup),
            "memory_size": int(args.gb_memory_size),
            "memory_momentum": float(args.gb_memory_momentum),
        },
        "artifacts": {
            "checkpoint": str(checkpoint_path),
            "checkpoint_meta": checkpoint_meta,
        },
        "args": _json_safe(vars(args)),
    }

    result_path = Path("models") / f"{args.dataset}_results.json"
    _atomic_write_json(result_path, payload)

    print(
        f"[RESULT] JSON saved: {result_path}",
        flush=True,
    )
    print(
        "[RESULT] Multi : " + _format_metrics(max_records[0]),
        flush=True,
    )
    print(
        "[RESULT] Single: " + _format_metrics(max_records[1]),
        flush=True,
    )
    print(
        "[RESULT] Mean  : " + _format_metrics(max_records[2]),
        flush=True,
    )

    logger.log(
        "Multi: " + _format_metrics(max_records[0])
    )
    logger.log(
        "Single: " + _format_metrics(max_records[1])
    )
    logger.log(
        "Mean: " + _format_metrics(max_records[2])
    )
    logger.log(
        f"Result JSON: {result_path}"
    )

    return payload


def main(args):
    device = args.device

    os.makedirs("./logs", exist_ok=True)
    os.makedirs("./models", exist_ok=True)

    now = time.localtime()
    time_str = "{}-{}-{}-{}-{}".format(
        now.tm_year,
        now.tm_mon,
        now.tm_mday,
        now.tm_hour,
        now.tm_min,
    )

    logger = Logger(
        f"./logs/{time_str}.txt",
        "w",
    )
    logger.log(args)

    total_epochs = (
        args.epochs
        * args.iteration
    )

    run_started = time.time()

    for exper_ite in range(1):
        logger.log(
            "experment iteration:{}".format(
                exper_ite + 1
            )
        )

        t1 = time.time()

        (
            dataset,
            dims,
            view,
            data_size,
            class_num,
        ) = load_data(
            args.dataset
        )

        args.dims = dims
        args.data_size = data_size
        args.class_num = class_num
        args.view = view

        logger.log(
            f"data size: {data_size}, "
            f"views: {view}, class: {class_num}"
        )

        data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True,
            drop_last=False,
        )

        criterion_gra = MultiviewGCLoss(
            args=args
        )
        criterion_LTwo = LTwoLoss(
            args.batch_size
        )
        criterion_Sim = SimLoss(
            args.batch_size
        )

        lossFunList = [
            criterion_gra,
            criterion_LTwo,
            criterion_Sim,
        ]

        tsne_fea = None
        labels_vector = None

        model = Network(
            view,
            dims,
            args.feature_dim,
            args.high_feature_dim,
            class_num,
            device,
        ).to(device)

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
        )

        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=args.ori_epochs + total_epochs,
            eta_min=0.0,
        )

        max_records = [
            [0.0, None, None, None],
            [0.0, None, None, None],
            [0.0, None, None, None],
        ]

        checkpoint_path = os.path.join(
            "models",
            f"{args.dataset}_best.pth",
        )

        print("Initial......")
        logger.log("Initial......")

        # ---------------------------------------------------------
        # SES
        # ---------------------------------------------------------
        best_ses_acc = -1.0
        no_improve = 0

        for epoch in range(
            1,
            args.ori_epochs + 1,
        ):
            (
                optimizer,
                lossFunList,
                scheduler,
            ) = SplitExtract(
                epoch,
                args.backbone,
                args,
                model,
                optimizer,
                data_loader,
                lossFunList,
                scheduler,
                logger,
            )

            (
                acc,
                pur,
                save_flag,
                max_records,
                res_fea,
                labels_vector,
            ) = valid(
                model,
                device,
                dataset,
                view,
                data_size,
                class_num,
                max_records=max_records,
            )

            # Track SES progress independently from the global max_records.
            current_best_acc = float(
                max_records[0][0]
            )

            if current_best_acc > best_ses_acc + 1e-12:
                best_ses_acc = current_best_acc
                no_improve = 0
            else:
                no_improve += 1

            if save_flag:
                torch.save(
                    {
                        "ori_epochs": epoch,
                        "Total_epochs": -1,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "max_records": max_records,
                    },
                    checkpoint_path,
                )

                tsne_fea = res_fea

                logger.log(
                    f"mse_epoch: {epoch}, Best Model saved."
                )
                print(
                    f"mse_epoch: {epoch}, Best Model saved."
                )

            if (
                args.ses_patience > 0
                and no_improve >= args.ses_patience
            ):
                print(
                    f"[EARLY STOP] SES stopped at epoch {epoch}; "
                    f"no improvement for {args.ses_patience} evaluations."
                )
                logger.log(
                    f"SES early stop at epoch {epoch}."
                )
                break

        # ---------------------------------------------------------
        # Minimal but important hotfix:
        # VFS must start from the best SES representation, not the
        # degraded last SES epoch.
        # ---------------------------------------------------------
        if (
            not args.no_reload_best_ses
            and os.path.exists(checkpoint_path)
        ):
            best_ckpt = torch.load(
                checkpoint_path,
                map_location=device,
            )

            # Only reload if the current best checkpoint was produced in SES.
            # If a future customized metric/checkpoint wrote a different marker,
            # this remains conservative.
            if best_ckpt.get("ori_epochs", -1) >= 0:
                model.load_state_dict(
                    best_ckpt[
                        "model_state_dict"
                    ]
                )

                print(
                    "[HOTFIX] Reloaded best SES checkpoint "
                    "before Views-Fusion Stage."
                )
                logger.log(
                    "Reloaded best SES checkpoint before VFS."
                )

                # Reset optimizer/scheduler because Adam moments from the
                # degraded last epoch must not be reused with older parameters.
                optimizer = torch.optim.Adam(
                    model.parameters(),
                    lr=args.learning_rate,
                    weight_decay=args.weight_decay,
                )

                scheduler = CosineAnnealingLR(
                    optimizer,
                    T_max=max(1, total_epochs),
                    eta_min=0.0,
                )

        logger.log(
            "=============Views Fusion Stage Start.============="
        )

        iteration = 1
        logger.log(
            f"----------------Iter {iteration}--------------"
        )

        # ---------------------------------------------------------
        # VFS
        # ---------------------------------------------------------
        for epoch in range(
            1,
            total_epochs + 1,
        ):
            (
                optimizer,
                scheduler,
            ) = ViewsFusion(
                epoch,
                args.backbone,
                args,
                model,
                optimizer,
                data_loader,
                scheduler,
                logger,
            )

            if epoch % args.epochs == 0:
                (
                    acc,
                    pur,
                    save_flag,
                    max_records,
                    res_fea,
                    labels_vector,
                ) = valid(
                    model,
                    device,
                    dataset,
                    view,
                    data_size,
                    class_num,
                    max_records=max_records,
                )

                if save_flag:
                    torch.save(
                        {
                            "ori_epochs": -1,
                            "Total_epochs": epoch,
                            "model_state_dict": model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "max_records": max_records,
                        },
                        checkpoint_path,
                    )

                    tsne_fea = res_fea

                    logger.log(
                        f"Total_con_epoch: {epoch}, Best Model saved."
                    )
                    print(
                        f"Total_con_epoch: {epoch}, Best Model saved."
                    )

                if epoch < total_epochs:
                    iteration += 1

                    print(
                        "Iteration "
                        + str(iteration)
                        + ":"
                    )

                    logger.log(
                        "Iteration "
                        + str(iteration)
                        + ":"
                    )

                    optimizer = torch.optim.Adam(
                        [
                            p
                            for p
                            in model.parameters()
                            if p.requires_grad
                        ],
                        lr=args.learning_rate,
                        weight_decay=args.weight_decay,
                    )

        t2 = time.time()

        print(
            "Time cost: "
            + str(t2 - t1)
        )
        print("End......")

        logger.log(
            "Time cost: "
            + str(t2 - t1)
        )
        logger.log("End......")

        # t-SNE source from the globally best evaluation encountered.
        if (
            labels_vector is not None
            and tsne_fea is not None
        ):
            try:
                import scipy.io as scio

                os.makedirs(
                    "./tSNE",
                    exist_ok=True,
                )

                scio.savemat(
                    f"./tSNE/{args.dataset}.mat",
                    {
                        "X": tsne_fea,
                        "Y": np.asarray(labels_vector),
                    },
                )

            except Exception as exc:
                print(
                    f"[WARNING] t-SNE export failed: {exc}"
                )

        # This was missing in the original improved-2 train.py and is why
        # run_pipeline.py marked a fully trained dataset as FAILED.
        _save_result_json(
            args=args,
            max_records=max_records,
            data_size=data_size,
            view=view,
            class_num=class_num,
            elapsed_seconds=(
                time.time() - run_started
            ),
            checkpoint_path=checkpoint_path,
            logger=logger,
        )


if __name__ == "__main__":
    main(getConfig())
